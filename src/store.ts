/**
 * The default local vector store (retrieval-blueprint §4).
 *
 * In-memory, zero-dependency, cosine over the deterministic embedding — the
 * blueprint's offline, dependency-light default. It also holds a per-token
 * INVERTED INDEX (token → the chunk ids that contain it, with a term count) built
 * at upsert time, so a lexical query is answered over the WHOLE index in
 * O(query terms) rather than re-scanning the corpus or a vector-narrowed pool.
 * That inverted index is what keeps an exact-token hit from being lost in hybrid
 * mode (§2: "never loses an exact-token hit").
 *
 * Optionally the store PERSISTS under `.voidrift/cache/retrieval/` (§4): a
 * versioned JSON index, owner-only (0700 dir / 0600 file), written atomically.
 * Each persisted chunk carries its source file's content hash; on load a chunk
 * is trusted ONLY when the caller re-confirms the current source hash matches
 * (`loadVerified`), so a stale or planted index entry for a changed/other file is
 * never returned. A corrupt or wrong-version file is ignored with one warning and
 * rebuilt. Embeddings are L2-normalised upstream, so a dot product is cosine.
 */
import { promises as fs } from "node:fs";
import * as path from "node:path";
import type { ChunkFilter, IndexedChunk, ScoredChunk, VectorStorePort } from "./ports.js";
import { tokenize } from "./embeddings.js";

function dot(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!;
  return s;
}

/** The on-disk index format version. A file of any other version is rebuilt. */
const INDEX_VERSION = 1;
const INDEX_FILE = "index.json";
/** Size cap for the persisted index; above it, oldest paths are dropped first. */
const DEFAULT_MAX_BYTES = 256 * 1024 * 1024;

/**
 * Inverted-index growth caps (security audit, LOW). An adversarial source file
 * with millions of distinct tokens could otherwise grow the in-memory inverted
 * Map without bound. Three fixed ceilings keep it linear and bounded, in code
 * (not config): distinct tokens indexed per chunk, distinct tokens per file
 * (summed across its chunks), and total postings entries over the whole index.
 * When any cap is hit the extra tokens are simply not indexed — the chunk text
 * and embedding are still stored and retrievable by vector/exact id — and a
 * `truncated` flag is set so the pass can surface it on `RETRIEVAL_INDEXED`.
 */
const MAX_DISTINCT_TOKENS_PER_CHUNK = 2_000;
const MAX_DISTINCT_TOKENS_PER_FILE = 20_000;
const MAX_TOTAL_POSTINGS = 5_000_000;

interface PersistedIndex {
  version: number;
  model: string;
  dimensions: number;
  chunks: IndexedChunk[];
}

function passesFilter(c: IndexedChunk, filter?: ChunkFilter): boolean {
  if (filter?.pathPrefix && !c.path.startsWith(filter.pathPrefix)) return false;
  if (filter?.trust && c.trust !== filter.trust) return false;
  return true;
}

export class LocalVectorStore implements VectorStorePort {
  /** chunk id → chunk. */
  private readonly chunks = new Map<string, IndexedChunk>();
  /** path → set of chunk ids, for path-scoped delete. */
  private readonly byPath = new Map<string, Set<string>>();
  /** path → content hash, for incremental change detection (§3). */
  private readonly hashes = new Map<string, string>();
  /** token → (chunk id → term frequency) — the inverted index for lexical scoring. */
  private readonly inverted = new Map<string, Map<string, number>>();
  /** Insertion order of paths, for oldest-first drop when the size cap is hit. */
  private readonly pathOrder: string[] = [];
  /** Distinct tokens indexed per PATH, for the per-file cap (security LOW). */
  private readonly pathTokenCount = new Map<string, number>();
  /** Running total of postings entries across the inverted index. */
  private totalPostings = 0;
  /** Set once any growth cap is hit; surfaced on `RETRIEVAL_INDEXED`. */
  private indexTruncated = false;

  /** True when a growth cap (per-chunk, per-file, or total postings) was hit
   * while indexing, so some tokens were not added to the inverted index. */
  truncated(): boolean {
    return this.indexTruncated;
  }

  async upsert(chunks: IndexedChunk[]): Promise<void> {
    for (const c of chunks) {
      if (!this.chunks.has(c.id)) {
        // new id — index its tokens
      } else {
        this.unindexTokens(c.id);
      }
      this.chunks.set(c.id, c);
      let set = this.byPath.get(c.path);
      if (!set) {
        set = new Set<string>();
        this.byPath.set(c.path, set);
        this.pathOrder.push(c.path);
      }
      set.add(c.id);
      this.hashes.set(c.path, c.contentHash);
      this.indexTokens(c);
    }
  }

  async deleteByPath(path: string): Promise<void> {
    const set = this.byPath.get(path);
    if (set) for (const id of set) {
      this.unindexTokens(id);
      this.chunks.delete(id);
    }
    this.byPath.delete(path);
    this.hashes.delete(path);
    this.pathTokenCount.delete(path);
    const i = this.pathOrder.indexOf(path);
    if (i >= 0) this.pathOrder.splice(i, 1);
  }

  async query(embedding: number[], k: number, filter?: ChunkFilter): Promise<ScoredChunk[]> {
    const hits: ScoredChunk[] = [];
    for (const c of this.chunks.values()) {
      if (!passesFilter(c, filter)) continue;
      hits.push({ ...c, score: dot(embedding, c.embedding) });
    }
    // Deterministic order: score desc, then id asc to break ties stably.
    hits.sort((a, b) => (b.score - a.score) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    return hits.slice(0, Math.max(0, k));
  }

  async lexicalCandidates(queryTokens: string[], k: number, filter?: ChunkFilter): Promise<ScoredChunk[]> {
    const unique = new Set(queryTokens);
    if (unique.size === 0) return [];
    // Walk ONLY the posting lists of the query tokens (O(query terms)), summing
    // per-chunk hits; the score is the fraction of distinct query tokens matched.
    const matched = new Map<string, number>(); // chunk id → distinct query tokens hit
    for (const tok of unique) {
      const postings = this.inverted.get(tok);
      if (!postings) continue;
      for (const id of postings.keys()) matched.set(id, (matched.get(id) ?? 0) + 1);
    }
    const hits: ScoredChunk[] = [];
    for (const [id, n] of matched) {
      const c = this.chunks.get(id);
      if (!c || !passesFilter(c, filter)) continue;
      hits.push({ ...c, score: n / unique.size });
    }
    hits.sort((a, b) => (b.score - a.score) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    return hits.slice(0, Math.max(0, k));
  }

  async count(): Promise<number> {
    return this.chunks.size;
  }

  async hashForPath(path: string): Promise<string | null> {
    return this.hashes.get(path) ?? null;
  }

  async indexedPaths(): Promise<Map<string, string>> {
    return new Map(this.hashes);
  }

  // --- Inverted-index maintenance -------------------------------------------

  private indexTokens(c: IndexedChunk): void {
    const counts = new Map<string, number>();
    for (const tok of tokenize(c.text)) {
      // Per-chunk distinct-token cap: once reached, keep counting frequency for
      // tokens already seen in this chunk but admit no NEW distinct token.
      if (!counts.has(tok) && counts.size >= MAX_DISTINCT_TOKENS_PER_CHUNK) {
        this.indexTruncated = true;
        continue;
      }
      counts.set(tok, (counts.get(tok) ?? 0) + 1);
    }
    let pathTokens = this.pathTokenCount.get(c.path) ?? 0;
    for (const [tok, n] of counts) {
      let postings = this.inverted.get(tok);
      const isNewToken = !postings;
      const addsPosting = !postings || !postings.has(c.id);
      // Per-file distinct-token cap: a token new to this file that would exceed
      // the file budget is skipped (the chunk text/embedding are still stored).
      if (isNewToken && pathTokens >= MAX_DISTINCT_TOKENS_PER_FILE) {
        this.indexTruncated = true;
        continue;
      }
      // Total postings cap: stop adding NEW postings once the ceiling is hit.
      if (addsPosting && this.totalPostings >= MAX_TOTAL_POSTINGS) {
        this.indexTruncated = true;
        continue;
      }
      if (!postings) this.inverted.set(tok, (postings = new Map()));
      if (addsPosting) {
        this.totalPostings++;
        if (isNewToken) pathTokens++;
      }
      postings.set(c.id, n);
    }
    this.pathTokenCount.set(c.path, pathTokens);
  }

  private unindexTokens(id: string): void {
    const c = this.chunks.get(id);
    if (!c) return;
    const freed = new Set<string>();
    for (const tok of new Set(tokenize(c.text))) {
      const postings = this.inverted.get(tok);
      if (!postings) continue;
      if (postings.delete(id)) {
        this.totalPostings--;
        freed.add(tok);
      }
      if (postings.size === 0) this.inverted.delete(tok);
    }
    // Decrement the per-file distinct-token count by the tokens this chunk freed.
    if (freed.size > 0) {
      const remaining = Math.max(0, (this.pathTokenCount.get(c.path) ?? 0) - freed.size);
      this.pathTokenCount.set(c.path, remaining);
    }
  }

  // --- Persistence (§4) ------------------------------------------------------

  /**
   * Load a persisted index from `dir`, trusting a chunk ONLY when `currentHash`
   * confirms the source file's current content hash matches the chunk's stored
   * hash (a stale or planted chunk for a changed/other file is dropped). A
   * missing/corrupt/wrong-version file is ignored (one warning via `onWarn`) and
   * the store stays empty for a full rebuild. `expect` guards the embeddings
   * identity: a different model/dimension invalidates the whole file (§4).
   * Returns the count of TRUSTED paths loaded.
   */
  async loadVerified(
    dir: string,
    currentHash: (path: string) => string | null | undefined,
    expect: { model: string; dimensions: number },
    onWarn?: (msg: string) => void,
  ): Promise<number> {
    let raw: string;
    try {
      raw = await fs.readFile(path.join(dir, INDEX_FILE), "utf8");
    } catch {
      return 0; // no index yet — a clean first run
    }
    let parsed: PersistedIndex;
    try {
      parsed = JSON.parse(raw) as PersistedIndex;
      if (
        !parsed ||
        parsed.version !== INDEX_VERSION ||
        parsed.model !== expect.model ||
        parsed.dimensions !== expect.dimensions ||
        !Array.isArray(parsed.chunks)
      ) {
        throw new Error("version/model mismatch");
      }
    } catch (e) {
      onWarn?.(`retrieval: ignoring unreadable/incompatible index at ${dir} (${String((e as Error).message)}); rebuilding`);
      return 0;
    }
    const trusted: IndexedChunk[] = [];
    const seen = new Set<string>();
    for (const c of parsed.chunks) {
      // Only trust a chunk whose source file's CURRENT hash matches the stored
      // one — defeats a stale/planted index in a cloned repo (FX-I).
      if (currentHash(c.path) !== c.contentHash) continue;
      trusted.push(c);
      seen.add(c.path);
    }
    await this.upsert(trusted);
    return seen.size;
  }

  /**
   * Persist the current index to `dir/index.json`, owner-only and atomic (§4).
   * Enforces a byte cap by dropping oldest paths first until it fits. Swallows a
   * write error (persistence is best-effort; the index rebuilds next run).
   */
  async persist(dir: string, model: string, dimensions: number, maxBytes = DEFAULT_MAX_BYTES): Promise<void> {
    try {
      await fs.mkdir(dir, { recursive: true, mode: 0o700 });
      let chunks = [...this.chunks.values()];
      let payload = this.serialize(chunks, model, dimensions);
      // Oldest-first drop until under the cap.
      let order = [...this.pathOrder];
      while (Buffer.byteLength(payload, "utf8") > maxBytes && order.length > 0) {
        const drop = new Set(order.splice(0, Math.max(1, Math.ceil(order.length / 10))));
        chunks = chunks.filter((c) => !drop.has(c.path));
        payload = this.serialize(chunks, model, dimensions);
      }
      const tmp = path.join(dir, `${INDEX_FILE}.${process.pid}.tmp`);
      await fs.writeFile(tmp, payload, { mode: 0o600 });
      await fs.rename(tmp, path.join(dir, INDEX_FILE));
    } catch {
      // best-effort — a failed persist just means a rebuild next run.
    }
  }

  private serialize(chunks: IndexedChunk[], model: string, dimensions: number): string {
    const doc: PersistedIndex = { version: INDEX_VERSION, model, dimensions, chunks };
    return JSON.stringify(doc);
  }
}

export { INDEX_VERSION, INDEX_FILE };

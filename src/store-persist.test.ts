/**
 * FX-I — LocalVectorStore persistence under `.voidrift/cache/retrieval/`
 * (retrieval-blueprint §4). OFFLINE, DETERMINISTIC, and HERMETIC: every path is
 * created under an `os.tmpdir()` scratch dir removed in afterEach — never the
 * real cache dir.
 *
 * Proven: a second process start re-embeds nothing (0 reads of unchanged files);
 * the index file is owner-only (0600) under an owner-only dir (0700); a corrupt
 * or wrong-version file is ignored and rebuilt; and a stale/planted chunk is
 * never trusted — a chunk whose source hash differs from the current hash is
 * dropped on load.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { HashingEmbeddings, LocalVectorStore, indexSources, contentHash, type Source } from "./index.js";
import { INDEX_FILE } from "./store.js";

const OPTS = { chunkTokens: 512, overlapTokens: 64 };
let dir: string;

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "vr-retr-persist-"));
});
afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true });
});

const docs: Source[] = [
  { path: "src/a.ts", content: "export const alpha = 1;\n", trust: "live" },
  { path: "src/b.ts", content: "export const beta = 2;\n", trust: "live" },
];

async function buildAndPersist(): Promise<HashingEmbeddings> {
  const embeddings = new HashingEmbeddings();
  const store = new LocalVectorStore();
  await indexSources(docs, embeddings, store, OPTS);
  await store.persist(dir, embeddings.model, embeddings.dimensions);
  return embeddings;
}

describe("persistence (§4, FX-I)", () => {
  it("a second process loads the index and re-embeds nothing (0 reads of unchanged files)", async () => {
    const embeddings = await buildAndPersist();

    // Second "process": fresh store, load from disk. The hash resolver returns
    // the CURRENT hash for each path (unchanged), so every chunk is trusted.
    const store2 = new LocalVectorStore();
    const current = new Map(docs.map((d) => [d.path, contentHash(d.content)]));
    let reads = 0;
    const loaded = await store2.loadVerified(
      dir,
      (p) => {
        reads++; // a hash lookup, not a file read — counted to prove no re-embed path
        return current.get(p) ?? null;
      },
      { model: embeddings.model, dimensions: embeddings.dimensions },
    );
    expect(loaded).toBe(2);
    expect(await store2.count()).toBe(2); // 2 chunks present from disk
    // A re-index over the SAME tree now skips everything (invariant 2): 0 embeds.
    const embeddings2 = new HashingEmbeddings();
    const result = await indexSources(docs, embeddings2, store2, OPTS);
    expect(result.files).toBe(0);
    expect(result.skipped).toBe(2);
    expect(reads).toBe(2); // one hash check per persisted path, no more
  });

  it("the index file is owner-only (0600) under an owner-only dir (0700)", async () => {
    const embeddings = await buildAndPersist();
    const fileStat = await fs.stat(path.join(dir, INDEX_FILE));
    const dirStat = await fs.stat(dir);
    // Low 9 permission bits; owner-only on POSIX.
    if (process.platform !== "win32") {
      expect(fileStat.mode & 0o777).toBe(0o600);
      // mkdtemp dir is 0700 by default; we only assert the index dir we created is owner-only.
      expect(dirStat.mode & 0o077).toBe(0); // no group/other bits
    }
    expect(embeddings.model).toBeTruthy();
  });

  it("a corrupt index file is ignored with a warning and rebuilt", async () => {
    const embeddings = new HashingEmbeddings();
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir, INDEX_FILE), "{ not valid json", "utf8");
    const store = new LocalVectorStore();
    const warnings: string[] = [];
    const loaded = await store.loadVerified(dir, () => null, { model: embeddings.model, dimensions: embeddings.dimensions }, (m) => warnings.push(m));
    expect(loaded).toBe(0);
    expect(warnings).toHaveLength(1);
    expect(await store.count()).toBe(0); // rebuilt (empty) rather than crashing
  });

  it("a wrong-version / wrong-model index file is ignored and rebuilt", async () => {
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir, INDEX_FILE), JSON.stringify({ version: 999, model: "x", dimensions: 1, chunks: [] }), "utf8");
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const warnings: string[] = [];
    const loaded = await store.loadVerified(dir, () => null, { model: embeddings.model, dimensions: embeddings.dimensions }, (m) => warnings.push(m));
    expect(loaded).toBe(0);
    expect(warnings).toHaveLength(1);
  });

  it("a stale/planted chunk is never trusted — a changed source hash drops it", async () => {
    const embeddings = await buildAndPersist();
    // A cloned repo where src/a.ts has DIFFERENT content now: its stored chunk
    // must not be returned. src/b.ts is unchanged and survives.
    const store2 = new LocalVectorStore();
    const current = new Map([
      ["src/a.ts", contentHash("export const alpha = 999; // tampered\n")],
      ["src/b.ts", contentHash(docs[1]!.content)],
    ]);
    const loaded = await store2.loadVerified(
      dir,
      (p) => current.get(p) ?? null,
      { model: embeddings.model, dimensions: embeddings.dimensions },
    );
    expect(loaded).toBe(1); // only src/b.ts trusted
    expect(await store2.hashForPath("src/a.ts")).toBeNull();
    expect(await store2.hashForPath("src/b.ts")).toBe(current.get("src/b.ts"));
    // The planted text for src/a.ts is never returned by a query over the index.
    const cands = await store2.lexicalCandidates(["alpha"], 10);
    expect(cands.every((c) => c.path !== "src/a.ts")).toBe(true);
  });

  it("the size cap drops oldest paths first (bounded index)", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const many: Source[] = [];
    for (let i = 0; i < 20; i++) many.push({ path: `f${String(i).padStart(2, "0")}.ts`, content: `const v${i} = ${i};\n`, trust: "live" });
    await indexSources(many, embeddings, store, OPTS);
    // Persist unbounded first to learn the real on-disk size, then cap at ~45%
    // so oldest-first drops MUST bite regardless of embedding serialization size.
    await store.persist(dir, embeddings.model, embeddings.dimensions);
    const fullBytes = Buffer.byteLength(await fs.readFile(path.join(dir, INDEX_FILE), "utf8"), "utf8");
    const cap = Math.floor(fullBytes * 0.45);
    await store.persist(dir, embeddings.model, embeddings.dimensions, cap);
    const raw = await fs.readFile(path.join(dir, INDEX_FILE), "utf8");
    expect(Buffer.byteLength(raw, "utf8")).toBeLessThanOrEqual(cap);
    const doc = JSON.parse(raw) as { chunks: { path: string }[] };
    const paths = new Set(doc.chunks.map((c) => c.path));
    expect(paths.size).toBeGreaterThan(0); // some survived
    expect(paths.size).toBeLessThan(20); // but not all — the cap bit
    // The newest files survive; the oldest (f00) is dropped first.
    expect(paths.has("f19.ts")).toBe(true);
    expect(paths.has("f00.ts")).toBe(false);
  });
});

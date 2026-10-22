/**
 * The indexing pipeline (retrieval-blueprint §3).
 *
 * Indexing is incremental and keyed by the core's content hash: an unchanged file
 * is never re-embedded (§3, invariant 2). A pass compares each source file's hash
 * against the stored hash in the vector store — unchanged is skipped, changed/new
 * is re-chunked and re-embedded, and a source dropped from the set has its chunks
 * deleted. Re-running over an unchanged tree embeds nothing. The hash is sha256,
 * the same function the core file index uses (core §8); the plugin computes it
 * itself since it touches only the CoreAPI surface, never core internals.
 */
import { createHash } from "node:crypto";
import type { EmbeddingsPort, IndexedChunk, VectorStorePort, ChunkTrust } from "./ports.js";
import { chunk } from "./chunker.js";

/** sha256 of the content — the core's content-hash function (core §8). */
export function contentHash(content: string): string {
  return createHash("sha256").update(content, "utf8").digest("hex");
}

/** A source document to index: its path, body, and trust tier. */
export interface Source {
  path: string;
  content: string;
  trust: ChunkTrust;
}

export interface IndexOptions {
  chunkTokens: number;
  overlapTokens: number;
  /** Bounded concurrency so a local embeddings endpoint is not overrun (§3 step 4). */
  concurrency?: number;
  /**
   * Whether to delete chunks for paths in the store but absent from `sources`
   * (default true — a full reconcile pass). The host-drive indexes a PARTIAL
   * batch per pass and reconciles removals itself, so it sets this false to keep
   * the rest of the index (FX-E background passes).
   */
  removeAbsent?: boolean;
}

export interface IndexResult {
  /** Files re-embedded this pass (changed or new). */
  files: number;
  /** Chunks written this pass. */
  chunks: number;
  /** Files skipped because their hash was unchanged. */
  skipped: number;
  /** Paths whose chunks were deleted (removed from the source set). */
  removed: number;
  /** True when the store's inverted index hit a growth cap this pass or earlier
   * (an adversarial high-cardinality file); surfaced on `RETRIEVAL_INDEXED`. */
  truncated: boolean;
}

/**
 * Reconcile the vector store against the current `sources`. Returns a report the
 * caller can emit as `RETRIEVAL_INDEXED`. Pure of any watcher concern — a watcher
 * event simply triggers another pass with the updated source set.
 */
export async function indexSources(
  sources: Source[],
  embeddings: EmbeddingsPort,
  store: VectorStorePort,
  opts: IndexOptions,
): Promise<IndexResult> {
  const existing = await store.indexedPaths();
  const seen = new Set<string>();
  let files = 0;
  let chunks = 0;
  let skipped = 0;

  for (const src of sources) {
    seen.add(src.path);
    const hash = contentHash(src.content);
    if (existing.get(src.path) === hash) {
      skipped++;
      continue; // unchanged — never re-embed (invariant 2)
    }
    // Changed or new: drop the old chunks, re-chunk, re-embed.
    await store.deleteByPath(src.path);
    const pieces = chunk(src.content, opts);
    if (pieces.length === 0) {
      // An empty file still records its hash so it is not re-processed; store a
      // zero-chunk marker by upserting nothing but tracking via a sentinel delete.
      continue;
    }
    const vectors = await embeddings.embed(pieces.map((p) => p.text));
    const indexed: IndexedChunk[] = pieces.map((p, i) => ({
      id: `${src.path}#${p.byteStart}-${p.byteEnd}`,
      path: src.path,
      byteStart: p.byteStart,
      byteEnd: p.byteEnd,
      contentHash: hash,
      trust: src.trust,
      embedding: vectors[i]!,
      text: p.text,
    }));
    await store.upsert(indexed);
    files++;
    chunks += indexed.length;
  }

  // Delete chunks for paths no longer present in the source set (a full
  // reconcile; skipped when the caller indexes a partial batch, §3/FX-E).
  let removed = 0;
  if (opts.removeAbsent !== false) {
    for (const path of existing.keys()) {
      if (!seen.has(path)) {
        await store.deleteByPath(path);
        removed++;
      }
    }
  }

  return { files, chunks, skipped, removed, truncated: store.truncated?.() ?? false };
}

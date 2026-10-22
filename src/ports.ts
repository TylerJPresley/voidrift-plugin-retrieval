/**
 * Retrieval ports (retrieval-blueprint §4).
 *
 * Retrieval persists behind two ports exactly as memory persists behind the
 * durable store port (core §6.1, §6.8): a local-default backend ships and a
 * hosted backend plugs in behind the same seam with no change above it. The
 * modes (§2), the chunker (§3), and the context integration (§5) know nothing
 * about which backend holds the vectors — the ports are the only coupling.
 */

/** The trust tier a chunk's source carries, mirroring the document registry (core §4.5). */
export type ChunkTrust = "live" | "snapshot";

/** A chunk of a source file with its embedding and provenance (§4). */
export interface IndexedChunk {
  /** Stable id: `${path}#${byteStart}-${byteEnd}`. */
  id: string;
  /** Source file, workspace-relative or snapshot absolute. */
  path: string;
  byteStart: number;
  byteEnd: number;
  /** The core file index's content hash for the source (core §8). */
  contentHash: string;
  trust: ChunkTrust;
  embedding: number[];
  /** The chunk body, returned as the snippet. */
  text: string;
}

export interface ScoredChunk extends IndexedChunk {
  score: number;
}

/** A query-time filter over chunks (e.g. restrict to a path prefix). */
export interface ChunkFilter {
  pathPrefix?: string;
  trust?: ChunkTrust;
}

/**
 * The embeddings port. The default is a deterministic offline embedding; a real
 * backend (an OpenAI-compatible `/embeddings` endpoint via LangChain) plugs in
 * behind this seam by config alone (§4, §7). `dimensions` is fixed per model and
 * fixes the vector width; changing the model invalidates and rebuilds the index.
 */
export interface EmbeddingsPort {
  /** Backend/model identity, recorded on each chunk. */
  readonly model: string;
  /** Embedding width, fixed per configured model. */
  readonly dimensions: number;
  /** Batch embed, order-preserving. */
  embed(texts: string[]): Promise<number[][]>;
}

/**
 * The vector-store port. The default is a local in-memory/embedded store; a
 * hosted vector database plugs in behind this seam by `retrieval.store` config
 * alone, with no change to the indexing pipeline or the context seam above it.
 */
export interface VectorStorePort {
  /** Add or replace by chunk id. */
  upsert(chunks: IndexedChunk[]): Promise<void>;
  /** Drop a removed/changed file's chunks. */
  deleteByPath(path: string): Promise<void>;
  /** Nearest-k by cosine over the embedding, optionally filtered. */
  query(embedding: number[], k: number, filter?: ChunkFilter): Promise<ScoredChunk[]>;
  /**
   * Lexical candidates for a set of query tokens over the WHOLE index (§2), not
   * a vector-narrowed pool — the fix for "never loses an exact-token hit". The
   * backend answers from an inverted/term-frequency index built at upsert time,
   * so this is O(query terms), not O(corpus). `score` is a BM25-style keyword
   * overlap. Returns up to `k` chunks that contain at least one query token,
   * ranked by overlap then id (stable). Optional `filter` applies as in `query`.
   */
  lexicalCandidates(queryTokens: string[], k: number, filter?: ChunkFilter): Promise<ScoredChunk[]>;
  /** Total chunk count — used to detect an empty index. */
  count(): Promise<number>;
  /** The content hash stored for a path, or null if unindexed (§3 change detection). */
  hashForPath(path: string): Promise<string | null>;
  /** Every indexed path with its stored hash — for incremental reconciliation (§3). */
  indexedPaths(): Promise<Map<string, string>>;
  /**
   * True when an inverted-index growth cap was hit (an adversarial file with a
   * huge distinct-token count), so some tokens were not indexed for lexical
   * matching (their chunks are still stored and retrievable by vector). Optional:
   * a hosted backend with its own bounded index may omit it. Surfaced on
   * `RETRIEVAL_INDEXED` as `truncated`.
   */
  truncated?(): boolean;
}

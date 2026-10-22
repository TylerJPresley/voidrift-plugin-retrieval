/**
 * The query path and the three modes (retrieval-blueprint §2, §6).
 *
 * One path resolves to the configured mode; a tool call may request a specific
 * mode where config allows. Vector matches query embedding against chunk
 * embeddings by cosine. Hybrid fuses a lexical rank (keyword overlap over the
 * same tokenizer the embedding uses — a BM25-style signal from the local index)
 * with the vector rank by reciprocal-rank fusion, so an exact-token hit is never
 * lost the way pure vector can lose it. Graph is delegated to the knowledge graph
 * (§6). The modes know nothing of which backend holds the vectors (§4).
 */
import type { EmbeddingsPort, ScoredChunk, VectorStorePort, ChunkFilter } from "./ports.js";
import { tokenize } from "./embeddings.js";

export type RetrievalMode = "vector" | "hybrid" | "graph";

export interface Hit {
  path: string;
  snippet: string;
  score: number;
  trust: "live" | "snapshot";
  contentHash: string;
}

const RRF_K = 60; // standard reciprocal-rank-fusion constant

function toHit(c: ScoredChunk): Hit {
  return { path: c.path, snippet: c.text, score: c.score, trust: c.trust, contentHash: c.contentHash };
}

export class Retriever {
  constructor(
    private readonly embeddings: EmbeddingsPort,
    private readonly store: VectorStorePort,
  ) {}

  /** Vector mode: cosine over the query embedding (§2). */
  async vector(query: string, k: number, filter?: ChunkFilter): Promise<Hit[]> {
    const [embedding] = await this.embeddings.embed([query]);
    const scored = await this.store.query(embedding!, k, filter);
    return scored.map(toHit);
  }

  /**
   * Hybrid mode: reciprocal-rank fusion of the vector rank and a lexical rank,
   * where the lexical rank is computed over the WHOLE index (an inverted index
   * in the store, §2), not a vector-narrowed pool. The candidate set is the
   * UNION of the two ranks, so an exact-token hit the weak offline embedder
   * ranks outside its top-`pool` is still scored and returned — "never loses an
   * exact-token hit".
   */
  async hybrid(query: string, k: number, filter?: ChunkFilter): Promise<Hit[]> {
    const pool = Math.max(k * 4, 20);
    const queryTokens = [...new Set(tokenize(query))];
    const [embedding] = await this.embeddings.embed([query]);
    const [vectorRanked, lexRanked] = await Promise.all([
      this.store.query(embedding!, pool, filter),
      this.store.lexicalCandidates(queryTokens, pool, filter),
    ]);

    // Keep a canonical chunk per id for the merged output (either source).
    const byId = new Map<string, ScoredChunk>();
    for (const c of vectorRanked) byId.set(c.id, c);
    for (const c of lexRanked) if (!byId.has(c.id)) byId.set(c.id, c);

    const rrf = new Map<string, number>();
    const add = (ranked: ScoredChunk[]): void => {
      ranked.forEach((c, i) => {
        rrf.set(c.id, (rrf.get(c.id) ?? 0) + 1 / (RRF_K + i + 1));
      });
    };
    add(vectorRanked);
    add(lexRanked);

    return [...rrf.entries()]
      .map(([id, score]) => ({ chunk: byId.get(id)!, score }))
      .sort((a, b) => (b.score - a.score) || (a.chunk.id < b.chunk.id ? -1 : 1))
      .slice(0, k)
      .map(({ chunk, score }) => ({ ...toHit(chunk), score }));
  }

  /** Resolve a mode to its hits. Graph is handled by the caller via KnowledgeGraph. */
  async search(mode: Exclude<RetrievalMode, "graph">, query: string, k: number, filter?: ChunkFilter): Promise<Hit[]> {
    return mode === "vector" ? this.vector(query, k, filter) : this.hybrid(query, k, filter);
  }
}

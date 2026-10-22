/**
 * The default offline embeddings backend (retrieval-blueprint §4).
 *
 * A deterministic hashing bag-of-words embedding: no network, no model, no cloud
 * dependency, so the plugin and its tests run fully offline (§4, AGENTS.md
 * testing). Each token is hashed into one of `dimensions` buckets and its count
 * accumulated, then the vector is L2-normalised so cosine similarity reduces to a
 * dot product. Two texts that share vocabulary land near each other — enough for
 * deterministic ranking without a learned model. A real embeddings endpoint
 * (OpenAI-compatible `/embeddings` via LangChain) plugs in behind EmbeddingsPort
 * by config; this is the local default the blueprint's offline-first rule requires.
 */
import { createHash } from "node:crypto";
import type { EmbeddingsPort } from "./ports.js";

const DEFAULT_DIMENSIONS = 256;

/** Lowercase alphanumeric word tokens; the one tokenizer both query and index use. */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
}

/** Stable bucket for a token in `[0, dimensions)`, from a sha256 prefix. */
function bucket(token: string, dimensions: number): number {
  const h = createHash("sha256").update(token, "utf8").digest();
  // First 4 bytes as an unsigned int, modulo the width — deterministic across runs.
  const n = (h[0]! << 24) | (h[1]! << 16) | (h[2]! << 8) | h[3]!;
  return (n >>> 0) % dimensions;
}

export class HashingEmbeddings implements EmbeddingsPort {
  readonly model: string;
  readonly dimensions: number;

  constructor(opts: { dimensions?: number; model?: string } = {}) {
    this.dimensions = opts.dimensions ?? DEFAULT_DIMENSIONS;
    this.model = opts.model ?? `hashing-bow-${this.dimensions}`;
  }

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.embedOne(t));
  }

  private embedOne(text: string): number[] {
    const vec = new Array<number>(this.dimensions).fill(0);
    for (const tok of tokenize(text)) {
      const b = bucket(tok, this.dimensions);
      vec[b] = (vec[b] ?? 0) + 1;
    }
    // L2-normalise so cosine == dot product; a zero vector stays zero.
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm === 0) return vec;
    for (let i = 0; i < vec.length; i++) vec[i] = (vec[i] ?? 0) / norm;
    return vec;
  }
}

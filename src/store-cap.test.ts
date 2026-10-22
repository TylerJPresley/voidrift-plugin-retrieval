/**
 * Inverted-index growth caps (security audit, LOW). An adversarial source file
 * with a huge distinct-token count must not grow the in-memory inverted Map
 * without bound. The store caps distinct tokens per chunk (2,000), per file
 * (20,000), and total postings (5,000,000), stops adding postings past a cap,
 * and sets a `truncated` flag surfaced on `RETRIEVAL_INDEXED`.
 *
 * OFFLINE and DETERMINISTIC: generated token streams, no network, no fs.
 */
import { describe, it, expect } from "vitest";
import { LocalVectorStore } from "./store.js";
import { HashingEmbeddings, indexSources, type Source } from "./index.js";
import type { IndexedChunk } from "./ports.js";

const MAX_PER_CHUNK = 2_000;
const MAX_PER_FILE = 20_000;

/** A chunk whose text is `n` distinct tokens (`t0 t1 … t{n-1}`). */
function adversarialChunk(path: string, n: number, idSuffix = "0-0"): IndexedChunk {
  const text = Array.from({ length: n }, (_, i) => `t${i}`).join(" ");
  return {
    id: `${path}#${idSuffix}`,
    path,
    byteStart: 0,
    byteEnd: text.length,
    contentHash: "h",
    trust: "live",
    embedding: [],
    text,
  };
}

describe("inverted-index growth caps", () => {
  it("caps distinct tokens per chunk and sets truncated", async () => {
    const store = new LocalVectorStore();
    const n = MAX_PER_CHUNK + 500;
    await store.upsert([adversarialChunk("src/a.ts", n)]);

    expect(store.truncated()).toBe(true);
    // Only the first MAX_PER_CHUNK distinct tokens were indexed for lexical hits.
    const inBudget = await store.lexicalCandidates(["t0"], 5);
    const overBudget = await store.lexicalCandidates([`t${n - 1}`], 5);
    expect(inBudget.length).toBe(1);
    expect(overBudget.length).toBe(0);
    // The chunk itself is still stored (retrievable by vector / exact id).
    expect(await store.count()).toBe(1);
  });

  it("caps distinct tokens per file across chunks and sets truncated", async () => {
    const store = new LocalVectorStore();
    // 11 chunks of 2,000 distinct tokens each = 22,000 > the 20,000 file budget.
    // Each chunk's tokens are disjoint so the file's distinct count accumulates.
    const perChunk = MAX_PER_CHUNK;
    const chunks: IndexedChunk[] = [];
    for (let c = 0; c < 11; c++) {
      const start = c * perChunk;
      const text = Array.from({ length: perChunk }, (_, i) => `f${start + i}`).join(" ");
      chunks.push({
        id: `src/big.ts#${c}`,
        path: "src/big.ts",
        byteStart: 0,
        byteEnd: text.length,
        contentHash: "h",
        trust: "live",
        embedding: [],
        text,
      });
    }
    await store.upsert(chunks);

    expect(store.truncated()).toBe(true);
    // A token well within the first 20,000 is indexed; one past the file budget is not.
    const early = await store.lexicalCandidates(["f0"], 5);
    const late = await store.lexicalCandidates([`f${11 * perChunk - 1}`], 5);
    expect(early.length).toBe(1);
    expect(late.length).toBe(0);
  });

  it("a normal file does not set truncated and indexes every token", async () => {
    const store = new LocalVectorStore();
    await store.upsert([adversarialChunk("src/ok.ts", 50)]);
    expect(store.truncated()).toBe(false);
    expect((await store.lexicalCandidates(["t49"], 5)).length).toBe(1);
  });

  it("surfaces truncated on the IndexResult from indexSources", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    // One giant single-line file; the chunker will split it, but the per-file
    // distinct-token cap is reached across chunks for an adversarial corpus.
    const huge = Array.from({ length: MAX_PER_FILE + 5_000 }, (_, i) => `w${i}`).join(" ");
    const sources: Source[] = [{ path: "src/huge.ts", content: huge, trust: "live" }];
    const result = await indexSources(sources, embeddings, store, { chunkTokens: 512, overlapTokens: 0 });
    expect(result.truncated).toBe(true);
  });

  it("re-indexing an unchanged chunk does not double-count postings", async () => {
    const store = new LocalVectorStore();
    const c = adversarialChunk("src/x.ts", 100);
    await store.upsert([c]);
    await store.upsert([c]); // same id — unindex then re-index
    expect(store.truncated()).toBe(false);
    expect((await store.lexicalCandidates(["t0"], 5)).length).toBe(1);
  });
});

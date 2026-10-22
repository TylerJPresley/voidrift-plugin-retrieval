/**
 * FX-E — the hybrid retriever never loses an exact-token hit
 * (retrieval-blueprint §2 invariant "never loses an exact-token hit").
 *
 * OFFLINE and DETERMINISTIC: the HashingEmbeddings + in-memory LocalVectorStore.
 * The regression: a unique token appended to a file that the weak offline
 * embedder ranks OUTSIDE its top-`pool` must still be returned by a hybrid
 * exact-token query, because lexical scoring now runs over the WHOLE inverted
 * index, not the vector-narrowed pool.
 */
import { describe, it, expect } from "vitest";
import { HashingEmbeddings, LocalVectorStore, Retriever, indexSources, type Source } from "./index.js";

const OPTS = { chunkTokens: 512, overlapTokens: 64 };

describe("hybrid preserves an exact-token hit over a large corpus (FX-E §2)", () => {
  it("returns the LAST indexed file for its unique token, though vector ranks it far", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const retriever = new Retriever(embeddings, store);

    // 330 files of similar, overlapping vocabulary so the weak embedder cannot
    // separate them; the unique token lives ONLY in the last file.
    const sources: Source[] = [];
    for (let i = 0; i < 329; i++) {
      sources.push({
        path: `src/mod${String(i).padStart(3, "0")}.ts`,
        content: `export function handler${i}() {\n  // process the common request and response payload here\n}\n`,
        trust: "live",
      });
    }
    const uniqueToken = "zzqqxylophonetoken42";
    sources.push({
      path: "src/last.ts",
      content: `export function last() {\n  // process the common request ${uniqueToken} payload here\n}\n`,
      trust: "live",
    });
    await indexSources(sources, embeddings, store, OPTS);
    expect(await store.count()).toBeGreaterThanOrEqual(330);

    const hits = await retriever.hybrid(uniqueToken, 5);
    expect(hits.some((h) => h.path === "src/last.ts")).toBe(true);
    expect(hits[0]!.path).toBe("src/last.ts"); // the exact-token hit ranks first
  });

  it("hybrid ordering sanity: an exact identifier outranks a mere semantic neighbour", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const retriever = new Retriever(embeddings, store);
    const docs: Source[] = [
      { path: "a.ts", content: "export class TokenStore { /* holds the session credential */ }\n", trust: "live" },
      { path: "b.ts", content: "export function login() { /* authenticate the user session */ }\n", trust: "live" },
    ];
    await indexSources(docs, embeddings, store, OPTS);
    const hits = await retriever.hybrid("TokenStore", 2);
    expect(hits[0]!.path).toBe("a.ts");
  });

  it("the store's lexicalCandidates scans the whole index in O(query terms)", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const sources: Source[] = [
      { path: "x.ts", content: "alpha beta gamma\n", trust: "live" },
      { path: "y.ts", content: "beta gamma delta\n", trust: "live" },
      { path: "z.ts", content: "nothing relevant here\n", trust: "live" },
    ];
    await indexSources(sources, embeddings, store, OPTS);
    const cands = await store.lexicalCandidates(["alpha"], 10);
    expect(cands.map((c) => c.path)).toEqual(["x.ts"]);
    const none = await store.lexicalCandidates(["missingtoken"], 10);
    expect(none).toHaveLength(0);
  });
});

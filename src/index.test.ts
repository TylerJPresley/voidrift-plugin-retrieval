/**
 * Retrieval plugin tests (retrieval-blueprint §11) — OFFLINE and DETERMINISTIC.
 *
 * No network, no model: the deterministic HashingEmbeddings and the in-memory
 * LocalVectorStore are the defaults, so every assertion is reproducible. The
 * suite proves the blueprint's invariants and integration-test index:
 * register(core) wires the two seams; indexing + a vector/hybrid query returns
 * ranked hits; incremental re-index skips unchanged files by content hash; the
 * context provider stays within budget; the tools are gated/read-only; retrieved
 * content is data.
 */
import { describe, it, expect, beforeEach } from "vitest";
import {
  register,
  indexSources,
  contentHash,
  HashingEmbeddings,
  LocalVectorStore,
  Retriever,
  KnowledgeGraph,
  type Source,
} from "./index.js";
import { renderManifest, renderPrefetch, manifestLine } from "./context-provider.js";
import { searchTool, relatedTool, searchSchema } from "./tools.js";
import type { ScopedCoreApi, ContextLayer, ContextProvider, ToolDefinition, ToolExecutor } from "@voidrift/core";

/** A minimal fake scoped CoreAPI capturing registrations (core §12.4). */
function fakeCore(): {
  core: ScopedCoreApi;
  tools: { def: ToolDefinition; exec: ToolExecutor }[];
  providers: { layer: ContextLayer; provider: ContextProvider }[];
  events: string[];
  emitted: { key: string; payload: unknown }[];
} {
  const tools: { def: ToolDefinition; exec: ToolExecutor }[] = [];
  const providers: { layer: ContextLayer; provider: ContextProvider }[] = [];
  const events: string[] = [];
  const emitted: { key: string; payload: unknown }[] = [];
  const core = {
    pluginId: "retrieval",
    register: {
      command: () => {},
      tool: (def: unknown, exec: unknown) =>
        tools.push({ def: def as ToolDefinition, exec: exec as ToolExecutor }),
      prompt: () => {},
      overridePrompt: () => {},
      extendPrompt: () => {},
      contextProvider: (layer: ContextLayer, provider: ContextProvider) => providers.push({ layer, provider }),
      modelProvider: () => {},
      exporter: () => {},
      panel: () => {},
      agent: () => {},
      guardrail: () => {},
      planScope: () => {},
      clearPlanScope: () => {},
    },
    registry: { set: () => {}, get: () => undefined, query: () => [], remove: () => {} },
    events: {
      register: (name: string) => events.push(name),
      subscribe: () => () => {},
      emit: (key: string, payload: unknown) => emitted.push({ key, payload }),
    },
    session: { runInNewSession: async () => ({}) },
  } as unknown as ScopedCoreApi;
  return { core, tools, providers, events, emitted };
}

const docs: Source[] = [
  { path: "src/retry.ts", content: "export function withBackoff() {\n  // exponential delay then retry the request\n}\n", trust: "live" },
  { path: "src/auth.ts", content: "export class TokenStore {\n  // holds the session credential\n}\n", trust: "live" },
  { path: "docs/guide.md", content: "# Guide\nHow to configure the embeddings endpoint for retrieval.\n", trust: "snapshot" },
];

describe("register wires the seams (§5, §9)", () => {
  it("registers the void context provider, both tools, and declares events", () => {
    const { core, tools, providers, events } = fakeCore();
    register(core);
    expect(providers.map((p) => p.layer)).toContain("void");
    expect(tools.map((t) => t.def.name).sort()).toEqual(["retrieval_related", "retrieval_search"]);
    expect(events).toContain("RETRIEVAL_INDEXED");
    expect(events).toContain("RETRIEVAL_INDEX_STALE");
  });
});

describe("vector and hybrid modes (§2)", () => {
  let embeddings: HashingEmbeddings;
  let store: LocalVectorStore;
  let retriever: Retriever;

  beforeEach(async () => {
    embeddings = new HashingEmbeddings();
    store = new LocalVectorStore();
    retriever = new Retriever(embeddings, store);
    await indexSources(docs, embeddings, store, { chunkTokens: 512, overlapTokens: 64 });
  });

  it("vector query ranks the semantically relevant chunk first", async () => {
    const hits = await retriever.vector("retry the request with backoff delay", 3);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]!.path).toBe("src/retry.ts");
  });

  it("hybrid query preserves an exact-identifier hit (RRF) and is deterministic", async () => {
    const a = await retriever.hybrid("TokenStore credential", 3);
    const b = await retriever.hybrid("TokenStore credential", 3);
    expect(a[0]!.path).toBe("src/auth.ts");
    expect(a).toEqual(b); // deterministic
  });
});

describe("incremental indexing by content hash (§3, invariant 2)", () => {
  it("skips unchanged files, re-embeds a changed file, drops a removed file", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const opts = { chunkTokens: 512, overlapTokens: 64 };

    const first = await indexSources(docs, embeddings, store, opts);
    expect(first.files).toBe(3);
    const countAfterFirst = await store.count();

    // Re-run over the unchanged tree: nothing re-embedded.
    const second = await indexSources(docs, embeddings, store, opts);
    expect(second.files).toBe(0);
    expect(second.skipped).toBe(3);
    expect(await store.count()).toBe(countAfterFirst);

    // Change one file: only it re-embeds.
    const edited: Source[] = docs.map((d) =>
      d.path === "src/retry.ts" ? { ...d, content: d.content + "// now with jitter\n" } : d,
    );
    const third = await indexSources(edited, embeddings, store, opts);
    expect(third.files).toBe(1);
    expect(third.skipped).toBe(2);
    expect(await store.hashForPath("src/retry.ts")).toBe(contentHash(edited[0]!.content));

    // Remove a file: its chunks are deleted.
    const fewer = edited.filter((d) => d.path !== "docs/guide.md");
    const fourth = await indexSources(fewer, embeddings, store, opts);
    expect(fourth.removed).toBe(1);
    expect(await store.hashForPath("docs/guide.md")).toBeNull();
  });
});

describe("context provider stays within budget (§5, invariant 4)", () => {
  it("manifest has one body-less line per entry, capped", () => {
    const hits = [
      { path: "a/x.ts", snippet: "BODY", score: 0.9, trust: "live" as const, contentHash: "h1" },
      { path: "b/y.ts", snippet: "BODY", score: 0.8, trust: "live" as const, contentHash: "h2" },
      { path: "c/z.ts", snippet: "BODY", score: 0.7, trust: "live" as const, contentHash: "h3" },
    ];
    const manifest = renderManifest(hits, 2);
    expect(manifest).not.toContain("BODY"); // no bodies
    const lines = manifest.split("\n").slice(1); // drop header
    expect(lines).toHaveLength(2); // capped to maxEntries
    expect(lines[0]).toBe(manifestLine(hits[0]!));
  });

  it("prefetch trims to maxSnippets and maxTokens, degrading rather than overflowing", () => {
    const big = "x".repeat(10_000);
    const hits = Array.from({ length: 10 }, (_, i) => ({
      path: `p${i}.ts`,
      snippet: big,
      score: 1 - i / 10,
      trust: "live" as const,
      contentHash: `hash${i}aaaaaaaa`,
    }));
    const out = renderPrefetch(hits, { maxSnippets: 5, maxTokens: 100 });
    // 100 tokens ~ 400 chars; output must not blow the budget wildly.
    expect(out.length).toBeLessThanOrEqual(100 * 4 + 200); // headers + one trimmed body
    // Fewer than 5 snippets survive because each is trimmed to the token cap.
    const headers = (out.match(/^# p\d+\.ts/gm) ?? []).length;
    expect(headers).toBeLessThanOrEqual(5);
    expect(headers).toBeGreaterThanOrEqual(1);
  });

  it("empty hits yield empty manifest and prefetch (degrade to nothing)", () => {
    expect(renderManifest([], 5)).toBe("");
    expect(renderPrefetch([], { maxSnippets: 5, maxTokens: 100 })).toBe("");
  });

  it("the live provider populates after an index + refresh, within budget", async () => {
    const { core, providers } = fakeCore();
    const engine = register(core, {
      getFocus: () => "retry backoff",
      getTurnInput: () => "where is retry handled",
      config: { prefetchMaxSnippets: 2, prefetchMaxTokens: 50, manifestMaxEntries: 3 },
    });
    await engine.index(docs);
    const provider = providers.find((p) => p.layer === "void")!.provider;
    const out = provider() as string;
    expect(out).toContain("Retrieved");
    // Prefetch is bounded by the token cap (50 tokens ~ 200 chars) plus headers/manifest.
    expect(out.length).toBeLessThan(2000);
  });
});

describe("gated read-only tools, honest fallback (§9, invariant 6)", () => {
  it("both tools are readOnly with Zod schemas (single source §8.1)", () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const retriever = new Retriever(embeddings, store);
    const graph = new KnowledgeGraph();
    const s = searchTool(retriever, store, { mode: "hybrid", k: 8 });
    const r = relatedTool(graph, { depth: 2 });
    expect(s.def.readOnly).toBe(true);
    expect(r.def.readOnly).toBe(true);
    expect(() => searchSchema.parse({ query: "" })).toThrow(); // validation at the boundary
    expect(searchSchema.parse({ query: "ok", mode: "vector", k: 3 })).toMatchObject({ mode: "vector" });
  });

  it("search returns ranked hits with path + content hash as data", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const retriever = new Retriever(embeddings, store);
    await indexSources(docs, embeddings, store, { chunkTokens: 512, overlapTokens: 64 });
    const { exec } = searchTool(retriever, store, { mode: "hybrid", k: 5 });
    const result = await exec({ query: "backoff retry" }, { workspaceRoot: "/w", sessionId: "s" });
    expect(result.status).toBe("success");
    const artifact = result.artifact as { hits: { path: string; contentHash: string }[] };
    expect(artifact.hits[0]!.path).toBe("src/retry.ts");
    expect(artifact.hits[0]!.contentHash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("search on an empty index returns the lexical fallback, never a fabricated hit", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const retriever = new Retriever(embeddings, store);
    const { exec } = searchTool(retriever, store, { mode: "hybrid", k: 5 });
    const result = await exec({ query: "anything" }, { workspaceRoot: "/w", sessionId: "s" });
    expect(result.content).toContain("search_contents");
    expect((result.artifact as { hits: unknown[] }).hits).toHaveLength(0);
  });
});

describe("knowledge-graph mode (§6)", () => {
  it("returns related entities/edges to the configured depth; honest when seed is unknown", async () => {
    const graph = new KnowledgeGraph();
    graph.indexFile("src/a.ts", 'import "./b.ts";\nexport function alpha() {}\n', "ha");
    graph.indexFile("src/b.ts", "export class Beta {}\n", "hb");

    const related = graph.related("alpha", 2);
    expect(related.nodes.some((n) => n.name === "alpha")).toBe(true);
    expect(related.edges.some((e) => e.kind === "defines")).toBe(true);

    const { exec } = relatedTool(graph, { depth: 2 });
    const miss = await exec({ seed: "doesNotExist" }, { workspaceRoot: "/w", sessionId: "s" });
    expect(miss.content).toContain("search_contents");
  });
});

describe("data, not authority (§8, invariant 5)", () => {
  it("snapshot trust is preserved on out-of-workspace docs and surfaced as data", async () => {
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const retriever = new Retriever(embeddings, store);
    await indexSources(docs, embeddings, store, { chunkTokens: 512, overlapTokens: 64 });
    const hits = await retriever.hybrid("configure embeddings endpoint", 5);
    const guide = hits.find((h) => h.path === "docs/guide.md");
    expect(guide?.trust).toBe("snapshot"); // trust carried through, not elevated
  });
});

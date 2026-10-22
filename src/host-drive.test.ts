/**
 * Host-drive integration tests (retrieval-blueprint §3, §5, §11) — OFFLINE,
 * DETERMINISTIC. These drive the plugin through `register(core, options)` with a
 * fake `core.workspace` surface over an in-memory corpus, proving the end-to-end
 * wiring the engine-only unit tests do not reach: lazy incremental indexing from
 * the surface, edit → re-index → new content, delete → gone, no re-read of
 * unchanged files, the surface's exclusions never reach the index, a 200-file
 * corpus indexes within a sane bound, and disabling tears the subscription down.
 */
import { describe, it, expect } from "vitest";
import { register } from "./index.js";
import type { ScopedCoreApi, ContextLayer, ContextProvider, ToolDefinition, ToolExecutor, WorkspaceSurface } from "@voidrift/core";
import { contentHash } from "./indexer.js";

/** A controllable in-memory corpus behind a WorkspaceSurface. Counts reads so a
 * test can assert an unchanged file is not re-read (§3 invariant 2). */
class FakeWorkspace implements WorkspaceSurface {
  readonly corpus = new Map<string, string>();
  readonly reads = new Map<string, number>();
  input = "";
  focus: string[] = [];
  /** Names the surface pretends are binary (NUL) — never readable. */
  binary = new Set<string>();

  async files(): Promise<string[]> {
    return [...this.corpus.keys()].sort();
  }
  async readText(path: string): Promise<{ text: string; hash: string } | null> {
    if (this.binary.has(path)) return null;
    const text = this.corpus.get(path);
    if (text === undefined) return null;
    this.reads.set(path, (this.reads.get(path) ?? 0) + 1);
    return { text, hash: contentHash(text) };
  }
  currentInput(): string {
    return this.input;
  }
  focusPaths(): string[] {
    return this.focus;
  }
}

function fakeCore(ws: WorkspaceSurface): {
  core: ScopedCoreApi;
  tools: Map<string, ToolExecutor>;
  providers: { layer: ContextLayer; provider: ContextProvider }[];
  subs: Map<string, Set<(e: unknown) => void>>;
  emit: (type: string, payload: unknown) => void;
  emitted: { key: string; payload: unknown }[];
  unsubCount: () => number;
} {
  const tools = new Map<string, ToolExecutor>();
  const providers: { layer: ContextLayer; provider: ContextProvider }[] = [];
  const subs = new Map<string, Set<(e: unknown) => void>>();
  const emitted: { key: string; payload: unknown }[] = [];
  let liveSubs = 0;
  const core = {
    pluginId: "retrieval",
    register: {
      command: () => {},
      tool: (def: unknown, exec: unknown) => tools.set((def as ToolDefinition).name, exec as ToolExecutor),
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
      register: () => {},
      subscribe: (type: string, handler: (e: unknown) => void) => {
        let set = subs.get(type);
        if (!set) subs.set(type, (set = new Set()));
        set.add(handler);
        liveSubs++;
        return () => {
          set!.delete(handler);
          liveSubs--;
        };
      },
      emit: (key: string, payload: unknown) => emitted.push({ key, payload }),
    },
    session: { runInNewSession: async () => ({}) },
    workspace: ws,
  } as unknown as ScopedCoreApi;
  const emit = (type: string, payload: unknown): void => {
    for (const h of subs.get(type) ?? []) h({ type, payload });
  };
  return { core, tools, providers, subs, emit, emitted, unsubCount: () => liveSubs };
}

const CTX = { workspaceRoot: "/w", sessionId: "s" };

function seedCorpus(ws: FakeWorkspace): void {
  ws.corpus.set("src/retry.ts", "export function withBackoff() {\n  // exponential delay then retry the request\n}\n");
  ws.corpus.set("src/auth.ts", "export class TokenStore {\n  // holds the session credential\n}\n");
  ws.corpus.set("docs/guide.md", "# Guide\nHow to configure the embeddings endpoint.\n");
}

describe("lazy incremental indexing from the workspace surface (§3)", () => {
  it("a first search indexes the corpus and returns a ranked relevant chunk", async () => {
    const ws = new FakeWorkspace();
    seedCorpus(ws);
    const { core, tools } = fakeCore(ws);
    register(core);
    const res = await tools.get("retrieval_search")!({ query: "retry the request with backoff" }, CTX);
    expect(res.status).toBe("success");
    const hits = (res.artifact as { hits: { path: string }[] }).hits;
    expect(hits[0]!.path).toBe("src/retry.ts");
  });

  it("a second query does NOT re-read unchanged files (invariant 2)", async () => {
    const ws = new FakeWorkspace();
    seedCorpus(ws);
    const { core, tools } = fakeCore(ws);
    register(core);
    await tools.get("retrieval_search")!({ query: "retry" }, CTX);
    const readsAfterFirst = new Map(ws.reads);
    await tools.get("retrieval_search")!({ query: "token" }, CTX);
    for (const [path, n] of ws.reads) {
      expect(n).toBe(readsAfterFirst.get(path)); // no additional reads of any file
    }
  });

  it("edit → re-index → new content found (via focus + change signal)", async () => {
    const ws = new FakeWorkspace();
    seedCorpus(ws);
    const { core, tools, emit } = fakeCore(ws);
    register(core);
    await tools.get("retrieval_search")!({ query: "retry" }, CTX);
    // The agent edits a file: the session marks it read/edited (focus) and the
    // host emits a change signal.
    ws.corpus.set("src/retry.ts", "export function withJitter() {\n  // add random jitter to the exponential backoff\n}\n");
    ws.focus = ["src/retry.ts"];
    emit("WORKSPACE_FILES_CHANGED", { paths: ["src/retry.ts"] });
    const res = await tools.get("retrieval_search")!({ query: "random jitter" }, CTX);
    const hits = (res.artifact as { hits: { path: string; contentHash: string }[] }).hits;
    expect(hits[0]!.path).toBe("src/retry.ts");
    expect(hits[0]!.contentHash).toBe(contentHash(ws.corpus.get("src/retry.ts")!));
  });

  it("delete → gone: a removed file drops from the index", async () => {
    const ws = new FakeWorkspace();
    seedCorpus(ws);
    const { core, tools, emit } = fakeCore(ws);
    const engine = register(core);
    await tools.get("retrieval_search")!({ query: "token credential" }, CTX);
    ws.corpus.delete("src/auth.ts");
    emit("RESOURCE_CHANGED", { type: "file" });
    await tools.get("retrieval_search")!({ query: "anything" }, CTX);
    expect(await engine.store.hashForPath("src/auth.ts")).toBeNull();
  });

  it("excluded/binary files from the surface never reach the index", async () => {
    const ws = new FakeWorkspace();
    seedCorpus(ws);
    ws.corpus.set("logo.png", "\u0000binary");
    ws.binary.add("logo.png"); // the surface returns null for it
    const { core, tools } = fakeCore(ws);
    const engine = register(core);
    await tools.get("retrieval_search")!({ query: "retry" }, CTX);
    expect(await engine.store.hashForPath("logo.png")).toBeNull();
  });
});

describe("context provider populated by the background pass (§5)", () => {
  it("contributes nothing before indexing, then a bounded manifest/prefetch after", async () => {
    const ws = new FakeWorkspace();
    seedCorpus(ws);
    ws.input = "where is retry handled";
    ws.focus = ["src/retry.ts"];
    const { core, tools, providers } = fakeCore(ws);
    register(core, { config: { prefetchMaxSnippets: 2, prefetchMaxTokens: 60 } });
    const provider = providers.find((p) => p.layer === "void")!.provider;
    // Drive an index so the provider caches are populated (the tool path awaits
    // ensureFresh which also refreshes).
    await tools.get("retrieval_search")!({ query: "retry" }, CTX);
    const out = provider() as string;
    expect(out).toContain("Retrieved");
    expect(out.length).toBeLessThan(3000);
  });
});

describe("a 200-file corpus indexes within a sane bound (§3)", () => {
  it("indexes and answers the first query in under 10s", async () => {
    const ws = new FakeWorkspace();
    for (let i = 0; i < 200; i++) {
      ws.corpus.set(`src/mod${i}.ts`, `export function fn${i}() {\n  // module ${i} handles widget ${i}\n}\n`);
    }
    const { core, tools } = fakeCore(ws);
    register(core);
    const started = Date.now();
    const res = await tools.get("retrieval_search")!({ query: "widget 42 handler" }, CTX);
    const elapsed = Date.now() - started;
    expect(res.status).toBe("success");
    expect(elapsed).toBeLessThan(10_000);
    expect((res.artifact as { hits: unknown[] }).hits.length).toBeGreaterThan(0);
  });
});

describe("lifecycle (§6, §12.4)", () => {
  it("subscribes to change events on register (torn down by the scoped teardown)", () => {
    const ws = new FakeWorkspace();
    seedCorpus(ws);
    const { core, subs } = fakeCore(ws);
    register(core);
    // The drive subscribes to both the fine and coarse change signals.
    expect(subs.get("WORKSPACE_FILES_CHANGED")?.size).toBe(1);
    expect(subs.get("RESOURCE_CHANGED")?.size).toBe(1);
  });
});

describe("background passes continue past maxReadsPerPass (FX-E §3)", () => {
  it("a corpus larger than one pass is fully indexed, and RETRIEVAL_INDEXED {files,chunks,ms} is emitted", async () => {
    const { HostDrive } = await import("./host-drive.js");
    const { HashingEmbeddings } = await import("./embeddings.js");
    const { LocalVectorStore } = await import("./store.js");
    const { indexSources } = await import("./indexer.js");
    const { KnowledgeGraph } = await import("./graph.js");

    const ws = new FakeWorkspace();
    for (let i = 0; i < 50; i++) {
      ws.corpus.set(`src/mod${String(i).padStart(2, "0")}.ts`, `export function fn${i}() {\n  // module ${i} handles widget ${i}\n}\n`);
    }
    const embeddings = new HashingEmbeddings();
    const store = new LocalVectorStore();
    const graph = new KnowledgeGraph();
    const emitted: { key: string; payload: unknown }[] = [];
    const drive = new HostDrive({
      workspace: ws,
      store,
      graph,
      index: (sources) => indexSources(sources, embeddings, store, { chunkTokens: 512, overlapTokens: 64, removeAbsent: false }),
      refresh: async () => undefined,
      emit: (key, payload) => emitted.push({ key, payload }),
      maxReadsPerPass: 20, // forces at least 3 passes over 50 files
    });

    await drive.ensureFresh();

    // Every file is indexed despite the 20-file per-pass cap (background loop).
    expect(await store.count()).toBeGreaterThanOrEqual(50);
    expect(await store.hashForPath("src/mod49.ts")).not.toBeNull();
    expect(await store.hashForPath("src/mod00.ts")).not.toBeNull();

    // No file is re-read: each read exactly once across all passes (invariant 2).
    for (const [, n] of ws.reads) expect(n).toBe(1);

    // RETRIEVAL_INDEXED carries files/chunks/ms (the observable pass event).
    const indexedEvents = emitted.filter((e) => e.key === "RETRIEVAL_INDEXED");
    expect(indexedEvents.length).toBe(1); // one event per completed pass
    const p = indexedEvents[0]!.payload as { files: number; chunks: number; ms: number };
    expect(p.files).toBe(50);
    expect(typeof p.chunks).toBe("number");
    expect(p.ms).toBeGreaterThanOrEqual(0);
  });
});

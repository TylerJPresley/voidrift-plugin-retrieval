/**
 * Retrieval defaults its persistence from the workspace surface (retrieval
 * §4; core §12.4). With no explicit `cacheDir`/`currentHash` the plugin uses
 * `core.workspace.cacheDir("retrieval")` and `core.workspace.hashFor`: run 1
 * persists the index under `.voidrift/cache/retrieval/`; run 2 (a fresh engine
 * over the same dir) reloads it and re-reads/re-embeds nothing for an unchanged
 * tree; a modified file is re-indexed; explicit options still win.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { register } from "./index.js";
import { contentHash } from "./indexer.js";
import { INDEX_FILE } from "./store.js";
import type {
  ScopedCoreApi,
  ContextLayer,
  ContextProvider,
  ToolDefinition,
  ToolExecutor,
  WorkspaceSurface,
} from "@voidrift/core";

const CTX = { workspaceRoot: "/w", sessionId: "s" };

/** A WorkspaceSurface over an in-memory corpus whose cacheDir is a REAL temp
 * directory (so persistence round-trips) and whose hashFor is a SYNC cache the
 * surface's own reads populate (what the real surface does). */
class FakeWorkspace implements WorkspaceSurface {
  readonly corpus = new Map<string, string>();
  readonly reads = new Map<string, number>();
  readonly hashes = new Map<string, string>();
  input = "";
  focus: string[] = [];
  constructor(private readonly root: string) {}
  async files(): Promise<string[]> {
    return [...this.corpus.keys()].sort();
  }
  async readText(p: string): Promise<{ text: string; hash: string } | null> {
    const text = this.corpus.get(p);
    if (text === undefined) return null;
    this.reads.set(p, (this.reads.get(p) ?? 0) + 1);
    const hash = contentHash(text);
    this.hashes.set(p, hash); // the surface's own read primes the hash cache
    return { text, hash };
  }
  currentInput(): string {
    return this.input;
  }
  focusPaths(): string[] {
    return this.focus;
  }
  cacheDir(namespace: string): string {
    if (!/^[a-z0-9-]+$/.test(namespace)) throw new Error(`bad namespace ${namespace}`);
    const dir = path.join(this.root, ".voidrift", "cache", namespace);
    return dir;
  }
  hashFor(p: string): string | null {
    // Mirror the real surface: a cached hash (from a prior readText) wins; on a
    // miss, resolve synchronously from the live tree (the surface's own sync
    // read) WITHOUT counting as a readText re-read. Null for an unknown path.
    const cached = this.hashes.get(p);
    if (cached !== undefined) return cached;
    const text = this.corpus.get(p);
    if (text === undefined) return null;
    const hash = contentHash(text);
    this.hashes.set(p, hash);
    return hash;
  }
}

function fakeCore(ws: WorkspaceSurface): {
  core: ScopedCoreApi;
  tools: Map<string, ToolExecutor>;
} {
  const tools = new Map<string, ToolExecutor>();
  const providers: { layer: ContextLayer; provider: ContextProvider }[] = [];
  const subs = new Map<string, Set<(e: unknown) => void>>();
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
        return () => set!.delete(handler);
      },
      emit: () => {},
    },
    session: { runInNewSession: async () => ({}) },
    workspace: ws,
  } as unknown as ScopedCoreApi;
  return { core, tools };
}

function seed(ws: FakeWorkspace): void {
  ws.corpus.set("src/retry.ts", "export function withBackoff() {\n  // exponential delay then retry the request\n}\n");
  ws.corpus.set("src/auth.ts", "export class TokenStore {\n  // holds the session credential\n}\n");
}

describe("retrieval defaults persistence from the workspace surface (§4)", () => {
  let root: string;
  beforeEach(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "voidrift-retr-"));
  });
  afterEach(async () => {
    await fs.rm(root, { recursive: true, force: true });
  });

  it("run 1 persists under .voidrift/cache/retrieval/; run 2 reloads and re-reads nothing", async () => {
    // Run 1: a fresh engine indexes the corpus and persists the index.
    const ws1 = new FakeWorkspace(root);
    seed(ws1);
    const { core: core1, tools: tools1 } = fakeCore(ws1);
    const engine1 = register(core1); // NO explicit cacheDir/currentHash
    await tools1.get("retrieval_search")!({ query: "retry the request" }, CTX);
    // The index file exists under the surface's cacheDir.
    const indexPath = path.join(root, ".voidrift", "cache", "retrieval", INDEX_FILE);
    expect(await fs.stat(indexPath).then(() => true).catch(() => false)).toBe(true);
    expect(await engine1.store.count()).toBeGreaterThan(0);

    // Run 2: a brand-new engine over the SAME workspace/cache dir.
    const ws2 = new FakeWorkspace(root);
    seed(ws2);
    const { core: core2, tools: tools2 } = fakeCore(ws2);
    const engine2 = register(core2);
    await tools2.get("retrieval_search")!({ query: "token credential" }, CTX);
    // The index reloaded (chunks present) and the unchanged files were NOT re-read.
    expect(await engine2.store.count()).toBeGreaterThan(0);
    expect(ws2.reads.get("src/retry.ts")).toBeUndefined();
    expect(ws2.reads.get("src/auth.ts")).toBeUndefined();
  });

  it("a modified file is re-indexed on run 2 (its hash no longer matches)", async () => {
    const ws1 = new FakeWorkspace(root);
    seed(ws1);
    const { core: core1, tools: tools1 } = fakeCore(ws1);
    register(core1);
    await tools1.get("retrieval_search")!({ query: "retry" }, CTX);

    const ws2 = new FakeWorkspace(root);
    seed(ws2);
    // Modify one file before run 2.
    ws2.corpus.set("src/retry.ts", "export function withJitter() {\n  // add random jitter to the backoff\n}\n");
    const { core: core2, tools: tools2 } = fakeCore(ws2);
    const engine2 = register(core2);
    const res = await tools2.get("retrieval_search")!({ query: "random jitter" }, CTX);
    const hits = (res.artifact as { hits: { path: string; contentHash: string }[] }).hits;
    expect(hits[0]!.path).toBe("src/retry.ts");
    expect(hits[0]!.contentHash).toBe(contentHash(ws2.corpus.get("src/retry.ts")!));
    // The modified file was re-read; the unchanged one was not.
    expect(ws2.reads.get("src/retry.ts")).toBe(1);
    expect(ws2.reads.get("src/auth.ts")).toBeUndefined();
    void engine2;
  });

  it("a corrupt/planted index file is ignored and rebuilt", async () => {
    const dir = path.join(root, ".voidrift", "cache", "retrieval");
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir, INDEX_FILE), "{ not valid json");
    const ws = new FakeWorkspace(root);
    seed(ws);
    const { core, tools } = fakeCore(ws);
    const engine = register(core);
    const res = await tools.get("retrieval_search")!({ query: "retry" }, CTX);
    expect(res.status).toBe("success");
    // Rebuilt from the tree despite the corrupt file.
    expect(await engine.store.count()).toBeGreaterThan(0);
    expect(ws.reads.get("src/retry.ts")).toBe(1);
  });

  it("explicit cacheDir/currentHash still win over the surface defaults", async () => {
    const explicitDir = path.join(root, "explicit-cache");
    const ws = new FakeWorkspace(root);
    seed(ws);
    const { core, tools } = fakeCore(ws);
    register(core, { cacheDir: explicitDir, currentHash: () => undefined });
    await tools.get("retrieval_search")!({ query: "retry" }, CTX);
    // The explicit dir was used, NOT the surface's cacheDir("retrieval").
    expect(await fs.stat(path.join(explicitDir, INDEX_FILE)).then(() => true).catch(() => false)).toBe(true);
    expect(
      await fs.stat(path.join(root, ".voidrift", "cache", "retrieval", INDEX_FILE)).then(() => true).catch(() => false),
    ).toBe(false);
  });
});

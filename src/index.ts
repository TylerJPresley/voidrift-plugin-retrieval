/**
 * @voidrift/plugin-retrieval — the Retrieval plugin (retrieval-blueprint.md).
 *
 * A first-party plugin that attaches ONLY through the CoreAPI surface
 * (`register.contextProvider`, `register.tool`, the registry, and the event
 * stream) and never touches SDK internals. It adds vector, hybrid, and
 * knowledge-graph search on top of the core's lexical index (core §8) and
 * document registry (core §4.5), surfacing ranked snippets within the head
 * budget (core §4.8) and gated, read-only tools (core §9). Everything it
 * surfaces is DATA, never authority (core §1 rule 15): a snippet, a manifest
 * line, and a relation inform the model; the gate still decides every action.
 *
 * Backends sit behind two ports (§4): the deterministic offline embedding and
 * the local in-memory store are the defaults so the plugin and its tests run
 * fully offline; a real embeddings endpoint or a hosted vector store plugs in by
 * injecting a port, with no change above it. The plugin degrades gracefully — a
 * missing embeddings backend never crashes the session (§7, core §12.4).
 */
import type { ScopedCoreApi, ToolExecutor, ToolResult } from "@voidrift/core";
import type { EmbeddingsPort, VectorStorePort } from "./ports.js";
import { HashingEmbeddings } from "./embeddings.js";
import { LocalVectorStore } from "./store.js";
import { indexSources, type Source, type IndexResult } from "./indexer.js";
import { KnowledgeGraph } from "./graph.js";
import { Retriever, type RetrievalMode, type Hit } from "./retriever.js";
import { renderManifest, renderPrefetch } from "./context-provider.js";
import { searchTool, relatedTool } from "./tools.js";
import { wireHostDrive } from "./host-drive.js";

export type { EmbeddingsPort, VectorStorePort, IndexedChunk, ScoredChunk, ChunkTrust } from "./ports.js";
export { HashingEmbeddings } from "./embeddings.js";
export { LocalVectorStore } from "./store.js";
export { Retriever } from "./retriever.js";
export type { RetrievalMode, Hit } from "./retriever.js";
export { KnowledgeGraph } from "./graph.js";
export { indexSources, contentHash } from "./indexer.js";
export type { Source, IndexResult } from "./indexer.js";

/** Resolved retrieval configuration; every value defaults to a blueprint constant (§7). */
export interface RetrievalConfig {
  mode: RetrievalMode;
  chunkTokens: number;
  chunkOverlapTokens: number;
  prefetchMaxSnippets: number;
  prefetchMaxTokens: number;
  manifestMaxEntries: number;
  graphDepth: number;
  defaultK: number;
}

/** The blueprint §7 defaults. A consumer overrides via the injected options. */
export const DEFAULT_CONFIG: RetrievalConfig = {
  mode: "hybrid",
  chunkTokens: 512,
  chunkOverlapTokens: 64,
  prefetchMaxSnippets: 5,
  prefetchMaxTokens: 1500,
  manifestMaxEntries: 10,
  graphDepth: 2,
  defaultK: 8,
};

/**
 * Options a host injects to swap the offline defaults for real backends (§4) or
 * to supply the current turn's focus/input and the source set. Everything is
 * optional: with nothing injected the plugin runs on the deterministic offline
 * embedding and the in-memory store, which is what the tests exercise.
 */
export interface RetrievalOptions {
  embeddings?: EmbeddingsPort;
  store?: VectorStorePort;
  config?: Partial<RetrievalConfig>;
  /** The current session focus (drives the stable Void-layer manifest, §5). */
  getFocus?: () => string;
  /** The current turn's user input (drives per-turn prefetch, §5). */
  getTurnInput?: () => string;
  /**
   * Directory for the persisted local index (§4), normally
   * `<workspace>/.voidrift/cache/retrieval/`. When set and the default
   * `LocalVectorStore` is in use, the index is loaded on register and persisted
   * after each pass so a second process re-embeds nothing. A chunk is trusted on
   * load only when its stored content hash still matches the source file's
   * current hash (`currentHash`), so a stale or planted index is never served.
   */
  cacheDir?: string;
  /** Resolve a path's CURRENT content hash for load-time verification (§4, FX-I);
   * a chunk whose stored hash differs is dropped. Absent, loaded chunks are
   * trusted as-is and the host-drive re-verifies on its reconcile pass. */
  currentHash?: (path: string) => string | null | undefined;
  /** Warning sink for a corrupt/incompatible index (one line, then rebuild). */
  onWarn?: (msg: string) => void;
}

/** The live retrieval engine a host drives: index passes and the shared retriever/graph. */
export interface RetrievalEngine {
  readonly embeddings: EmbeddingsPort;
  readonly store: VectorStorePort;
  readonly graph: KnowledgeGraph;
  readonly retriever: Retriever;
  readonly config: RetrievalConfig;
  /** Run an incremental index pass over the current source set (§3). */
  index(sources: Source[]): Promise<IndexResult>;
}

/**
 * The plugin entry point. Wires the two seams and returns the engine so a host
 * (or a test) can drive index passes and queries. The scoped CoreAPI tags every
 * registration with the plugin id and namespaces the registry (§12.4).
 */
export function register(core: ScopedCoreApi, options: RetrievalOptions = {}): RetrievalEngine {
  const config: RetrievalConfig = { ...DEFAULT_CONFIG, ...options.config };
  const embeddings = options.embeddings ?? new HashingEmbeddings();
  const store = options.store ?? new LocalVectorStore();
  const graph = new KnowledgeGraph();
  const retriever = new Retriever(embeddings, store);

  core.events.register("RETRIEVAL_INDEXED");
  core.events.register("RETRIEVAL_INDEX_STALE");

  // --- Persistence defaults from the workspace surface (§4, core §12.4). With
  // NO explicit `cacheDir`/`currentHash` the plugin derives them from
  // `core.workspace`: the owner-only `cacheDir("retrieval")` under
  // `<workspace>/.voidrift/cache/retrieval/`, and `hashFor` for load-time
  // verification (a SYNC content hash the surface primed by reading files during
  // indexing). Explicit options still WIN. A missing/older surface leaves both
  // undefined and persistence simply stays off (the engine still runs).
  const resolvedCacheDir =
    options.cacheDir ??
    (() => {
      try {
        return core.workspace?.cacheDir?.("retrieval");
      } catch {
        return undefined;
      }
    })();
  const resolvedCurrentHash: ((path: string) => string | null | undefined) | undefined =
    options.currentHash ??
    (typeof core.workspace?.hashFor === "function"
      ? (path: string) => core.workspace.hashFor(path)
      : undefined);

  // --- Persistence (§4, FX-I). When a cache dir is configured and the default
  // local store is in use, load the on-disk index (hash-verified) and persist it
  // after each pass. A hosted store persists itself, so skip it there.
  const persistable = resolvedCacheDir && store instanceof LocalVectorStore ? store : null;
  const cacheDir = resolvedCacheDir;
  const persistIndex = async (): Promise<void> => {
    if (persistable && cacheDir) await persistable.persist(cacheDir, embeddings.model, embeddings.dimensions);
  };

  // --- Index engine (§3). A host feeds the source set; indexing is incremental.
  // `removeAbsent` controls the full-reconcile removal: the public `engine.index`
  // reconciles (true); the host-drive indexes a PARTIAL batch per background pass
  // and reconciles removals itself, so it passes false (FX-E).
  const index = async (sources: Source[], removeAbsent = true): Promise<IndexResult> => {
    const result = await indexSources(sources, embeddings, store, {
      chunkTokens: config.chunkTokens,
      overlapTokens: config.chunkOverlapTokens,
      removeAbsent,
    });
    // Maintain the knowledge graph on the same change signal (§6).
    const seen = new Set<string>();
    for (const src of sources) {
      seen.add(src.path);
      const h = (await store.hashForPath(src.path)) ?? "";
      if (graph.hashForPath(src.path) !== h) graph.indexFile(src.path, src.content, h);
    }
    if (result.files > 0 || result.removed > 0) await persistIndex();
    core.events.emit("RETRIEVAL_INDEXED", { files: result.files, chunks: result.chunks, mode: config.mode, truncated: result.truncated });
    return result;
  };

  // --- Context provider (§5): Void-layer manifest + per-turn prefetch, both bounded.
  // The core ContextProvider is synchronous (returns string|string[]), so
  // retrieval precomputes its manifest/prefetch when the host signals a new focus
  // or turn (via `refresh`, below) and the provider reads the latest cache. This
  // mirrors the pipeline's just-in-time rebuild (core §4.1): a turn queries
  // whatever is indexed now and never blocks on an index pass.
  let lastManifest = "";
  let lastPrefetch = "";

  core.register.contextProvider("void", () => {
    try {
      const blocks: string[] = [];
      if (lastManifest) blocks.push(lastManifest);
      if (lastPrefetch) blocks.push(lastPrefetch);
      return blocks.join("\n\n");
    } catch {
      // Never crash the turn — degrade to no retrieval content (§7, core §12.4).
      return "";
    }
  });

  // `refresh` is the hook a host calls after indexing or on new input; it
  // precomputes the manifest and prefetch within budget (§5). With no host
  // wiring the caches stay empty and the provider contributes nothing.
  const refresh = async (): Promise<void> => {
    const focus = options.getFocus?.() ?? defaultFocus(core) ?? "";
    const input = options.getTurnInput?.() ?? core.workspace?.currentInput?.() ?? focus;
    if ((await store.count()) === 0) {
      lastManifest = "";
      lastPrefetch = "";
      return;
    }
    if (focus) {
      const hits = await topHits(retriever, config, focus);
      lastManifest = renderManifest(hits, config.manifestMaxEntries);
    }
    if (input) {
      const hits = await topHits(retriever, config, input);
      lastPrefetch = renderPrefetch(hits, {
        maxSnippets: config.prefetchMaxSnippets,
        maxTokens: config.prefetchMaxTokens,
      });
    }
  };

  // --- Gated read-only tools (§9).
  const search = searchTool(retriever, store, { mode: config.mode, k: config.defaultK });
  const related = relatedTool(graph, { depth: config.graphDepth });

  // --- Host drive (§3): index incrementally from the core's read-only workspace
  // surface, lazily on the first tool call or provider use, refreshed on a core
  // file-change event. Absent `core.workspace` (older host / direct-drive test)
  // the drive is null and the engine is driven manually via `index()`.
  const wired = wireHostDrive(core, {
    store,
    graph,
    index: (sources) => index(sources, /* removeAbsent */ false),
    refresh,
  });
  const drive = wired?.drive ?? null;

  // Load the persisted index once (§4, FX-I) before the first query brings the
  // index up to date; a hash-verified load means a second process re-embeds
  // nothing. Best-effort and awaited by the lazy trigger below.
  const loaded: Promise<void> = (async () => {
    if (persistable && cacheDir) {
      await persistable
        .loadVerified(cacheDir, resolvedCurrentHash ?? (() => undefined), { model: embeddings.model, dimensions: embeddings.dimensions }, options.onWarn)
        .then(() => undefined)
        .catch(() => undefined);
    }
  })();

  // Wrap each tool so the first call (or any call after a change) brings the
  // index up to date before querying — the lazy trigger (§3). Degrades: if the
  // drive is absent the executor runs unchanged.
  const withEnsure = (exec: ToolExecutor): ToolExecutor => {
    if (!drive) return exec;
    return async (args, ctx): Promise<ToolResult> => {
      await loaded;
      await drive.ensureFresh();
      return exec(args, ctx);
    };
  };
  core.register.tool(search.def, withEnsure(search.exec));
  core.register.tool(related.def, withEnsure(related.exec));

  // Kick a non-blocking background first pass so the context provider can be
  // populated for the next turn without any tool call (§3, §5). It MUST wait for
  // the persisted index to load first (FX-I) — otherwise the pass races ahead of
  // `loaded`, sees an empty store, and re-reads every file a reload would have
  // restored. Safe when the drive is absent.
  if (drive) {
    void loaded.then(() => drive.kickBackground());
  }

  return {
    embeddings,
    store,
    graph,
    retriever,
    config,
    index: async (sources) => {
      const r = await index(sources);
      await refresh();
      return r;
    },
  };
}

/** The session focus as a query string: the recently read/edited file names,
 * joined — a stable seed for the Void-layer manifest (§5). Empty when the host
 * has no workspace surface. */
function defaultFocus(core: ScopedCoreApi): string | undefined {
  const paths = core.workspace?.focusPaths?.();
  if (!paths || paths.length === 0) return undefined;
  return paths.map((p) => p.split("/").pop() ?? p).join(" ");
}

/** Resolve the configured mode to hits; graph falls back to hybrid for the manifest/prefetch surface. */
async function topHits(retriever: Retriever, config: RetrievalConfig, query: string): Promise<Hit[]> {
  const mode = config.mode === "graph" ? "hybrid" : config.mode;
  return retriever.search(mode, query, config.defaultK);
}

export default { register };

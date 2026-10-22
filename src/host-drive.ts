/**
 * Host drive (retrieval-blueprint §3, §5) — the glue between the core's
 * read-only workspace surface and the retrieval engine.
 *
 * The plugin owns NO watcher and reaches the host ONLY through the scoped
 * CoreAPI (`core.workspace`, `core.events`). This module drives the index from
 * that surface:
 *
 *  - LAZY: the first `retrieval_search`/`retrieval_related` call or the first
 *    context-provider use triggers the first index pass; nothing is read before.
 *  - INCREMENTAL: `core.workspace.files()` is a cheap glob (NO content read). A
 *    file already in the store is NOT re-read; only a NEW path, or a path the
 *    session has touched this turn (`core.workspace.focusPaths()` — "recently
 *    read/edited", the blueprint's focus signal), is read via
 *    `core.workspace.readText()`. A path that dropped out of the listing has its
 *    chunks and graph nodes deleted. So a second query over an unchanged tree
 *    re-reads nothing (invariant 2).
 *  - BOUNDED: the first pass runs in the background and yields between files so a
 *    turn is never blocked more than a short slice; a query answers from whatever
 *    is indexed now (§3). `maxReadsPerPass` caps the reads one pass performs.
 *  - REFRESH: a core file-change event (`WORKSPACE_FILES_CHANGED` when the host
 *    emits it, else `RESOURCE_CHANGED`) marks the index dirty; the next query or
 *    provider use reconciles. No second watcher stack is added.
 *
 * All of it degrades safely: a missing `core.workspace` (older host, or a unit
 * test that drives the engine directly) leaves the drive inert, and any surface
 * error is swallowed so a turn never fails because of retrieval (§7, core §12.4).
 */
import type { ScopedCoreApi, WorkspaceSurface } from "@voidrift/core";
import type { VectorStorePort } from "./ports.js";
import type { KnowledgeGraph } from "./graph.js";
import type { Source, IndexResult } from "./indexer.js";

export interface HostDriveDeps {
  workspace: WorkspaceSurface;
  store: VectorStorePort;
  graph: KnowledgeGraph;
  /**
   * Index a PARTIAL batch of freshly-read sources WITHOUT removing the rest of
   * the index (the drive reconciles removals itself). Returns the pass report.
   */
  index(sources: Source[]): Promise<IndexResult>;
  /** Precompute the manifest/prefetch from the latest focus/input (engine.refresh). */
  refresh(): Promise<void>;
  emit(key: string, payload: unknown): void;
  /** Max files read in a single pass — the bound that keeps a pass short (§3). */
  maxReadsPerPass?: number;
}

const DEFAULT_MAX_READS = 200;

/**
 * The drive is a small state machine: it tracks whether a first pass has run,
 * whether the index is dirty (a change event arrived), and serializes passes so
 * two triggers never index concurrently. `ensureFresh` is idempotent and cheap
 * when nothing changed.
 */
export class HostDrive {
  private started = false;
  private dirty = true;
  private running: Promise<void> | null = null;
  private readonly maxReads: number;

  constructor(private readonly deps: HostDriveDeps) {
    this.maxReads = deps.maxReadsPerPass ?? DEFAULT_MAX_READS;
  }

  /** Mark the index dirty on a file-change event (§3 refresh). */
  markDirty(): void {
    this.dirty = true;
  }

  /**
   * Ensure the index reflects the workspace, then refresh the provider caches.
   * Lazy: the first call runs the first pass. Idempotent: when nothing is dirty
   * and a pass has already run, only the (cheap) provider refresh runs. Never
   * throws into the caller.
   */
  async ensureFresh(): Promise<void> {
    if (this.running) {
      await this.running.catch(() => undefined);
      return;
    }
    if (this.started && !this.dirty) {
      await this.deps.refresh().catch(() => undefined);
      return;
    }
    this.running = this.pass().finally(() => {
      this.running = null;
    });
    await this.running.catch(() => undefined);
  }

  /** Kick a background first pass without awaiting it (lazy, non-blocking §3). */
  kickBackground(): void {
    if (this.started || this.running) return;
    void this.ensureFresh();
  }

  private async pass(): Promise<void> {
    this.dirty = false;
    this.started = true;
    const startedAt = Date.now();
    let totalFiles = 0;
    let totalChunks = 0;
    let truncated = false;
    try {
      const listing = await this.deps.workspace.files();
      const present = new Set(listing);

      // Reconcile removals first: a path no longer listed loses its chunks/graph.
      const indexed = await this.deps.store.indexedPaths();
      for (const path of indexed.keys()) {
        if (!present.has(path)) {
          await this.deps.store.deleteByPath(path);
          this.deps.graph.removePath(path);
        }
      }

      // Decide what to read: a NEW path (not in the store), or a FOCUS path the
      // session touched this turn (so an edit the agent made is picked up). An
      // unchanged, already-indexed, unfocused path is NEVER re-read (invariant 2).
      // A single pass reads at most `maxReads`; the loop below CONTINUES in the
      // background until nothing unindexed remains, yielding between batches so a
      // turn is never blocked (§3) even on a corpus larger than one batch.
      const focus = new Set(this.deps.workspace.focusPaths());
      let done = false;
      while (!done) {
        const have = await this.deps.store.indexedPaths();
        const batch: string[] = [];
        for (const path of listing) {
          const already = have.has(path);
          if (!already || focus.has(path)) batch.push(path);
          if (batch.length >= this.maxReads) break;
        }
        // Focus paths are re-read every iteration; stop when only focus remains
        // (they are already current after the first iteration) or nothing is left.
        const anyNew = batch.some((p) => !have.has(p));
        if (batch.length === 0 || !anyNew) done = true;

        const sources: Source[] = [];
        for (const path of batch) {
          const read = await this.deps.workspace.readText(path);
          if (!read) continue; // excluded/binary/oversized/unreadable — skip
          sources.push({ path, content: read.text, trust: "live" });
          if (sources.length % 20 === 0) await yieldTick();
        }
        if (sources.length === 0) break;
        const r = await this.deps.index(sources);
        totalFiles += r.files;
        totalChunks += r.chunks;
        truncated = truncated || r.truncated;
        // Yield between batches so a large first pass never blocks a turn (§3).
        await yieldTick();
        if (done) break;
      }
    } catch {
      // Contained — a surface failure never breaks the turn (§7, core §12.4).
    }
    this.deps.emit("RETRIEVAL_INDEXED", {
      files: totalFiles,
      chunks: totalChunks,
      ms: Date.now() - startedAt,
      truncated,
    });
    await this.deps.refresh().catch(() => undefined);
  }
}

function yieldTick(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

/**
 * Wire the drive to the core: subscribe to file-change events and return the
 * drive plus a teardown. A missing `core.workspace` returns null (nothing to
 * drive). The subscription is torn down with the plugin on disable (§12.4): the
 * returned `dispose` unsubscribes.
 */
export function wireHostDrive(
  core: ScopedCoreApi,
  deps: Omit<HostDriveDeps, "workspace" | "emit"> & { workspace?: WorkspaceSurface },
): { drive: HostDrive; dispose: () => void } | null {
  const workspace = deps.workspace ?? core.workspace;
  if (!workspace || typeof workspace.files !== "function") return null;
  const drive = new HostDrive({
    ...deps,
    workspace,
    emit: (key, payload) => core.events.emit(key, payload),
  });

  // Refresh on a workspace change. The host emits `WORKSPACE_FILES_CHANGED`
  // {paths} when a watcher batches changes; absent that, a `RESOURCE_CHANGED`
  // of a file-shaped type is a coarse signal. Either marks the index dirty; the
  // next query/provider reconciles incrementally. No second watcher is added.
  const unsubs: (() => void)[] = [];
  const onChange = (): void => drive.markDirty();
  try {
    unsubs.push(core.events.subscribe("WORKSPACE_FILES_CHANGED", onChange));
    unsubs.push(core.events.subscribe("RESOURCE_CHANGED", onChange));
  } catch {
    // A host whose bus rejects an unregistered type — ignore; lazy re-index on
    // each query still picks up new/removed files.
  }
  return { drive, dispose: () => unsubs.forEach((u) => u()) };
}

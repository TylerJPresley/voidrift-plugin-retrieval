/**
 * Knowledge-graph context (retrieval-blueprint §6).
 *
 * The graph indexes structure, not prose: entities (files and symbols) and
 * relations (defines, imports, references, contains). A node carries the same
 * content hash as its source file, so the graph is maintained incrementally on
 * the same change signal as the vector index (§6). A query takes a seed and
 * returns related nodes and edges to a bounded depth, surfaced as a compact
 * relation list — data that tells the model what relates to what; it still reads
 * the actual files through the gated read tool to act (§6, §8).
 *
 * Extraction is lexical and language-light: it recognises declarations and
 * imports common to the TypeScript/JS and Markdown the workspace holds. It is a
 * best-effort structural map, not a compiler — enough to answer relational
 * questions a snippet list cannot express.
 */

export type EdgeKind = "defines" | "imports" | "references" | "contains";

export interface GraphEdge {
  from: string; // node id
  to: string; // node id
  kind: EdgeKind;
}

export interface GraphNode {
  id: string; // `file:<path>` or `symbol:<name>`
  kind: "file" | "symbol";
  name: string;
  path: string;
  contentHash: string;
}

const DECL_RE =
  /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)/;
const IMPORT_RE = /^\s*import\s+(?:[\s\S]*?\s+from\s+)?["']([^"']+)["']/;
const REQUIRE_RE = /require\(\s*["']([^"']+)["']\s*\)/g;

export class KnowledgeGraph {
  private readonly nodes = new Map<string, GraphNode>();
  private readonly edges: GraphEdge[] = [];
  /** path → node ids owned by that file, for incremental removal. */
  private readonly byPath = new Map<string, Set<string>>();
  private readonly hashes = new Map<string, string>();

  /** sha256 content hash stored for a path, or null (incremental §6). */
  hashForPath(path: string): string | null {
    return this.hashes.get(path) ?? null;
  }

  /** Remove every node/edge a file owns — a changed or removed file (§6). */
  removePath(path: string): void {
    const owned = this.byPath.get(path);
    if (owned) {
      for (const id of owned) this.nodes.delete(id);
      for (let i = this.edges.length - 1; i >= 0; i--) {
        const e = this.edges[i]!;
        if (owned.has(e.from) || owned.has(e.to)) this.edges.splice(i, 1);
      }
    }
    this.byPath.delete(path);
    this.hashes.delete(path);
  }

  /** Extract and store entities/relations for one file (idempotent per path). */
  indexFile(path: string, content: string, contentHash: string): void {
    this.removePath(path);
    const owned = new Set<string>();
    this.hashes.set(path, contentHash);

    const fileId = `file:${path}`;
    this.nodes.set(fileId, { id: fileId, kind: "file", name: path, path, contentHash });
    owned.add(fileId);

    for (const raw of content.split(/\r?\n/)) {
      const decl = DECL_RE.exec(raw);
      if (decl) {
        const name = decl[1]!;
        const symId = `symbol:${name}`;
        if (!this.nodes.has(symId)) {
          this.nodes.set(symId, { id: symId, kind: "symbol", name, path, contentHash });
        }
        owned.add(symId);
        this.edges.push({ from: fileId, to: symId, kind: "contains" });
        this.edges.push({ from: fileId, to: symId, kind: "defines" });
      }
      const imp = IMPORT_RE.exec(raw);
      if (imp) {
        const target = `file:${imp[1]!}`;
        this.edges.push({ from: fileId, to: target, kind: "imports" });
      }
      let req: RegExpExecArray | null;
      REQUIRE_RE.lastIndex = 0;
      while ((req = REQUIRE_RE.exec(raw)) !== null) {
        this.edges.push({ from: fileId, to: `file:${req[1]!}`, kind: "imports" });
      }
    }
    this.byPath.set(path, owned);
  }

  /**
   * Breadth-first traversal from a seed node to `depth`, returning the reached
   * nodes (those present in the graph) and the edges walked. A seed may be a
   * symbol name, a file path, or a bare token naming either (§6).
   */
  related(seed: string, depth: number): { nodes: GraphNode[]; edges: GraphEdge[] } {
    const start = this.resolveSeed(seed);
    if (!start) return { nodes: [], edges: [] };

    const visited = new Set<string>([start]);
    const outNodes = new Map<string, GraphNode>();
    const outEdges: GraphEdge[] = [];
    let frontier = [start];
    const startNode = this.nodes.get(start);
    if (startNode) outNodes.set(start, startNode);

    for (let d = 0; d < Math.max(0, depth) && frontier.length > 0; d++) {
      const next: string[] = [];
      for (const e of this.edges) {
        let other: string | null = null;
        if (frontier.includes(e.from)) other = e.to;
        else if (frontier.includes(e.to)) other = e.from;
        if (other === null) continue;
        outEdges.push(e);
        const node = this.nodes.get(other);
        if (node) outNodes.set(other, node);
        if (!visited.has(other)) {
          visited.add(other);
          next.push(other);
        }
      }
      frontier = next;
    }
    return { nodes: [...outNodes.values()], edges: dedupeEdges(outEdges) };
  }

  private resolveSeed(seed: string): string | null {
    if (this.nodes.has(seed)) return seed;
    if (this.nodes.has(`symbol:${seed}`)) return `symbol:${seed}`;
    if (this.nodes.has(`file:${seed}`)) return `file:${seed}`;
    // Fall back to a case-insensitive name match on a symbol or file.
    const lower = seed.toLowerCase();
    for (const n of this.nodes.values()) {
      if (n.name.toLowerCase() === lower) return n.id;
    }
    return null;
  }
}

function dedupeEdges(edges: GraphEdge[]): GraphEdge[] {
  const seen = new Set<string>();
  const out: GraphEdge[] = [];
  for (const e of edges) {
    const key = `${e.from}|${e.kind}|${e.to}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(e);
  }
  return out;
}

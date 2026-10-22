/**
 * Tool + knowledge-graph branch coverage (retrieval-blueprint §6, §9) —
 * OFFLINE, DETERMINISTIC. These exercise the success render of
 * `retrieval_related` (edges rendered back as data) and the graph's
 * `removePath` re-index path, which the integration suite does not reach.
 */
import { describe, it, expect } from "vitest";
import { KnowledgeGraph } from "./graph.js";
import { relatedTool } from "./tools.js";

const CTX = { workspaceRoot: "/w", sessionId: "s" };

describe("retrieval_related renders matched entities as data (§9)", () => {
  it("returns the seed's edges as structural lines with a nodes/edges artifact", async () => {
    const graph = new KnowledgeGraph();
    graph.indexFile("src/a.ts", 'import "./b.ts";\nexport function alpha() {}\n', "ha");
    graph.indexFile("src/b.ts", "export class Beta {}\n", "hb");

    const { exec } = relatedTool(graph, { depth: 2 });
    const res = await exec({ seed: "alpha" }, CTX);
    expect(res.status).toBe("success");
    // The success branch renders `from --kind--> to` edge lines (tools §9).
    expect(res.content).toMatch(/alpha|src\/a\.ts/);
    expect(res.content).toContain("-->");
    const artifact = res.artifact as { nodes: unknown[]; edges: unknown[] };
    expect(artifact.nodes.length).toBeGreaterThan(0);
    expect(artifact.edges.length).toBeGreaterThan(0);
    // It is data, not a fallback: no lexical-fallback pointer when there is a hit.
    expect(res.content).not.toContain("search_contents");
  });

  it("honours an explicit depth override", async () => {
    const graph = new KnowledgeGraph();
    graph.indexFile("src/a.ts", "export function alpha() {}\n", "ha");
    const { exec } = relatedTool(graph, { depth: 2 });
    const res = await exec({ seed: "alpha", depth: 1 }, CTX);
    expect(res.status).toBe("success");
  });
});

describe("KnowledgeGraph.removePath on re-index (§6, invariant 2)", () => {
  it("drops a changed file's nodes and its edges, then re-adds from new content", () => {
    const graph = new KnowledgeGraph();
    graph.indexFile("src/a.ts", 'import "./b.ts";\nexport function alpha() {}\n', "h1");
    expect(graph.related("alpha", 2).nodes.some((n) => n.name === "alpha")).toBe(true);

    // Re-index the same path with different content: removePath clears the old
    // symbol + edges before the new extraction (the uncovered removal loop).
    graph.indexFile("src/a.ts", "export function omega() {}\n", "h2");
    expect(graph.related("alpha", 2).nodes.some((n) => n.name === "alpha")).toBe(false);
    expect(graph.related("omega", 2).nodes.some((n) => n.name === "omega")).toBe(true);
    expect(graph.hashForPath("src/a.ts")).toBe("h2");
  });

  it("recognises a require()-style import edge", () => {
    const graph = new KnowledgeGraph();
    graph.indexFile("src/c.ts", 'const b = require("./b.ts");\n', "hc");
    const edges = graph.related("src/c.ts", 2).edges;
    expect(edges.some((e) => e.kind === "imports" && e.to.includes("b.ts"))).toBe(true);
  });
});

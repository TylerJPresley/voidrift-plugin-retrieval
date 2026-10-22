/**
 * Chunker branch coverage (retrieval-blueprint §3) — OFFLINE, DETERMINISTIC.
 *
 * The integration suite indexes tiny docs with a 512-token window, so the
 * flush/overlap/boundary branches of `chunk()` never fire. These behavioural
 * tests drive a small window so the structural-boundary flush, the window
 * overflow, and the overlap rewind are all exercised, and assert the byte spans
 * stay exact and non-overlapping-in-text-but-overlapping-in-span as designed.
 */
import { describe, it, expect } from "vitest";
import { chunk } from "./chunker.js";

describe("chunk() windowing and boundaries (§3)", () => {
  it("returns no chunks for empty content", () => {
    expect(chunk("", { chunkTokens: 8, overlapTokens: 2 })).toEqual([]);
  });

  it("emits a single chunk when the content fits the window", () => {
    const content = "one small line\n";
    const chunks = chunk(content, { chunkTokens: 512, overlapTokens: 64 });
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toEqual({ text: content, byteStart: 0, byteEnd: content.length });
  });

  it("splits on window overflow and keeps byte spans exact and covering the file", () => {
    // ~16 chars per chunk (chunkTokens 4 * 4 chars). Ten lines force overflow.
    const lines = Array.from({ length: 10 }, (_, i) => `line number ${i}\n`);
    const content = lines.join("");
    const chunks = chunk(content, { chunkTokens: 4, overlapTokens: 1 });
    expect(chunks.length).toBeGreaterThan(1);
    // Each chunk's text matches its recorded byte span exactly.
    for (const c of chunks) {
      expect(content.slice(c.byteStart, c.byteEnd)).toBe(c.text);
      expect(c.byteEnd).toBeGreaterThan(c.byteStart);
    }
    // The chunks cover to the end of the file.
    expect(chunks[chunks.length - 1]!.byteEnd).toBe(content.length);
  });

  it("starts a new chunk at a structural boundary once past half the window, with overlap rewind", () => {
    // A heading boundary after enough content triggers the boundary flush; the
    // next chunk's start is rewound by the overlap, so spans overlap in bytes.
    const content =
      "# First Section\n" +
      "body line a\nbody line b\nbody line c\nbody line d\n" +
      "# Second Section\n" +
      "body line e\nbody line f\n";
    const chunks = chunk(content, { chunkTokens: 6, overlapTokens: 2 });
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    // Overlap means some later chunk starts before the previous chunk ends.
    const overlapped = chunks.some((c, i) => i > 0 && c.byteStart < chunks[i - 1]!.byteEnd);
    expect(overlapped).toBe(true);
    // Spans remain exact against the source.
    for (const c of chunks) expect(content.slice(c.byteStart, c.byteEnd)).toBe(c.text);
  });

  it("clamps overlap below the window so a chunk always advances (no infinite loop)", () => {
    // overlapTokens >= chunkTokens would stall; the clamp (maxChars - 1) prevents it.
    const content = Array.from({ length: 6 }, (_, i) => `xxxx line ${i}\n`).join("");
    const chunks = chunk(content, { chunkTokens: 3, overlapTokens: 99 });
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[chunks.length - 1]!.byteEnd).toBe(content.length);
  });
});

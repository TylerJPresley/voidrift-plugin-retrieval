/**
 * Chunking (retrieval-blueprint §3, step 3).
 *
 * A file is split into overlapping chunks sized to the embeddings window, split
 * on structural boundaries first (heading, blank-line paragraph, function/class
 * line) so a chunk is a coherent unit, falling back to a sliding window when no
 * structure is found. Each chunk records its source path and byte span, so a
 * result points back to an exact location (§3). Sizes are expressed in tokens;
 * the deterministic default approximates a token as ~4 chars, which is enough for
 * bounded, overlapping chunks without a model tokenizer.
 */

export interface Chunk {
  text: string;
  byteStart: number;
  byteEnd: number;
}

const CHARS_PER_TOKEN = 4;

/** A structural boundary: markdown heading, or a top-level declaration line. */
function isBoundary(line: string): boolean {
  return /^(#{1,6}\s|\s*(export\s+)?(async\s+)?(function|class|interface|type|enum)\s|\s*def\s|\s*fn\s)/.test(
    line,
  );
}

/**
 * Split `content` into overlapping chunks. `chunkTokens`/`overlapTokens` bound
 * the window; structural boundaries start a new chunk when the current one is
 * already past half the window, keeping chunks coherent without unbounded growth.
 */
export function chunk(
  content: string,
  opts: { chunkTokens: number; overlapTokens: number },
): Chunk[] {
  const maxChars = Math.max(1, opts.chunkTokens * CHARS_PER_TOKEN);
  const overlapChars = Math.max(0, Math.min(opts.overlapTokens * CHARS_PER_TOKEN, maxChars - 1));
  if (content.length === 0) return [];

  const chunks: Chunk[] = [];
  const lines = content.split(/(?<=\n)/); // keep newlines so byte spans are exact
  let bufStart = 0;
  let bufEnd = 0;
  let bufLen = 0;

  const flush = (): void => {
    if (bufLen === 0) return;
    chunks.push({ text: content.slice(bufStart, bufEnd), byteStart: bufStart, byteEnd: bufEnd });
  };

  for (const line of lines) {
    const atBoundary = isBoundary(line) && bufLen >= maxChars / 2;
    const wouldOverflow = bufLen + line.length > maxChars && bufLen > 0;
    if (atBoundary || wouldOverflow) {
      flush();
      // Overlap: rewind the next chunk's start by `overlapChars` within the file.
      const nextStart = Math.max(0, bufEnd - overlapChars);
      bufStart = nextStart;
      bufLen = bufEnd - bufStart;
    }
    bufEnd += line.length;
    bufLen = bufEnd - bufStart;
  }
  flush();
  return chunks;
}

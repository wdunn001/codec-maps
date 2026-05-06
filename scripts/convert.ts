/**
 * convert.ts
 *
 * Converts a HuggingFace tokenizer.json → CodecTokenizerMap JSON.
 *
 * Handles three tokenizer families that cover ~95% of open models:
 *
 *   1. ByteLevel BPE  (Llama-3, Qwen2, Phi-3, Mistral-Nemo, DeepSeek-V2+)
 *      GPT-2-style byte→unicode encoding baked into token text.
 *      Tokens are decoded by reversing the byte-to-unicode table.
 *      No separate byte_fallback range — every token already decodes to bytes.
 *
 *   2. Metaspace / SentencePiece BPE  (Llama-2, Mistral-7B-v0.1)
 *      ▁ (U+2581) prefix marks word starts (decoded as space).
 *      Byte fallback tokens are <0x00>–<0xFF>, consecutive IDs.
 *
 *   3. SentencePiece Unigram  (Gemma, T5, AlBERT)
 *      Same ▁ convention, same <0x??> byte fallback range.
 *
 * Output schema: TokenizerMap from @codecai/web
 *   { id, version, vocab_size, tokens, special_tokens?,
 *     byte_fallback_start?, byte_fallback_end?, published_at }
 */

// ── GPT-2 byte-to-unicode table ──────────────────────────────────────────────
// The 256-entry bijection used by GPT-2 / tiktoken BPE tokenizers.
// Bytes 33–126 (printable ASCII), 161–172, 174–255 map to themselves.
// Everything else maps to codepoints starting at 256 (Ā).

function buildBytesToUnicode(): Map<number, string> {
  const bs: number[] = [
    ...range(33, 127),   // ! through ~
    ...range(161, 173),  // ¡ through ¬
    ...range(174, 256),  // ® through ÿ
  ];
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  const map = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) {
    map.set(cs[i]!, String.fromCodePoint(bs[i]!));
  }
  return map;
}

function range(start: number, end: number): number[] {
  return Array.from({ length: end - start }, (_, i) => start + i);
}

// unicode codepoint → byte value (reverse of the GPT-2 table)
const UNICODE_TO_BYTE: Map<number, number> = (() => {
  const fwd = buildBytesToUnicode();
  const rev = new Map<number, number>();
  for (const [byteVal, char] of fwd) {
    rev.set(char.codePointAt(0)!, byteVal);
  }
  return rev;
})();

/**
 * Decode a GPT-2 byte-level BPE token string to actual bytes.
 * Each character in the token string is a GPT-2-encoded byte.
 */
function decodeByteLevelToken(encoded: string): Uint8Array {
  const bytes: number[] = [];
  for (const char of encoded) {
    const cp = char.codePointAt(0)!;
    const b = UNICODE_TO_BYTE.get(cp);
    if (b !== undefined) {
      bytes.push(b);
    } else {
      // Character is a literal — encode as UTF-8 bytes.
      const enc = new TextEncoder().encode(char);
      bytes.push(...enc);
    }
  }
  return new Uint8Array(bytes);
}

// ── HuggingFace tokenizer.json types ────────────────────────────────────────

interface HFTokenizerJson {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges?: string[];
  };
  added_tokens: Array<{
    id: number;
    content: string;
    special: boolean;
  }>;
  decoder?: {
    type: string;
    decoders?: Array<{ type: string; replacement?: string; prepend_scheme?: string }>;
    replacement?: string;
    prepend_scheme?: string;
  };
  normalizer?: { type: string } | null;
}

// ── CodecTokenizerMap types (subset — avoids importing @codecai/web at build time) ──

export interface CodecTokenizerMap {
  id: string;
  version: string;
  vocab_size: number;
  tokens: Record<string, string>;        // decoded_text → token_id_string
  special_tokens?: Record<string, number>; // name → id
  byte_fallback_start?: number;
  byte_fallback_end?: number;
  published_at: string;
}

// ── Decoder detection helpers ────────────────────────────────────────────────

function isByteLevelDecoder(hf: HFTokenizerJson): boolean {
  const d = hf.decoder;
  if (!d) return false;
  if (d.type === 'ByteLevel') return true;
  if (d.type === 'Sequence' && d.decoders) {
    return d.decoders.some((x) => x.type === 'ByteLevel');
  }
  return false;
}

function isMetaspaceDecoder(hf: HFTokenizerJson): boolean {
  const d = hf.decoder;
  if (!d) return false;
  if (d.type === 'Metaspace') return true;
  if (d.type === 'Sequence' && d.decoders) {
    return d.decoders.some((x) => x.type === 'Metaspace');
  }
  return false;
}

// ── Core conversion ──────────────────────────────────────────────────────────

const BYTE_FALLBACK_RE = /^<0x([0-9A-Fa-f]{2})>$/;
const SPECIAL_RE = /^<[|[].*[|\]]>$|^\[.*\]$|^<s>$|^<\/s>$|^<unk>$|^<pad>$|^<mask>$/;

export function convert(
  hf: HFTokenizerJson,
  modelId: string,
): CodecTokenizerMap {
  const vocab = hf.model.vocab;
  const byteLevel = isByteLevelDecoder(hf);
  const metaspace = isMetaspaceDecoder(hf);

  const tokens: Record<string, string> = {};
  const specialTokens: Record<string, number> = {};

  // Collect byte fallback token IDs for SentencePiece models
  const byteFallbackIds: Map<number, number> = new Map(); // byte_value → id

  // Build a set of special token IDs for quick lookup
  const specialIds = new Set(
    hf.added_tokens.filter((t) => t.special).map((t) => t.id),
  );
  // Also add any added tokens marked special
  for (const t of hf.added_tokens) {
    if (t.special || SPECIAL_RE.test(t.content)) {
      specialTokens[t.content] = t.id;
    }
  }

  const td = new TextDecoder('utf-8', { fatal: false });

  for (const [tokenText, id] of Object.entries(vocab)) {
    // Skip special tokens — they go into special_tokens map.
    if (specialIds.has(id)) continue;

    // ── ByteLevel BPE (Llama-3, Qwen2, Phi-3, DeepSeek-V2+) ────────────────
    if (byteLevel) {
      const bytes = decodeByteLevelToken(tokenText);
      const decoded = td.decode(bytes);
      tokens[decoded] = String(id);
      continue;
    }

    // ── SentencePiece / Metaspace (Llama-2, Mistral, Gemma) ─────────────────

    // Byte fallback token (<0xHH>)
    const bfMatch = tokenText.match(BYTE_FALLBACK_RE);
    if (bfMatch) {
      const byteVal = parseInt(bfMatch[1]!, 16);
      byteFallbackIds.set(byteVal, id);
      continue;
    }

    // Ordinary token: replace ▁ (U+2581) with space.
    const decoded = tokenText.replace(/▁/g, ' ');
    tokens[decoded] = String(id);
  }

  // Compute byte_fallback_start / byte_fallback_end from the byte fallback map.
  // They should be consecutive IDs 0x00–0xFF.
  let bfStart: number | undefined;
  let bfEnd: number | undefined;
  if (byteFallbackIds.size > 0) {
    // Find the ID for byte 0x00 — that's the start.
    const id0 = byteFallbackIds.get(0);
    const id255 = byteFallbackIds.get(255);
    if (id0 !== undefined && id255 !== undefined) {
      bfStart = id0;
      bfEnd = id255;
    } else {
      // Fall back to min/max over all byte fallback IDs.
      const allIds = [...byteFallbackIds.values()];
      bfStart = Math.min(...allIds);
      bfEnd = Math.max(...allIds);
    }
  }

  const result: CodecTokenizerMap = {
    id: modelId,
    version: '1',
    vocab_size: Object.keys(vocab).length + hf.added_tokens.filter((t) => t.special).length,
    tokens,
    published_at: new Date().toISOString(),
  };

  if (Object.keys(specialTokens).length > 0) {
    result.special_tokens = specialTokens;
  }
  if (bfStart !== undefined && bfEnd !== undefined) {
    result.byte_fallback_start = bfStart;
    result.byte_fallback_end = bfEnd;
  }

  return result;
}

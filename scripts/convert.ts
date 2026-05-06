/**
 * convert.ts
 *
 * Converts a HuggingFace tokenizer.json → CodecTokenizerMap JSON.
 *
 * Output schema (v2):
 *
 *   {
 *     id, version, vocab_size,
 *     vocab:    { raw_token_text: id }         // raw form from tokenizer.json
 *     encoder:  "byte_level" | "metaspace"     // how raw tokens map to bytes
 *     merges?:  ["a b", ...]                   // BPE merge rules (priority order)
 *     pre_tokenizer_pattern?: string           // regex (byte_level only)
 *     byte_fallback_start?, byte_fallback_end? // SentencePiece byte tokens
 *     special_tokens?: { name: id }
 *     published_at
 *   }
 *
 * "Raw" form means the literal string keys in tokenizer.json's vocab — for
 * byte-level BPE these contain GPT-2-encoded characters (Ġ, etc.); for
 * metaspace these contain ▁-prefixed strings. The Detokenizer applies the
 * appropriate decoder to recover human-readable text. The BPE Tokenizer
 * uses raw form directly so its lookups match HuggingFace exactly.
 */

// ── HuggingFace tokenizer.json types ────────────────────────────────────────

interface HFPretokenizer {
  type: string;
  pattern?: { Regex?: string; String?: string };
  pretokenizers?: HFPretokenizer[];
  replacement?: string;
}

interface HFDecoder {
  type: string;
  decoders?: HFDecoder[];
}

interface HFTokenizerJson {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges?: string[] | Array<[string, string]>;
    byte_fallback?: boolean;
  };
  added_tokens: Array<{
    id: number;
    content: string;
    special: boolean;
  }>;
  pre_tokenizer?: HFPretokenizer | null;
  decoder?: HFDecoder | null;
}

// ── Output schema ────────────────────────────────────────────────────────────

export interface CodecTokenizerMap {
  id: string;
  version: string;
  vocab_size: number;
  vocab: Record<string, number>;
  encoder?: 'byte_level' | 'metaspace';
  merges?: string[];
  pre_tokenizer_pattern?: string;
  byte_fallback_start?: number;
  byte_fallback_end?: number;
  special_tokens?: Record<string, number>;
  published_at: string;
}

// ── Helpers to walk HF decoder/pre_tokenizer trees ──────────────────────────

function findInTree(node: HFPretokenizer | HFDecoder | null | undefined, type: string): HFPretokenizer | HFDecoder | null {
  if (!node) return null;
  if (node.type === type) return node;
  const children = (node as HFPretokenizer).pretokenizers ?? (node as HFDecoder).decoders;
  if (children) {
    for (const child of children) {
      const found = findInTree(child, type);
      if (found) return found;
    }
  }
  return null;
}

function detectEncoder(hf: HFTokenizerJson): 'byte_level' | 'metaspace' | undefined {
  if (findInTree(hf.decoder, 'ByteLevel')) return 'byte_level';
  if (findInTree(hf.pre_tokenizer, 'ByteLevel')) return 'byte_level';
  if (findInTree(hf.decoder, 'Metaspace') || findInTree(hf.pre_tokenizer, 'Metaspace')) {
    return 'metaspace';
  }
  // Models with byte_fallback=true and no explicit decoder type are metaspace.
  if (hf.model.byte_fallback) return 'metaspace';
  return undefined;
}

function extractPreTokenizerPattern(hf: HFTokenizerJson): string | undefined {
  const split = findInTree(hf.pre_tokenizer, 'Split') as HFPretokenizer | null;
  if (split?.pattern?.Regex) return split.pattern.Regex;
  return undefined;
}

function normalizeMerges(hf: HFTokenizerJson): string[] | undefined {
  const m = hf.model.merges;
  if (!m || m.length === 0) return undefined;
  if (Array.isArray(m[0])) {
    return (m as Array<[string, string]>).map((pair) => `${pair[0]} ${pair[1]}`);
  }
  return m as string[];
}

// ── Core conversion ──────────────────────────────────────────────────────────

const BYTE_FALLBACK_RE = /^<0x([0-9A-Fa-f]{2})>$/;

export function convert(hf: HFTokenizerJson, modelId: string): CodecTokenizerMap {
  const encoder = detectEncoder(hf);
  const merges = normalizeMerges(hf);
  const pre_tokenizer_pattern =
    encoder === 'byte_level' ? extractPreTokenizerPattern(hf) : undefined;

  // Vocab — keep tokenizer.json keys exactly as they are (raw form).
  const vocab: Record<string, number> = { ...hf.model.vocab };

  // Add special tokens to vocab so the detokenizer can resolve their IDs.
  // Track them separately so they can be skipped during text rendering.
  const special_tokens: Record<string, number> = {};
  for (const t of hf.added_tokens) {
    vocab[t.content] = t.id;
    if (t.special) special_tokens[t.content] = t.id;
  }

  // SentencePiece byte fallback range — locate the contiguous IDs for
  // <0x00>…<0xFF> if they exist.
  let byte_fallback_start: number | undefined;
  let byte_fallback_end: number | undefined;
  if (encoder === 'metaspace' || hf.model.byte_fallback) {
    const byteIds = new Map<number, number>();
    for (const [token, id] of Object.entries(vocab)) {
      const m = token.match(BYTE_FALLBACK_RE);
      if (m) byteIds.set(parseInt(m[1]!, 16), id);
    }
    if (byteIds.size === 256) {
      byte_fallback_start = byteIds.get(0)!;
      byte_fallback_end = byteIds.get(255)!;
    }
  }

  const result: CodecTokenizerMap = {
    id: modelId,
    version: '2',
    vocab_size: Object.keys(vocab).length,
    vocab,
    published_at: new Date().toISOString(),
  };

  if (encoder) result.encoder = encoder;
  if (merges) result.merges = merges;
  if (pre_tokenizer_pattern) result.pre_tokenizer_pattern = pre_tokenizer_pattern;
  if (byte_fallback_start !== undefined) {
    result.byte_fallback_start = byte_fallback_start;
    result.byte_fallback_end = byte_fallback_end;
  }
  if (Object.keys(special_tokens).length > 0) {
    result.special_tokens = special_tokens;
  }

  return result;
}

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
  tool_calling?: ToolCallingBlock;
  published_at: string;
}

export interface ToolCallingBlock {
  convention:
    | 'llama3'
    | 'qwen25'
    | 'phi4'
    | 'mistral_nemo'
    | 'deepseek_v3'
    | 'deepseek_r1'
    | 'custom';
  markers: { start: string; end: string };
  args_format: 'json' | 'python_args';
  result_format: 'text' | 'json';
}

/**
 * Subset of `tokenizer_config.json` we care about. The chat_template
 * is the authoritative signal for which tool-calling convention a
 * model uses; everything else in the config is ignored here.
 */
export interface HFTokenizerConfig {
  chat_template?: string | Array<{ name: string; template: string }>;
}

// ── Tool-calling convention registry ────────────────────────────────────────
//
// Mirrors the registry in @codecai/maps-cli/src/convert.ts. Detection is
// substring-based on chat_template: first signature match wins. Both
// markers MUST resolve to IDs (in special_tokens or vocab) before the
// block is emitted; vocab-only markers are PROMOTED into special_tokens
// so the spec contract holds. See spec/PROTOCOL.md § Tool-call calling
// conventions in the map for the normative definition.

interface ConventionEntry {
  convention: ToolCallingBlock['convention'];
  templateSignature: string;
  markers: { start: string; end: string };
  args_format: ToolCallingBlock['args_format'];
  result_format: ToolCallingBlock['result_format'];
}

// Auto-detection registry — only conventions whose markers come as a
// paired (start, end) special-token pair AND whose chat templates carry
// a unique unambiguous signature. Auto-detection is conservative on
// purpose; if a convention's template doesn't fit the paired-marker
// model cleanly, it stays out of auto-detection and operators opt in
// via the CLI `--convention=<name>` override.
//
// Known opt-in-only cases (rationale, in case re-derivation looks
// possible later):
//   - mistral_nemo: opens with `[TOOL_CALLS][` but the closing `]`
//     is the JSON array's closing bracket, not a paired marker token.
//     The paired-markers schema can't represent this without inventing
//     a sentinel; accepting `--convention=mistral_nemo` and pinning
//     start='[TOOL_CALLS]' end='[TOOL_CALLS]' would technically work
//     for raw-ID detection but lies about end-marker semantics.
//   - phi4: the public phi-4 chat template is short enough that it
//     doesn't carry an explicit tool-call marker pair; phi-4-with-
//     tools deployments use a longer template variant. Auto-detection
//     can't tell which variant a deployment will use.
const CONVENTIONS: ConventionEntry[] = [
  {
    convention: 'llama3',
    templateSignature: '<|python_tag|>',
    markers: { start: '<|python_tag|>', end: '<|eom_id|>' },
    args_format: 'python_args',
    result_format: 'json',
  },
  {
    convention: 'qwen25',
    templateSignature: '<tool_call>',
    markers: { start: '<tool_call>', end: '</tool_call>' },
    args_format: 'json',
    result_format: 'json',
  },
  {
    convention: 'deepseek_v3',
    templateSignature: '<｜tool▁calls▁begin｜>',
    markers: {
      start: '<｜tool▁calls▁begin｜>',
      end: '<｜tool▁calls▁end｜>',
    },
    args_format: 'json',
    result_format: 'json',
  },
];

function extractChatTemplate(cfg: HFTokenizerConfig | undefined): string | undefined {
  const t = cfg?.chat_template;
  if (!t) return undefined;
  if (typeof t === 'string') return t;
  if (Array.isArray(t)) return t.map((e) => e.template).join('\n');
  return undefined;
}

/**
 * Resolve markers in special_tokens or vocab; promote vocab-only
 * markers into special_tokens (in-place); return the matching block,
 * or undefined if either marker can't be resolved.
 */
export function deriveToolCalling(
  cfg: HFTokenizerConfig | undefined,
  vocab: Record<string, number>,
  specialTokens: Record<string, number>,
  override?: ToolCallingBlock['convention'],
): ToolCallingBlock | undefined {
  const resolveMarker = (name: string): number | undefined => {
    if (name in specialTokens) return specialTokens[name];
    if (name in vocab) return vocab[name];
    return undefined;
  };
  const tryEntry = (entry: ConventionEntry): ToolCallingBlock | undefined => {
    const startId = resolveMarker(entry.markers.start);
    const endId = resolveMarker(entry.markers.end);
    if (startId === undefined || endId === undefined) return undefined;
    if (!(entry.markers.start in specialTokens)) {
      specialTokens[entry.markers.start] = startId;
    }
    if (!(entry.markers.end in specialTokens)) {
      specialTokens[entry.markers.end] = endId;
    }
    return {
      convention: entry.convention,
      markers: entry.markers,
      args_format: entry.args_format,
      result_format: entry.result_format,
    };
  };
  if (override && override !== 'custom') {
    const entry = CONVENTIONS.find((c) => c.convention === override);
    return entry ? tryEntry(entry) : undefined;
  }
  const template = extractChatTemplate(cfg);
  if (!template) return undefined;
  for (const entry of CONVENTIONS) {
    if (!template.includes(entry.templateSignature)) continue;
    const block = tryEntry(entry);
    if (block) return block;
  }
  return undefined;
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

export function convert(
  hf: HFTokenizerJson,
  modelId: string,
  tokenizerConfig?: HFTokenizerConfig,
  conventionOverride?: ToolCallingBlock['convention'],
): CodecTokenizerMap {
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

  // Derive the tool_calling block when a chat template signals a known
  // convention. The deriver may promote vocab-only markers into
  // special_tokens (in-place) — re-attach the dict to the result so
  // those promotions are persisted even if special_tokens was empty
  // before.
  const toolCalling = deriveToolCalling(
    tokenizerConfig,
    vocab,
    special_tokens,
    conventionOverride,
  );
  if (toolCalling) {
    result.special_tokens = special_tokens;
    result.tool_calling = toolCalling;
  }

  return result;
}

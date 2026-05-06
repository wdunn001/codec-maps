/**
 * fetch.ts
 *
 * Fetches tokenizer.json from HuggingFace for each model in the MODELS list,
 * converts to CodecTokenizerMap, and writes to maps/<org>/<model>.json.
 *
 * Usage:
 *   node --import tsx/esm scripts/fetch.ts
 *   node --import tsx/esm scripts/fetch.ts mistralai/Mistral-7B-v0.3
 */

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { convert } from './convert.ts';

// ── Model list ordered by popularity ────────────────────────────────────────
// Models that share a tokenizer are noted — only one fetch needed.
// Aliases are handled in aliases.json.

const MODELS: Array<{
  hfId: string;          // HuggingFace model ID (org/name)
  mapId: string;         // Canonical Codec map ID
  aliases?: string[];    // Other model IDs that share this exact tokenizer
}> = [
  // ── Llama 3 family — tiktoken BPE, 128,256 tokens ───────────────────────
  {
    hfId: 'meta-llama/Meta-Llama-3.1-8B',
    mapId: 'meta-llama/llama-3',
    aliases: [
      'meta-llama/Meta-Llama-3.1-8B-Instruct',
      'meta-llama/Meta-Llama-3.1-70B',
      'meta-llama/Meta-Llama-3.1-70B-Instruct',
      'meta-llama/Meta-Llama-3.1-405B',
      'meta-llama/Meta-Llama-3.1-405B-Instruct',
      'meta-llama/Meta-Llama-3-8B',
      'meta-llama/Meta-Llama-3-8B-Instruct',
      'meta-llama/Meta-Llama-3-70B',
      'meta-llama/Meta-Llama-3-70B-Instruct',
      'meta-llama/Llama-3.2-1B',
      'meta-llama/Llama-3.2-3B',
      'meta-llama/Llama-3.2-11B-Vision-Instruct',
      'meta-llama/Llama-3.3-70B-Instruct',
    ],
  },

  // ── Qwen 2.5 family — BPE, 151,936 tokens ───────────────────────────────
  {
    hfId: 'Qwen/Qwen2.5-7B-Instruct',
    mapId: 'qwen/qwen2',
    aliases: [
      'Qwen/Qwen2.5-0.5B',
      'Qwen/Qwen2.5-0.5B-Instruct',
      'Qwen/Qwen2.5-1.5B',
      'Qwen/Qwen2.5-1.5B-Instruct',
      'Qwen/Qwen2.5-3B',
      'Qwen/Qwen2.5-3B-Instruct',
      'Qwen/Qwen2.5-7B',
      'Qwen/Qwen2.5-14B',
      'Qwen/Qwen2.5-14B-Instruct',
      'Qwen/Qwen2.5-32B',
      'Qwen/Qwen2.5-32B-Instruct',
      'Qwen/Qwen2.5-72B',
      'Qwen/Qwen2.5-72B-Instruct',
      'Qwen/Qwen2.5-Coder-7B-Instruct',
      'Qwen/Qwen2.5-Coder-32B-Instruct',
      'Qwen/QwQ-32B',
      'Qwen/Qwen2-7B',
      'Qwen/Qwen2-7B-Instruct',
      'Qwen/Qwen2-72B-Instruct',
    ],
  },

  // ── Mistral v0.3 — SentencePiece BPE, 32,768 tokens ────────────────────
  {
    hfId: 'mistralai/Mistral-7B-v0.3',
    mapId: 'mistralai/mistral-v3',
    aliases: [
      'mistralai/Mistral-7B-Instruct-v0.3',
      'mistralai/Mistral-7B-v0.1',
      'mistralai/Mistral-7B-Instruct-v0.1',
      'mistralai/Mistral-7B-Instruct-v0.2',
    ],
  },

  // ── Mistral Nemo — BPE, 131,072 tokens (tekken tokenizer) ───────────────
  {
    hfId: 'mistralai/Mistral-Nemo-Instruct-2407',
    mapId: 'mistralai/mistral-nemo',
    aliases: [
      'mistralai/Mistral-Nemo-Base-2407',
    ],
  },

  // ── Mixtral — same tokenizer as Mistral v0.1 ────────────────────────────
  {
    hfId: 'mistralai/Mixtral-8x7B-v0.1',
    mapId: 'mistralai/mixtral',
    aliases: [
      'mistralai/Mixtral-8x7B-Instruct-v0.1',
      'mistralai/Mixtral-8x22B-v0.1',
      'mistralai/Mixtral-8x22B-Instruct-v0.1',
    ],
  },

  // ── Gemma 2 — SentencePiece, 256,000 tokens ─────────────────────────────
  {
    hfId: 'google/gemma-2-9b',
    mapId: 'google/gemma-2',
    aliases: [
      'google/gemma-2-2b',
      'google/gemma-2-2b-it',
      'google/gemma-2-9b-it',
      'google/gemma-2-27b',
      'google/gemma-2-27b-it',
    ],
  },

  // ── Gemma 1 — SentencePiece, 256,000 tokens ─────────────────────────────
  {
    hfId: 'google/gemma-7b',
    mapId: 'google/gemma-1',
    aliases: [
      'google/gemma-2b',
      'google/gemma-7b-it',
      'google/gemma-2b-it',
    ],
  },

  // ── Phi-3.5 — tiktoken BPE, 32,064 tokens ───────────────────────────────
  {
    hfId: 'microsoft/Phi-3.5-mini-instruct',
    mapId: 'microsoft/phi-3',
    aliases: [
      'microsoft/Phi-3-mini-4k-instruct',
      'microsoft/Phi-3-mini-128k-instruct',
      'microsoft/Phi-3-small-8k-instruct',
      'microsoft/Phi-3-medium-4k-instruct',
      'microsoft/Phi-3.5-MoE-instruct',
    ],
  },

  // ── Phi-4 — BPE, 100,352 tokens ─────────────────────────────────────────
  {
    hfId: 'microsoft/phi-4',
    mapId: 'microsoft/phi-4',
    aliases: [
      'microsoft/phi-4-mini-instruct',
    ],
  },

  // ── DeepSeek-V3 / R1 — BPE, 129,280 tokens ─────────────────────────────
  {
    hfId: 'deepseek-ai/DeepSeek-V3',
    mapId: 'deepseek-ai/deepseek-v3',
    aliases: [
      'deepseek-ai/DeepSeek-R1',
      'deepseek-ai/DeepSeek-R1-Zero',
      'deepseek-ai/DeepSeek-V3-Base',
    ],
  },

  // ── Llama 2 family — SentencePiece BPE, 32,000 tokens ───────────────────
  {
    hfId: 'meta-llama/Llama-2-7b-hf',
    mapId: 'meta-llama/llama-2',
    aliases: [
      'meta-llama/Llama-2-7b-chat-hf',
      'meta-llama/Llama-2-13b-hf',
      'meta-llama/Llama-2-13b-chat-hf',
      'meta-llama/Llama-2-70b-hf',
      'meta-llama/Llama-2-70b-chat-hf',
    ],
  },

  // ── Falcon — BPE, 65,024 tokens ─────────────────────────────────────────
  {
    hfId: 'tiiuae/falcon-7b',
    mapId: 'tiiuae/falcon',
    aliases: [
      'tiiuae/falcon-7b-instruct',
      'tiiuae/falcon-40b',
      'tiiuae/falcon-180B',
    ],
  },

  // ── Codestral / Mistral Codestral ────────────────────────────────────────
  {
    hfId: 'mistralai/Codestral-22B-v0.1',
    mapId: 'mistralai/codestral',
    aliases: [
      'mistralai/Codestral-Mamba-7B-v0.1',
    ],
  },

  // ── SmolLM2 — BPE, 49,152 tokens ────────────────────────────────────────
  {
    hfId: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    mapId: 'huggingfacetb/smollm2',
    aliases: [
      'HuggingFaceTB/SmolLM2-135M',
      'HuggingFaceTB/SmolLM2-360M',
    ],
  },
];

// ── Fetch helpers ────────────────────────────────────────────────────────────

const HF_BASE = 'https://huggingface.co';

async function fetchTokenizerJson(hfId: string): Promise<unknown> {
  const url = `${HF_BASE}/${hfId}/resolve/main/tokenizer.json`;
  console.log(`  GET ${url}`);
  const resp = await fetch(url, {
    headers: { 'User-Agent': 'codecai-maps/0.1 (+https://github.com/wdunn001/Codec)' },
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${hfId}`);
  return resp.json();
}

async function sha256(data: string): Promise<string> {
  const bytes = Buffer.from(data);
  const hash = crypto.createHash('sha256').update(bytes).digest('hex');
  return `sha256:${hash}`;
}

// ── Main ─────────────────────────────────────────────────────────────────────

const MAPS_DIR = new URL('../maps', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');

async function processModel(entry: (typeof MODELS)[number]) {
  const { hfId, mapId, aliases = [] } = entry;
  const outPath = path.join(MAPS_DIR, `${mapId}.json`);

  console.log(`\n▶ ${mapId}  (from ${hfId})`);

  // Create output directory
  await fs.mkdir(path.dirname(outPath), { recursive: true });

  // Fetch tokenizer.json
  let hfJson: unknown;
  try {
    hfJson = await fetchTokenizerJson(hfId);
  } catch (err) {
    console.error(`  ✗ fetch failed: ${err}`);
    return null;
  }

  // Convert
  let map: ReturnType<typeof convert>;
  try {
    map = convert(hfJson as Parameters<typeof convert>[0], mapId);
  } catch (err) {
    console.error(`  ✗ conversion failed: ${err}`);
    return null;
  }

  const json = JSON.stringify(map, null, 2);
  const hash = await sha256(json);

  // Write map file
  await fs.writeFile(outPath, json, 'utf-8');
  console.log(`  ✓ written  ${outPath}`);
  console.log(`    vocab_size=${map.vocab_size}  tokens=${Object.keys(map.tokens).length}  hash=${hash}`);
  if (map.byte_fallback_start !== undefined) {
    console.log(`    byte_fallback: ${map.byte_fallback_start}–${map.byte_fallback_end}`);
  }

  return { mapId, hash, aliases };
}

async function main() {
  const target = process.argv[2]; // optional: single model to fetch
  const list = target
    ? MODELS.filter((m) => m.hfId === target || m.mapId === target)
    : MODELS;

  if (list.length === 0) {
    console.error(`No model matching: ${target}`);
    process.exit(1);
  }

  const aliasMap: Record<string, string> = {};
  const hashIndex: Record<string, string> = {};

  for (const entry of list) {
    const result = await processModel(entry);
    if (result) {
      hashIndex[result.mapId] = result.hash;
      for (const alias of result.aliases) {
        aliasMap[alias] = result.mapId;
      }
    }
  }

  // Write/update aliases.json
  const aliasPath = path.join(MAPS_DIR, '..', 'aliases.json');
  let existing: Record<string, string> = {};
  try {
    existing = JSON.parse(await fs.readFile(aliasPath, 'utf-8'));
  } catch {}
  const merged = { ...existing, ...aliasMap };
  await fs.writeFile(aliasPath, JSON.stringify(merged, null, 2) + '\n', 'utf-8');
  console.log(`\n✓ aliases.json updated (${Object.keys(merged).length} aliases)`);

  // Write/update index.json with hashes (for clients to pin)
  const indexPath = path.join(MAPS_DIR, '..', 'index.json');
  let existingIndex: Record<string, string> = {};
  try {
    existingIndex = JSON.parse(await fs.readFile(indexPath, 'utf-8'));
  } catch {}
  const mergedIndex = { ...existingIndex, ...hashIndex };
  await fs.writeFile(indexPath, JSON.stringify(mergedIndex, null, 2) + '\n', 'utf-8');
  console.log(`✓ index.json updated (${Object.keys(mergedIndex).length} maps)`);

  console.log('\nDone.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

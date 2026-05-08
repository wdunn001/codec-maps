/**
 * fetch-tiktoken.ts — pull OpenAI .tiktoken files and emit CodecTokenizerMap
 * JSON files under maps/openai/.
 *
 * Sibling of fetch.ts (which mirrors HuggingFace tokenizers). Same output
 * shape — same v2 TokenizerMap, same downstream contract — just a different
 * upstream source. The OpenAI BPE tables are public and MIT-licensed via
 * tiktoken; this script pulls them from openaipublic.blob.core.windows.net.
 *
 * Usage:
 *   node --import tsx/esm scripts/fetch-tiktoken.ts
 *   node --import tsx/esm scripts/fetch-tiktoken.ts cl100k_base
 */

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { convertTiktoken, ENCODINGS } from './convert-tiktoken.ts';

// ── Encoding → map metadata ──────────────────────────────────────────────────

interface MapEntry {
  encoding: keyof typeof ENCODINGS;
  mapId: string;
  aliases?: string[];
}

const MAPS: MapEntry[] = [
  // ── cl100k_base — gpt-3.5-turbo + gpt-4 family ────────────────────────────
  {
    encoding: 'cl100k_base',
    mapId: 'openai/cl100k_base',
    aliases: [
      'openai/gpt-3.5-turbo',
      'openai/gpt-3.5-turbo-16k',
      'openai/gpt-4',
      'openai/gpt-4-turbo',
      'openai/gpt-4-32k',
      'openai/text-embedding-ada-002',
      'openai/text-embedding-3-small',
      'openai/text-embedding-3-large',
    ],
  },

  // ── o200k_base — gpt-4o + o1/o3/o4 reasoning family ───────────────────────
  {
    encoding: 'o200k_base',
    mapId: 'openai/o200k_base',
    aliases: [
      'openai/gpt-4o',
      'openai/gpt-4o-mini',
      'openai/o1',
      'openai/o1-mini',
      'openai/o1-preview',
      'openai/o3',
      'openai/o3-mini',
      'openai/o4-mini',
    ],
  },

  // ── p50k_base — Codex / text-davinci-002/003 ─────────────────────────────
  {
    encoding: 'p50k_base',
    mapId: 'openai/p50k_base',
    aliases: [
      'openai/code-davinci-002',
      'openai/text-davinci-002',
      'openai/text-davinci-003',
    ],
  },

  // ── p50k_edit — text/code-davinci-edit ────────────────────────────────────
  {
    encoding: 'p50k_edit',
    mapId: 'openai/p50k_edit',
    aliases: [
      'openai/text-davinci-edit-001',
      'openai/code-davinci-edit-001',
    ],
  },

  // ── r50k_base — gpt-2 / gpt-3 (legacy) ────────────────────────────────────
  {
    encoding: 'r50k_base',
    mapId: 'openai/r50k_base',
    aliases: [
      'openai/gpt2',
      'openai/davinci',
      'openai/curie',
      'openai/babbage',
      'openai/ada',
    ],
  },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..');
const MAPS_DIR = path.join(REPO_ROOT, 'maps');

async function sha256(data: string): Promise<string> {
  return 'sha256:' + crypto.createHash('sha256').update(Buffer.from(data)).digest('hex');
}

async function processEncoding(entry: MapEntry): Promise<{
  mapId: string;
  hash: string;
  aliases: string[];
} | null> {
  const spec = ENCODINGS[entry.encoding];
  const outPath = path.join(MAPS_DIR, `${entry.mapId}.json`);

  console.log(`\n▶ ${entry.mapId}  (encoding: ${entry.encoding})`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });

  // Fetch raw .tiktoken bytes from the OpenAI public CDN.
  console.log(`  GET ${spec.url}`);
  const resp = await fetch(spec.url, {
    headers: { 'User-Agent': 'codec-maps/0.1 (+https://github.com/wdunn001/codec-maps)' },
  });
  if (!resp.ok) {
    console.error(`  ✗ HTTP ${resp.status} — skipping`);
    return null;
  }
  const buf = new Uint8Array(await resp.arrayBuffer());
  console.log(`  bytes: ${buf.length}`);

  // Convert.
  const map = convertTiktoken(buf, {
    id: entry.mapId,
    encoding: entry.encoding,
  });

  // Persist + hash.
  const json = JSON.stringify(map, null, 2);
  await fs.writeFile(outPath, json, 'utf-8');
  const hash = await sha256(json);

  console.log(`  ✓ written  ${outPath}`);
  console.log(
    `    vocab_size=${map.vocab_size}  merges=${map.merges?.length ?? 0}` +
      `  encoder=${map.encoder ?? 'identity'}  hash=${hash}`,
  );
  console.log(
    `    tool_calling: omitted (OpenAI doesn't ship a chat_template signature ` +
      `we can derive from)`,
  );

  return { mapId: entry.mapId, hash, aliases: entry.aliases ?? [] };
}

async function main() {
  const target = process.argv[2];
  const list = target
    ? MAPS.filter((m) => m.encoding === target || m.mapId === target)
    : MAPS;

  if (list.length === 0) {
    console.error(`No matching map for ${target}. Known: ${MAPS.map((m) => m.encoding).join(', ')}`);
    process.exit(1);
  }

  const results: Array<{ mapId: string; hash: string; aliases: string[] }> = [];
  for (const entry of list) {
    const r = await processEncoding(entry);
    if (r) results.push(r);
  }

  // ── Update aliases.json + index.json ─────────────────────────────────────
  const aliasesPath = path.join(REPO_ROOT, 'aliases.json');
  const indexPath = path.join(REPO_ROOT, 'index.json');

  let aliases: Record<string, string> = {};
  try {
    aliases = JSON.parse(await fs.readFile(aliasesPath, 'utf-8'));
  } catch {
    /* empty */
  }
  let index: Record<string, string> = {};
  try {
    index = JSON.parse(await fs.readFile(indexPath, 'utf-8'));
  } catch {
    /* empty */
  }

  for (const r of results) {
    index[r.mapId] = r.hash;
    for (const a of r.aliases) aliases[a] = r.mapId;
  }

  await fs.writeFile(aliasesPath, JSON.stringify(aliases, null, 2) + '\n', 'utf-8');
  await fs.writeFile(indexPath, JSON.stringify(index, null, 2) + '\n', 'utf-8');

  console.log(`\n✓ aliases.json updated (${Object.keys(aliases).length} aliases)`);
  console.log(`✓ index.json updated (${Object.keys(index).length} maps)`);

  console.log('\nDone.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

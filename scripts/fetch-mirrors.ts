/**
 * fetch-mirrors.ts
 * Fetches gated model tokenizers via public community mirrors on HuggingFace.
 */
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { convert } from './convert.ts';

const MAPS_DIR = 'H:/dev/codec-maps/maps';

const MIRRORS = [
  {
    url: 'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/resolve/main/tokenizer.json',
    mapId: 'meta-llama/llama-3',
    aliases: [
      'meta-llama/Meta-Llama-3.1-8B',
      'meta-llama/Meta-Llama-3.1-8B-Instruct',
      'meta-llama/Meta-Llama-3.1-70B',
      'meta-llama/Meta-Llama-3.1-70B-Instruct',
      'meta-llama/Meta-Llama-3.1-405B',
      'meta-llama/Meta-Llama-3-8B',
      'meta-llama/Meta-Llama-3-8B-Instruct',
      'meta-llama/Meta-Llama-3-70B',
      'meta-llama/Llama-3.2-1B',
      'meta-llama/Llama-3.2-3B',
      'meta-llama/Llama-3.3-70B-Instruct',
    ],
  },
  {
    url: 'https://huggingface.co/NousResearch/Llama-2-7b-hf/resolve/main/tokenizer.json',
    mapId: 'meta-llama/llama-2',
    aliases: [
      'meta-llama/Llama-2-7b-hf',
      'meta-llama/Llama-2-13b-hf',
      'meta-llama/Llama-2-70b-hf',
      'meta-llama/Llama-2-7b-chat-hf',
      'meta-llama/Llama-2-13b-chat-hf',
    ],
  },
  {
    url: 'https://huggingface.co/unsloth/gemma-2-9b/resolve/main/tokenizer.json',
    mapId: 'google/gemma-2',
    aliases: [
      'google/gemma-2-2b',
      'google/gemma-2-2b-it',
      'google/gemma-2-9b',
      'google/gemma-2-9b-it',
      'google/gemma-2-27b',
      'google/gemma-2-27b-it',
    ],
  },
  {
    url: 'https://huggingface.co/unsloth/gemma-7b/resolve/main/tokenizer.json',
    mapId: 'google/gemma-1',
    aliases: [
      'google/gemma-2b',
      'google/gemma-7b',
      'google/gemma-7b-it',
      'google/gemma-2b-it',
    ],
  },
];

const aliasMap: Record<string, string> = {};
const hashIndex: Record<string, string> = {};

for (const { url, mapId, aliases } of MIRRORS) {
  console.log('\n▶', mapId);
  console.log('  GET', url);
  try {
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'codecai-maps/0.1 (+https://github.com/wdunn001/Codec)' },
    });
    if (!resp.ok) {
      console.error('  ✗ HTTP', resp.status);
      continue;
    }
    const hfJson = await resp.json() as Parameters<typeof convert>[0];
    const map = convert(hfJson, mapId);
    const json = JSON.stringify(map, null, 2);
    const hash = 'sha256:' + crypto.createHash('sha256').update(Buffer.from(json)).digest('hex');
    const outPath = path.join(MAPS_DIR, mapId + '.json');
    await fs.mkdir(path.dirname(outPath), { recursive: true });
    await fs.writeFile(outPath, json, 'utf-8');
    console.log(
      '  ✓ vocab_size=' + map.vocab_size +
      '  vocab=' + Object.keys(map.vocab).length +
      '  merges=' + (map.merges?.length ?? 0) +
      '  encoder=' + (map.encoder ?? 'identity') +
      '  hash=' + hash,
    );
    if (map.byte_fallback_start !== undefined) {
      console.log('    byte_fallback:', map.byte_fallback_start + '-' + map.byte_fallback_end);
    }
    hashIndex[mapId] = hash;
    for (const a of aliases) aliasMap[a] = mapId;
  } catch (e: unknown) {
    console.error('  ✗', (e as Error).message);
  }
}

const indexPath = 'H:/dev/codec-maps/index.json';
const aliasPath = 'H:/dev/codec-maps/aliases.json';
const ei: Record<string, string> = JSON.parse(await fs.readFile(indexPath, 'utf-8').catch(() => '{}'));
const ea: Record<string, string> = JSON.parse(await fs.readFile(aliasPath, 'utf-8').catch(() => '{}'));
await fs.writeFile(indexPath, JSON.stringify({ ...ei, ...hashIndex }, null, 2) + '\n');
await fs.writeFile(aliasPath, JSON.stringify({ ...ea, ...aliasMap }, null, 2) + '\n');
console.log('\nDone. total maps:', Object.keys({ ...ei, ...hashIndex }).length);

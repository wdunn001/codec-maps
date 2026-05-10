/**
 * sync-specials.ts
 *
 * Walk every map under maps/ and promote any vocab key in `<|...|>`
 * shape that isn't already in `special_tokens` into the
 * `special_tokens` field. Updates index.json with the new content
 * hashes.
 *
 * Why: published maps shipped before some chat-template revisions
 * carry the delimiters in `vocab` (so encode/decode works in the
 * tokenizers themselves once the special-token pre-scan is in place)
 * but DON'T list them in `special_tokens`. Anything keying on
 * `special_tokens` for delimiter discovery (ToolWatcher, chat-template
 * parsers, safety classifiers) misses them otherwise. Qwen-2.5
 * specifically dropped 6 FIM specials this way: `<|fim_prefix|>`,
 * `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>`,
 * `<|file_sep|>`.
 *
 *   npx tsx scripts/sync-specials.ts           # dry-run + report
 *   npx tsx scripts/sync-specials.ts --write   # rewrite maps + index.json
 */

import { createHash } from 'node:crypto';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { argv, cwd, exit } from 'node:process';

const ROOT = cwd();
const MAPS_DIR = join(ROOT, 'maps');
const INDEX_PATH = join(ROOT, 'index.json');
const WRITE = argv.includes('--write');

interface Map {
  id: string;
  vocab?: Record<string, number>;
  special_tokens?: Record<string, number>;
  [k: string]: unknown;
}

/**
 * Match `<|body|>` where `body` is non-empty and identifier-like
 * (letters/digits/`_`/`-`). Catches every shipped chat-template and
 * tool-call delimiter while excluding pathological vocab BPE tokens
 * like `<|>` or `<| |>` that happen to share the start/end pair.
 */
function isDelimiterShape(tok: string): boolean {
  if (tok.length <= 4) return false; // `<|>` and `<||>` are not specials.
  if (!tok.startsWith('<|') || !tok.endsWith('|>')) return false;
  const body = tok.slice(2, -2);
  return /^[A-Za-z0-9_\-]+$/.test(body);
}

async function walk(dir: string): Promise<string[]> {
  const out: string[] = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(p)));
    else if (entry.isFile() && entry.name.endsWith('.json')) out.push(p);
  }
  return out;
}

function canonicalBytes(obj: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(obj, null, 2));
}

function sha256(bytes: Uint8Array): string {
  return 'sha256:' + createHash('sha256').update(bytes).digest('hex');
}

async function main(): Promise<void> {
  const files = await walk(MAPS_DIR);
  const indexBytes = await readFile(INDEX_PATH);
  const index = JSON.parse(indexBytes.toString('utf-8')) as Record<string, string>;

  const updates: Array<{
    file: string;
    id: string;
    oldHash: string;
    newHash: string;
    added: Array<[string, number]>;
  }> = [];
  const skipped: Array<{ file: string; id: string; reason: string }> = [];

  for (const file of files) {
    const raw = await readFile(file);
    const map = JSON.parse(raw.toString('utf-8')) as Map;
    const rel = relative(ROOT, file);

    if (!map.vocab) {
      skipped.push({ file: rel, id: map.id, reason: 'no vocab' });
      continue;
    }

    const existing = new Set(Object.keys(map.special_tokens ?? {}));
    const additions: Array<[string, number]> = [];
    for (const [tok, id] of Object.entries(map.vocab)) {
      if (existing.has(tok)) continue;
      if (isDelimiterShape(tok)) additions.push([tok, id]);
    }

    if (additions.length === 0) {
      skipped.push({ file: rel, id: map.id, reason: 'no new specials to add' });
      continue;
    }

    // Append the new specials to special_tokens in ID order. Preserve the
    // overall map insertion order for hash stability.
    additions.sort((a, b) => a[1] - b[1]);
    const updatedSpecials: Record<string, number> = { ...(map.special_tokens ?? {}) };
    for (const [tok, id] of additions) updatedSpecials[tok] = id;

    const updated: Record<string, unknown> = { ...map, special_tokens: updatedSpecials };

    const oldHash = sha256(raw);
    const newBytes = canonicalBytes(updated);
    const newHash = sha256(newBytes);

    updates.push({ file: rel, id: map.id, oldHash, newHash, added: additions });

    if (WRITE) {
      await writeFile(file, newBytes);
      index[map.id] = newHash;
    }
  }

  console.log('=== updates ===');
  for (const u of updates) {
    console.log(`  ${u.id.padEnd(28)} ${u.file}`);
    console.log(`    old: ${u.oldHash}`);
    console.log(`    new: ${u.newHash}`);
    console.log(`    added (${u.added.length}):`);
    for (const [tok, id] of u.added) {
      console.log(`      ${id}: ${tok}`);
    }
  }
  console.log(`\n${updates.length} maps ${WRITE ? 'updated' : 'would update'}`);

  console.log('\n=== skipped ===');
  for (const s of skipped) {
    console.log(`  ${s.id.padEnd(28)} ${s.reason}`);
  }
  console.log(`${skipped.length} maps skipped`);

  if (WRITE) {
    await writeFile(INDEX_PATH, JSON.stringify(index, null, 2));
    console.log(`\nindex.json rewritten with ${updates.length} new hashes`);
  } else {
    console.log('\n(dry-run) re-run with --write to apply');
  }
}

main().catch((err) => {
  console.error(err);
  exit(1);
});

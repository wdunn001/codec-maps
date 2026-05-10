/**
 * add-pretok-program.ts
 *
 * One-off backfill: walk every map under maps/, for any byte_level map
 * that carries `pre_tokenizer_pattern` but not `pre_tokenizer_program`,
 * compile the pattern with @codecai/maps-cli's recogniser and add the
 * resulting op-list. Updates index.json with the new content hashes.
 *
 * Why: published GPT-2-family maps (qwen2, llama-3, phi-4, cl100k,
 * o200k) use `(?i:...)` inline-flag groups that throw on Chrome <125,
 * iOS Safari <18, Firefox <132, and Node <23 — every mobile-leaning
 * runtime. Maps with `pre_tokenizer_program` bypass the regex path
 * entirely, so adding the field unblocks BPETokenizer.encode() on
 * those runtimes.
 *
 *   npx tsx scripts/add-pretok-program.ts            # dry-run + report
 *   npx tsx scripts/add-pretok-program.ts --write    # rewrite maps + index.json
 */

import { createHash } from 'node:crypto';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { argv, cwd, exit } from 'node:process';

import { compilePreTokenizerRegex } from '../../Project Codec/Codec/packages/maps-cli/src/compile-pretok.ts';

const ROOT = cwd();
const MAPS_DIR = join(ROOT, 'maps');
const INDEX_PATH = join(ROOT, 'index.json');
const WRITE = argv.includes('--write');

interface Map {
  id: string;
  version?: string;
  encoder?: string;
  pre_tokenizer_pattern?: string;
  pre_tokenizer_program?: unknown;
  [k: string]: unknown;
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

  const updates: Array<{ file: string; id: string; oldHash: string; newHash: string }> = [];
  const skipped: Array<{ file: string; id: string; reason: string }> = [];

  for (const file of files) {
    const raw = await readFile(file);
    const map = JSON.parse(raw.toString('utf-8')) as Map;
    const rel = relative(ROOT, file);

    if (map.encoder !== 'byte_level') {
      skipped.push({ file: rel, id: map.id, reason: `encoder=${map.encoder}` });
      continue;
    }
    if (map.pre_tokenizer_program) {
      skipped.push({ file: rel, id: map.id, reason: 'already has program' });
      continue;
    }
    if (!map.pre_tokenizer_pattern) {
      skipped.push({ file: rel, id: map.id, reason: 'no pre_tokenizer_pattern' });
      continue;
    }

    const program = compilePreTokenizerRegex(map.pre_tokenizer_pattern);
    if (!program) {
      skipped.push({ file: rel, id: map.id, reason: 'compiler did not recognise pattern' });
      continue;
    }

    // Insert pre_tokenizer_program adjacent to pre_tokenizer_pattern for
    // diff readability. Hash is order-sensitive (JSON.stringify with
    // indent preserves insertion order) so be deliberate.
    const updated: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(map)) {
      updated[k] = v;
      if (k === 'pre_tokenizer_pattern') {
        updated.pre_tokenizer_program = program;
      }
    }

    const oldHash = sha256(raw);
    const newBytes = canonicalBytes(updated);
    const newHash = sha256(newBytes);

    updates.push({ file: rel, id: map.id, oldHash, newHash });

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

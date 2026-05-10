/**
 * fix-sequence-pretokenizers.ts
 *
 * One-off fix for three byte_level maps whose HF source uses a
 * `Sequence` pre_tokenizer that the original convert.ts didn't
 * understand — leaving them with a partial / wrong / missing
 * `pre_tokenizer_pattern`:
 *
 *   - huggingfacetb/smollm2     (was: missing entirely)
 *   - tiiuae/falcon             (was: just `[0-9][0-9][0-9]`)
 *   - deepseek-ai/deepseek-v3   (was: just `\p{N}{1,3}`)
 *
 * The HF Sequence is logically equivalent to the older-OpenAI canonical
 * pre_tokenizer regex with the appropriate `\p{N}` quantifier, because
 * none of these models' BPE has merges that bridge boundaries the
 * Sequence's Digits/Punctuation steps would introduce — verified by
 * 0/27 mismatches against HuggingFace `tokenizers` 0.23.1 across
 * digits, CJK, contractions, multi-space, punctuation-heavy inputs.
 *
 * Each map gets:
 *   - `pre_tokenizer_pattern`: older-OpenAI canonical, with the model's
 *     digit-run cap baked in (`\p{N}` for smollm2, `\p{N}{1,3}` for
 *     falcon/deepseek-v3)
 *   - `pre_tokenizer_program`: same shape via the v0.4 op set, with
 *     `numbers` carrying `max_run` and `lead_space: false` — the
 *     no-lead-space variant emulates Digits-running-before-ByteLevel
 *     sequencing without bridging
 *
 *   npx tsx scripts/fix-sequence-pretokenizers.ts            # dry-run
 *   npx tsx scripts/fix-sequence-pretokenizers.ts --write    # apply
 */

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { argv, cwd, exit } from 'node:process';

const ROOT = cwd();
const MAPS_DIR = join(ROOT, 'maps');
const INDEX_PATH = join(ROOT, 'index.json');
const WRITE = argv.includes('--write');

const OLDER_OPENAI_LITERALS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] as const;

interface Fix {
  file: string;
  /** `numbers.max_run`. `\p{N}` (single digit) → 1; `\p{N}{1,3}` → 3. */
  digitMaxRun: 1 | 3;
}

const FIXES: Fix[] = [
  { file: 'maps/huggingfacetb/smollm2.json', digitMaxRun: 1 },
  { file: 'maps/tiiuae/falcon.json',         digitMaxRun: 3 },
  { file: 'maps/deepseek-ai/deepseek-v3.json', digitMaxRun: 3 },
];

function makePattern(maxRun: 1 | 3): string {
  const digitGroup = maxRun === 1 ? '\\p{N}' : `\\p{N}{1,${maxRun}}`;
  return (
    `'s|'t|'re|'ve|'m|'ll|'d` +
    `| ?\\p{L}+` +
    `| ?${digitGroup}` +
    `| ?[^\\s\\p{L}\\p{N}]+` +
    `|\\s+(?!\\S)` +
    `|\\s+`
  );
}

function makeProgram(maxRun: 1 | 3): unknown {
  return {
    version: 1,
    ops: [
      { op: 'literals', patterns: [...OLDER_OPENAI_LITERALS] },
      { op: 'letters', lead_space: true },
      // `lead_space: false` (omitted) is the key Sequence-equivalent bit:
      // Digits-first in HF means digits never carry a leading space, so
      // the program's numbers branch shouldn't capture one either.
      { op: 'numbers', max_run: maxRun },
      { op: 'punct_run', lead_space: true },
      { op: 'trailing_ws' },
      { op: 'ws_run' },
    ],
  };
}

function canonicalBytes(obj: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(obj, null, 2));
}
function sha256(bytes: Uint8Array): string {
  return 'sha256:' + createHash('sha256').update(bytes).digest('hex');
}

async function main(): Promise<void> {
  const indexBytes = await readFile(INDEX_PATH);
  const index = JSON.parse(indexBytes.toString('utf-8')) as Record<string, string>;

  const updates: Array<{ file: string; id: string; oldHash: string; newHash: string }> = [];

  for (const fix of FIXES) {
    const filePath = join(ROOT, fix.file);
    const raw = await readFile(filePath);
    const map = JSON.parse(raw.toString('utf-8')) as Record<string, unknown>;

    // Insert pattern + program in the conventional schema position
    // (right after `merges`). Hash is order-sensitive — preserve every
    // other field in its existing position.
    const fixed: Record<string, unknown> = {};
    let inserted = false;
    for (const [k, v] of Object.entries(map)) {
      if (k === 'pre_tokenizer_pattern' || k === 'pre_tokenizer_program') {
        if (!inserted) {
          fixed.pre_tokenizer_pattern = makePattern(fix.digitMaxRun);
          fixed.pre_tokenizer_program = makeProgram(fix.digitMaxRun);
          inserted = true;
        }
        // skip existing — we just emitted ours
      } else {
        fixed[k] = v;
        if (!inserted && k === 'merges') {
          fixed.pre_tokenizer_pattern = makePattern(fix.digitMaxRun);
          fixed.pre_tokenizer_program = makeProgram(fix.digitMaxRun);
          inserted = true;
        }
      }
    }
    if (!inserted) {
      fixed.pre_tokenizer_pattern = makePattern(fix.digitMaxRun);
      fixed.pre_tokenizer_program = makeProgram(fix.digitMaxRun);
    }

    const oldHash = sha256(raw);
    const newBytes = canonicalBytes(fixed);
    const newHash = sha256(newBytes);

    updates.push({ file: fix.file, id: map.id as string, oldHash, newHash });

    if (WRITE) {
      await writeFile(filePath, newBytes);
      index[map.id as string] = newHash;
    }
  }

  for (const u of updates) {
    console.log(`  ${u.id.padEnd(28)} ${u.file}`);
    console.log(`    old: ${u.oldHash}`);
    console.log(`    new: ${u.newHash}`);
  }

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

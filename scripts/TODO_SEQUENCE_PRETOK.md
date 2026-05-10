# TODO: Sequence-pretokenizer support for codec-maps

Three maps in this repo carry pre_tokenizer_pattern that doesn't match
what HuggingFace `tokenizers` actually produces, because HF stores their
pre-tokenizer as a **Sequence** of multiple pretokenizers and the
original `scripts/convert.ts` only captured one Split's regex:

| Map | HF actual pretokenizer | Currently stored pattern |
|---|---|---|
| `deepseek-ai/deepseek-v3` | Sequence[Split(\p{N}{1,3}), Split(CJK ranges), Split(GPT-2 variant)] | `\p{N}{1,3}` (just the digit Split) |
| `huggingfacetb/smollm2` | Sequence[Digits(individual_digits=true), ByteLevel(use_regex=true)] | _empty_ (no pattern at all) |
| `tiiuae/falcon` | Sequence[Punctuation(Contiguous), ByteLevel(use_regex=true), Digits, Split([0-9][0-9][0-9])] | `[0-9][0-9][0-9]` (just the trailing Split) |

## Why this isn't a "fix the pattern" job

A single regex can't reproduce Sequence semantics in general:

- **smollm2**: Digits(individual_digits=true) splits "1 2 3" → ["1", " ", "2", " ", "3"]
  BEFORE the GPT-2 canonical regex runs. With a single regex like
  `\p{N}| ?\p{L}+|\s+|...` the leading space would attach to the digit
  (` 2` as one piece), producing 3 pieces instead of HF's 5. Verified
  locally: my best single-regex attempt for smollm2 produces `['1',
  ' 2', ' 3']` where HF produces `['1', 'Ġ', '2', 'Ġ', '3']`.
- **falcon**: Punctuation(Contiguous) isolates each `'` from contractions,
  so `isn't` tokenises as `is n ' t` — five pieces, not three. The GPT-2
  canonical regex's contractions branch `'s|'t|...` keeps them together.
  No single regex can express both behaviors simultaneously.
- **deepseek-v3**: The third Split's regex (which IS expressible
  as-is) handles letters and punct. Combined with the first Split's
  `\p{N}{1,3}` for digits and the second Split's CJK ranges, it's a
  three-way split that needs to apply in order. The third Split alone
  isn't byte-identical to HF for digit-heavy or CJK-heavy text.

## What this needs

Schema extension on TokenizerMap to carry a sequence of pre-tokenizer
ops, mirroring HF's structure:

```jsonc
{
  // existing fields...
  "pre_tokenizer_sequence": [
    { "type": "digits", "individual_digits": true },
    { "type": "split", "pattern": "..." },
    { "type": "byte_level_regex" }
  ]
}
```

Each pre-tokenizer kind (`punctuation`, `digits`, `split`,
`byte_level_regex`, `whitespace_split`) needs runtime support in every
language client. The interpreter applies them in order: each one
operates on the pieces produced by the previous one.

## Path

1. Extend `spec/tokenizer-map.schema.json` with `pre_tokenizer_sequence`
   (optional, present only on Sequence-based tokenizers).
2. Update `scripts/convert.ts` in this repo to detect Sequence
   pretokenizers and emit the new field. Re-run conversion for
   smollm2 / deepseek-v3 / falcon.
3. Port the new runtime to TS / Python / Rust / .NET / Java / C in
   `Codec/packages/*`.
4. Test against HF for each affected map.

Tracked as v2.2 of the tokenizer-map schema (additive — old clients
ignore the new field and continue using `pre_tokenizer_pattern`).

## Current workaround for users

For the three affected maps, encoding via @codecai/web / codecai
BPETokenizer produces wrong IDs on inputs that hit the un-captured
Sequence steps:

- **smollm2**: wrong on inputs with digits (treats `\p{N}+` as a single
  run instead of per-digit). Non-digit text tokenises correctly.
- **deepseek-v3**: wrong on inputs with letters (current pattern only
  matches digit runs, so letters fall through to the defensive single-
  cp fallback).
- **falcon**: wrong on inputs with punctuation runs longer than 3
  digits (only matches `[0-9][0-9][0-9]`). Falcon's pre-tokenizer is
  the most divergent — Punctuation(Contiguous) isolates each `'`
  so contractions like `isn't` tokenise as 5 pieces, not the 3 that
  the GPT-2 canonical regex would produce.

Until this is fixed, consumers needing exact HF parity for these
models should use HuggingFace `tokenizers` directly rather than
@codecai/web encoding.

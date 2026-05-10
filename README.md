# codec-maps

**Community registry of tokenizer dialect maps for the [Codec](https://github.com/wdunn001/Codec) binary transport protocol.**

Each map is a content-addressed JSON file describing one model's tokenizer
vocabulary. Clients use it to decode binary token streams back into text
([`@codecai/web` Detokenizer](https://www.npmjs.com/package/@codecai/web))
and to tokenize user input on the way in
([`@codecai/web` BPETokenizer](https://www.npmjs.com/package/@codecai/web)).

Think of this repo as DefinitelyTyped for LLM token vocabularies.

## What's here

14 model families covering 70+ aliases:

| Family             | encoder      | vocab size | merges  | Notes                    |
|--------------------|--------------|-----------:|--------:|--------------------------|
| `meta-llama/llama-3` | byte_level   |    128,256 | 280,147 | Llama 3.x family         |
| `meta-llama/llama-2` | metaspace    |     32,000 |  61,249 | Llama 2 family           |
| `qwen/qwen2`         | byte_level   |    151,665 | 151,387 | Qwen 2 / 2.5 family      |
| `mistralai/mistral-v3` | metaspace  |     32,768 |  58,980 | Mistral v0.3 family      |
| `mistralai/mixtral`  | metaspace    |     32,000 |  61,249 | Mixtral 8x7B / 8x22B     |
| `mistralai/codestral`| metaspace    |     32,768 |  58,980 | Codestral 22B            |
| `mistralai/mistral-nemo` | byte_level | 131,072 | (varies) | Mistral Nemo / Large     |
| `microsoft/phi-3`    | metaspace    |     32,011 |  61,249 | Phi-3.x family           |
| `microsoft/phi-4`    | byte_level   |    100,352 | 100,000 | Phi-4 family             |
| `google/gemma-1`     | metaspace    |    256,000 | 580,604 | Gemma 1 family           |
| `google/gemma-2`     | metaspace    |    256,000 | 580,604 | Gemma 2 family           |
| `deepseek-ai/deepseek-v3` | byte_level | 128,815 | 127,741 | DeepSeek-V3 / V3.5       |
| `tiiuae/falcon`      | byte_level   |     65,024 |  64,784 | Falcon family            |
| `huggingfacetb/smollm2` | byte_level |     49,152 |  48,900 | SmolLM2 family           |

See [`index.json`](./index.json) for the full machine-readable list with
URLs and content hashes. See [`aliases.json`](./aliases.json) for the alias
table that resolves things like `meta-llama/Llama-3.1-8B-Instruct` to
`meta-llama/llama-3`.

## Use a map

The recommended access pattern is via `@codecai/web`'s `loadMap()`, which
fetches and verifies the sha256 hash:

```ts
import { loadMap, Detokenizer, decodeStream } from '@codecai/web';

const map = await loadMap({
  url:  'https://cdn.jsdelivr.net/gh/wdunn001/codec-maps/maps/qwen/qwen2.json',
  hash: 'sha256:887311099cdc09e7022001a01fa1da396750d669b7ed2c242a000b9badd09791',  // from index.json
});

const detok = new Detokenizer(map);

// Stream from a Codec-compliant server (vLLM, SGLang)
const resp = await fetch('http://your-server/v1/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'Qwen/Qwen2.5-7B-Instruct',
    prompt: 'Explain entropy.',
    stream_format: 'msgpack',
    max_tokens: 256,
  }),
});

for await (const frame of decodeStream(resp.body!, 'msgpack')) {
  output.append(detok.render(frame.ids, { partial: !frame.done }));
}
```

The hash is content-addressed: same hash means byte-identical map,
regardless of where it's hosted. Pin the hash in your deployment, never
the URL alone.

## Hosted via jsDelivr

GitHub raw content is served through [jsDelivr](https://www.jsdelivr.com/)
as a global CDN with proper caching, no egress fees, and excellent uptime.
Two URL forms work:

```
https://cdn.jsdelivr.net/gh/wdunn001/codec-maps/maps/<family>.json
https://cdn.jsdelivr.net/gh/wdunn001/codec-maps@<commit-or-tag>/maps/<family>.json
```

Pin to a specific commit if you want absolute reproducibility (the hash
verification covers correctness either way).

## Why this exists — wire impact

Codec replaces the JSON-SSE wire format with binary token IDs. Real
measurements (`Codec/packages/bench`) show what that buys you:

| Configuration                              | B/token | vs JSON-SSE |
|--------------------------------------------|---------|-------------|
| JSON-SSE, live Ollama qwen2.5 (320 tokens) |   186.4 |        1.0× |
| JSON-SSE, synthetic 1 token/chunk          |   154.0 |        1.0× |
| Codec msgpack, identity                    |    16.0 |        9.6× |
| Codec protobuf, identity                   |    10.9 |   **14.2×** |
| Codec msgpack + `Content-Encoding: zstd`   |     3.4 |   **45.0×** |
| Codec protobuf + `Content-Encoding: zstd`  |     3.6 |       43.1× |

Agent-to-agent handoffs see a **3.6× end-to-end speedup** on a 1024-token
round-trip because both the wire shrinks AND the detokenize → JSON →
re-tokenize text round-trip is eliminated.

A map is the data needed to make this work without giving up
human-readable text — clients decode IDs to text only when a human is
actually going to read the output. Agent-to-agent traffic skips
detokenization entirely.

## Schema

Maps conform to [`tokenizer-map.schema.json`](https://github.com/wdunn001/Codec/blob/main/spec/tokenizer-map.schema.json)
in the Codec spec repo. Abbreviated v2 shape:

```jsonc
{
  "id": "qwen/qwen2",                       // stable, lowercased "org/model-family"
  "version": "2",                            // schema version
  "vocab_size": 151665,
  "vocab": { "Hello": 9707, "Ġworld": 1879 },// raw HF tokenizer.json keys
  "encoder": "byte_level",                   // or "metaspace" or omitted
  "merges": ["Ġ Ġ", "ĠĠ ĠĠ", "i n", ...],   // BPE merge rules in priority order
  "pre_tokenizer_pattern": "(?i:'s|...)|...", // byte_level only
  "byte_fallback_start": 3,                  // SentencePiece-style maps only
  "byte_fallback_end": 258,
  "special_tokens": { "<|endoftext|>": 151643, ... },
  "published_at": "2026-05-06T00:00:00Z"
}
```

Three encoder families cover ~95% of open models:

- **`byte_level`** — GPT-2 byte→unicode mapping. Llama-3, Qwen, Phi-3/4, DeepSeek-V3, Mistral-Nemo, Falcon, SmolLM2.
- **`metaspace`** — `▁`-prefix SentencePiece. Llama-2, Mistral-v3, Mixtral, Gemma, Codestral.
- **identity** — vocab is already decoded text. For canonical-IR / synthetic test maps.

## Generate a new map

This repo is community-maintainable — anyone can contribute a map for
their model. The `@codecai/maps-cli` tool generates schema-compliant maps
from any HuggingFace `tokenizer.json`:

```bash
# Install
npm install -g @codecai/maps-cli

# Build a map for a model
codecai-maps build Qwen/Qwen2.5-7B-Instruct --id=qwen/qwen2

# For gated models (Llama, Gemma)
codecai-maps build meta-llama/Llama-3.1-8B --id=meta-llama/llama-3 --token=hf_xxx

# Sanity-check a round-trip
codecai-maps preview qwen_qwen2.json --text="Explain entropy."
# → "exact match: YES"

# Get the canonical sha256 for pinning
codecai-maps hash qwen_qwen2.json
# → sha256:c73972f7a580936d724ffd8df9df2ce546d255c543e9d09b6d75e5bf69b1a64d
```

## Repository layout

```
codec-maps/
├── maps/                   # one JSON file per model family
│   ├── meta-llama/
│   │   ├── llama-3.json
│   │   └── llama-2.json
│   ├── qwen/
│   │   └── qwen2.json
│   └── ...
├── scripts/
│   ├── convert.ts          # HF tokenizer.json → CodecTokenizerMap
│   ├── fetch.ts            # bulk fetch from public HF repos
│   └── fetch-mirrors.ts    # fetch via mirror repos for gated models
├── index.json              # machine-readable {id, url, hash, vocab_size, ...}
├── aliases.json            # {hf-model-id → canonical map id}
└── README.md
```

The `scripts/` directory is how the maintained set was generated. If
you're adding a single map for your own model you don't need them — use
`@codecai/maps-cli` directly.

## Contributing

Add or update a map:

1. Generate it: `codecai-maps build <hf-model> --id=<family-id>`
2. Verify the round-trip: `codecai-maps preview <map.json>`
3. Commit to `maps/<org>/<family>.json` and update `index.json` + `aliases.json`
4. Open a PR. CI will validate the schema + check the hash matches the file.

For widely-used models (top-50 on HuggingFace) we'll generally accept the
PR. For niche models, hosting the map alongside your own model weights is
also fine — `@codecai/web`'s `loadMap()` doesn't care where the file
lives, only that the hash matches.

## Map immutability

Maps are immutable once published. A new model version (vocab change,
new special tokens, etc.) MUST publish a new map at a new `id` or
filename — never overwrite an existing hash. Clients pin hashes; an
overwrite would silently break every consumer.

## Related

- **Codec spec**: https://github.com/wdunn001/Codec
- **`@codecai/web`** (browser/Node client): https://www.npmjs.com/package/@codecai/web
- **`@codecai/maps-cli`** (this generator): https://www.npmjs.com/package/@codecai/maps-cli
- **vLLM PR**: https://github.com/vllm-project/vllm/pull/41765
- **SGLang PR**: https://github.com/sgl-project/sglang/pull/24483

### A note on safety policies (v0.4)

This registry hosts **tokenizer maps** at
`/.well-known/codec/maps/<id>.json`. The v0.4 safety-policy
negotiation adds a *separate* descriptor type at
`/.well-known/codec/policies/<id>.json` — see
[`spec/versions/v0.4.md`](https://github.com/wdunn001/Codec/blob/main/spec/versions/v0.4.md#safety-policy-negotiation)
and [`spec/safety-policy.schema.json`](https://github.com/wdunn001/Codec/blob/main/spec/safety-policy.schema.json).
Safety policies are operator-published, not community-curated, so
there's no `codec-policies` repo analogous to this one — each
operator hosts their own under their own origin. Tokenizer maps in
this repo are unaffected by v0.4 (the schema is unchanged; v0.4 is
wire-additive over v0.3).

## License

Maps in this repo are derived from each model's tokenizer file and inherit
the license of the source model. The conversion scripts and this
documentation are MIT — see [`LICENSE`](./LICENSE).

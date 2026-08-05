# thinking-traces

What thinking traces actually buy you, per task type. Thinking on versus thinking off on the
same local model, over the same items, at temperature 0, with intervals and token cost.

Catalog task: `RSCH-011`, from the public build catalog
[722 things to build](https://github.com/JesseRWeigel/722-things-to-build).

**Results page:** https://jesserweigel.github.io/thinking-traces/

## What this is

Reasoning models are usually sold with a single number: thinking on scores higher. That number is
an average over a task mix, and it hides two things a caller actually needs to know. Which kinds of
task get nothing from the traces, and what the traces cost when they do help.

This measures both, per task type, on the same model, with the identical prompt in both
conditions. The only difference between conditions is one boolean.

### Task types

Five, picked to be different in kind rather than five flavours of the same thing.

| Type | What it is | Why it is here |
|---|---|---|
| `arith` | Three and four step arithmetic word problems | Serial computation. Traces should help. |
| `deduct` | Five runner ordering puzzles with a unique solution | Search over constraints. Traces should help. |
| `recall` | One fact, one short answer | Retrieval. There is nothing to compute, so traces should do nothing. |
| `instruct` | Output format constraints checked by code | Word counts, JSON key order, forbidden letters. Compliance rather than reasoning. |
| `overthink` | Puzzles that look like famous ones and are not | The literal reading is trivially correct and the memorised look-alike answer is wrong. Built as a place where reasoning can talk itself out of the right answer. |

40 items per type, 200 items per condition, 400 responses per model.

The `overthink` set is the one built so a negative result can appear. "All but 7 of 17 sheep run
away" has the answer 7, and a model that reaches for arithmetic gets 10. "Which weighs more, 3 kg
of feathers or 1 kg of lead" has the answer feathers, and a model that recognises the famous
version answers "the same". A run where thinking helped everywhere would be a suspicious result,
so the item set has to contain places where it can lose.

### Ground truth

Every grader is closed form code. No model judges another model, so the same cached response always
produces the same grade.

The deduction answers are not taken on trust from the generator. The test suite parses the
constraints back out of the English prompt and brute forces all 120 orderings, a code path that
shares nothing with the generator, then asserts each puzzle has exactly one solution and that the
labelled answer is it.

### Three outcomes, not two

`usable` and `correct` are tracked separately. A response that produced nothing to grade, whether
because the model returned an empty body or because it burned the whole token budget on reasoning
and never emitted an answer, is counted as unusable rather than as wrong. A 0 percent accuracy cell
therefore cannot be an empty output in disguise, and the usable count sits next to every accuracy.

## The measurement

- **Models.** Local, through Ollama. `qwen3:8b` and `qwen3.5:9b`. Both fit alongside the other work
  already resident on the shared card.
- **Temperature 0**, `top_k` 1, seed 7, `num_predict` 4096, single turn, no system prompt.
- **The `think` flag is passed as a top level field on `/api/chat`**, not as an option and not
  through the OpenAI compatible endpoint, which drops it. Verified by effect rather than by
  parameter: the reasoning payload has to be present on the think-on responses and absent on the
  think-off ones. `analyze` refuses to publish a model that fails this and `verify.sh` prints the
  evidence. A silently ignored flag would make both conditions the same run and produce a clean
  null result that means nothing.
- **Intervals.** 95 percent Wilson for a single accuracy. For the difference between conditions the
  two conditions see identical items, so the reported interval is on the mean of the per-item
  differences. When that interval contains zero the row reads `indistinguishable`, which is a claim
  about this sample size and not a claim that the effect is zero. At 40 items per cell an effect
  below roughly 10 to 15 points is not detectable, and the page says so.
- **Cost, two ways.** Generated tokens, which are unaffected by what else is running. And decode
  seconds from the server's own `eval_duration`, which excludes model load time. Raw wall clock is
  recorded and reported too, with the caveat that it partly measures the neighbours on a shared
  card.

### Shared GPU

One card, several agents. While this ran, `gpt-oss:20b` sat resident at 14.9 GB serving a Minecraft
bot swarm that is not mine to stop, another builder's `qwen2.5-coder:7b` came and went, and GPU
utilisation from that other work sat between 55 and 80 percent. That contention inflates wall clock
and leaves generated tokens untouched, which is why tokens are the headline cost column and wall
clock carries a caveat.

## Running it

The measurement needs a GPU and a running Ollama. Everything else runs from the cached responses
committed under `data/raw/`.

```bash
# reproduce the measurement (needs Ollama, one to two hours per model on a contended card)
PYTHONPATH=src python3 -m thinktrace.runner --model qwen3:8b   --think both
PYTHONPATH=src python3 -m thinktrace.runner --model qwen3.5:9b --think both

# everything below needs no GPU
PYTHONPATH=src python3 -m thinktrace.items     # regenerate data/items.json
PYTHONPATH=src python3 -m thinktrace.analyze   # regenerate results/summary.json
PYTHONPATH=src python3 -m thinktrace.site      # regenerate docs/index.html

bash scripts/verify.sh
```

## Verification

`bash scripts/verify.sh` is the verify command. It runs offline and needs no GPU.

1. A unit suite. Every assertion is paired with a negative control, a case that must fail, so an
   assertion cannot pass by accepting everything.
2. `data/items.json` is regenerated from its seed and compared byte for byte.
3. `results/summary.json` is regenerated from the cached responses and compared byte for byte.
4. The think flag evidence is printed and enforced.
5. **An independent recomputation in Node** that imports nothing from the Python package. It
   re-derives the answer extraction, the normalisation, all four graders, the Wilson interval, the
   paired interval and every cost aggregate from the raw JSONL, then compares against the published
   summary. All float comparisons are relative to the natural scale of the quantity, never a
   hardcoded number of accuracy points.
6. The page is rebuilt and compared, then checked against the summary for the actual numbers, so a
   template that silently dropped a column fails.
7. A hygiene scan that reads every file as raw bytes rather than as text, because one NUL byte
   makes git and grep treat a file as binary and skip it silently.
8. A README check: this file has to carry the success line and the real test count.
9. **A sabotage suite.** Eight attacks, each on a copy of the tree, each proving three things in
   order: the patch applied, it changed the measured output, and a check caught it. An attack that
   fails either of the first two is reported as a no-op and fails the suite rather than being
   credited to the detector.

| Sabotage | Broken | Caught by |
|---|---|---|
| `numeric-grader-always-true` | grading | independent Node recomputation |
| `empty-output-counted-usable` | usable versus wrong | unit suite |
| `wilson-zero-width` | single accuracy interval | unit suite |
| `paired-diff-sign-flip` | difference interval | independent Node recomputation |
| `token-cost-undercount` | token aggregation | independent Node recomputation |
| `flag-evidence-wrong-field` | think flag evidence | independent Node recomputation |
| `ground-truth-off-by-one` | item ground truth | unit suite |
| `page-numbers-blanked` | publication | page content check |

## Findings

See `results/summary.json` and the results page. The headline is the `verdict` column: which cells
are `helps`, which are `indistinguishable` at this sample size, and what each one cost.

## Status

TODO: pasted verify output goes here once the measurement run completes.

## Unfinished

TODO

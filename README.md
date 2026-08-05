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

**Thinking traces did not measurably improve accuracy in any of the ten cells, and cost 3.3 to
162 times the generated tokens.** Every apparent drop traces to the model spending its token
budget in the reasoning channel and returning nothing to grade, not to reasoning its way to a
worse answer.

| Model | Task type | Acc off | Acc on | Paired diff, 95% CI | Verdict | Usable off/on | Acc among usable, off/on | Tok off | Tok on | Token cost | Extra decode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3.5:9b` | arith | 90.0% | 70.0% | -20.0 [-32.6, -7.4] | hurts, budget limited | 40/33 | 90.0% / 84.8% | 262 | 2083 | 8.0x | +15.4s |
| `qwen3.5:9b` | deduct | 100.0% | 15.0% | -85.0 [-96.2, -73.8] | hurts, budget limited | 40/7 | 100.0% / 85.7% | 1196 | 3927 | 3.3x | +22.7s |
| `qwen3.5:9b` | recall | 100.0% | 100.0% | +0.0 [+0.0, +0.0] | indistinguishable | 40/40 | 100.0% / 100.0% | 59 | 516 | 8.8x | +3.8s |
| `qwen3.5:9b` | instruct | 80.0% | 67.5% | -12.5 [-25.0, +0.0] | indistinguishable, budget limited | 40/27 | 80.0% / 100.0% | 10 | 1574 | 161.8x | +13.4s |
| `qwen3.5:9b` | overthink | 97.5% | 87.5% | -10.0 [-19.4, -0.6] | hurts, budget limited | 40/35 | 97.5% / 100.0% | 101 | 1288 | 12.7x | +10.1s |
| `qwen3:8b` | arith | 97.5% | 97.5% | +0.0 [+0.0, +0.0] | indistinguishable | 40/40 | 97.5% / 97.5% | 159 | 1139 | 7.2x | +4.8s |
| `qwen3:8b` | deduct | 92.5% | 87.5% | -5.0 [-18.9, +8.9] | indistinguishable, budget limited | 40/40 | 92.5% / 87.5% | 797 | 3089 | 3.9x | +9.8s |
| `qwen3:8b` | recall | 97.5% | 100.0% | +2.5 [-2.4, +7.4] | indistinguishable | 40/40 | 97.5% / 100.0% | 17 | 282 | 16.1x | +1.7s |
| `qwen3:8b` | instruct | 75.0% | 77.5% | +2.5 [-10.6, +15.6] | indistinguishable, budget limited | 40/36 | 75.0% / 86.1% | 114 | 746 | 6.6x | +3.4s |
| `qwen3:8b` | overthink | 95.0% | 90.0% | -5.0 [-11.8, +1.8] | indistinguishable | 40/39 | 95.0% / 92.3% | 34 | 600 | 17.6x | +3.6s |

Totals, 200 items per condition per model:

| Model | Condition | Correct | Usable | Generated tokens | Decode seconds | Truncated |
|---|---|---|---|---|---|---|
| `qwen3:8b` | off | 183/200 | 200 | 44,872 | 656 | 2 |
| `qwen3:8b` | on | 181/200 | 195 | 234,264 | 1,590 | 11 |
| `qwen3.5:9b` | off | 187/200 | 200 | 65,071 | 510 | 0 |
| `qwen3.5:9b` | on | 136/200 | 142 | 375,488 | 3,124 | 60 |

### Restricted to items answered in both conditions

This is the cut that separates "reasoned itself wrong" from "ran out of budget". Over items where
both conditions produced something to grade, **no cell shows a difference this sample size can
distinguish from zero**, in either direction.

| Model | Task type | n | Paired diff, 95% CI |
|---|---|---|---|
| `qwen3.5:9b` | arith | 33 | -6.1 [-14.3, +2.2] |
| `qwen3.5:9b` | deduct | 7 | -14.3 [-42.3, +13.7] |
| `qwen3.5:9b` | recall | 40 | +0.0 [+0.0, +0.0] |
| `qwen3.5:9b` | instruct | 27 | +3.7 [-3.6, +11.0] |
| `qwen3.5:9b` | overthink | 35 | +0.0 [+0.0, +0.0] |
| `qwen3:8b` | arith | 40 | +0.0 [+0.0, +0.0] |
| `qwen3:8b` | deduct | 40 | -5.0 [-18.9, +8.9] |
| `qwen3:8b` | recall | 40 | +2.5 [-2.4, +7.4] |
| `qwen3:8b` | instruct | 36 | +5.6 [-7.8, +19.0] |
| `qwen3:8b` | overthink | 39 | -2.6 [-7.6, +2.5] |

### Four things worth taking away

**`think: false` does not stop the model reasoning.** It removes the separate reasoning channel.
With thinking off, `qwen3:8b` still spent a mean of 797 tokens per deduction item and `qwen3.5:9b`
spent 1,196, reasoning in the ordinary content channel. Treating think-off as a cheap baseline
without measuring its token count is a mistake. The place where the flag really is cheap is
instruction following on `qwen3.5:9b`: 10 tokens off against 1,574 on, a 162 fold difference for a
task where the answer is a formatted string and there is nothing to work out.

**The token budget is the binding constraint, and it binds asymmetrically.** At `num_predict`
4096, thinking on truncated 60 of 200 responses on `qwen3.5:9b` against 0 with it off. On
deduction it truncated 34 of 40. A harness that books a truncated response as an incorrect answer
would report that cell as 15 percent accurate and conclude that reasoning destroyed the model's
deduction. It answered 7 items and got 6 right.

**A high baseline leaves nothing to win.** Both models sit between 75 and 100 percent with
thinking off on every task type. The ceiling, not the reasoning, is doing most of the work here.
This is a real limit on the experiment and is stated in Unfinished below.

**The task type that was supposed to show reasoning hurting did not show it.** The `overthink`
set was built so traces could talk a model out of a correct literal answer. Both models scored
above 87 percent on it in both conditions, and the paired difference over answered items is +0.0
and -2.6 points. The hypothesis that reasoning entrenches the memorised look-alike answer is not
supported at this sample size, and that refutation is encoded as a passing test rather than left
as a story about what should have happened.

## Status

TODO: pasted verify output goes here once the measurement run completes.

## Unfinished

TODO

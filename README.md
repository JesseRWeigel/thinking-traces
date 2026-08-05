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
bot swarm that is not mine to stop. Two other builders' models came and went alongside it,
`qwen2.5-coder:7b` and `mistral:7b`, and GPU utilisation from that other work sat between 55 and 95
percent. My model was evicted and reloaded several times mid run, which is visible in the recorded
`load_duration_ns` per response.

That contention inflates wall clock and leaves generated tokens untouched, which is why tokens are
the headline cost column. Decode seconds come from the server's own `eval_duration` and exclude
model load, so they survive eviction but not compute contention. The measured throughput ranged
from 25 to 220 tokens per second on the same model depending on what else was running, so the
absolute seconds in the tables are specific to a busy card. The token counts are not.

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
9. The self-disagreement floor is enforced, and fails while unmeasured rather than passing quietly.
10. **A sabotage suite.** Twelve attacks, each on a copy of the tree, each proving three things in
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
| `accuracy-usable-over-all-attempts` | no-answer versus wrong-answer | independent Node recomputation |
| `paired-usable-includes-unanswered` | the answered-only comparison | independent Node recomputation |
| `budget-limited-suppressed` | the truncation marker | independent Node recomputation |
| `noise-floor-uses-point-estimate` | the effect-size floor | independent Node recomputation |
| `page-numbers-blanked` | publication | page content check |

The detectors are run against the sabotaged artefact, not the committed one. That distinction is
load bearing: an earlier version of this harness pointed the Node checker at the pristine
`summary.json`, so it recomputed correct numbers, compared them to correct numbers, and passed while
four of the attacks sat undetected.

### Checked in a real browser, separately

Not in `verify.sh`, because it needs a browser and the network. Run against the deployed page at a
390 by 844 viewport, asserting page identity inside the evaluation because the browser is shared
with other agents on this workstation:

```
clientWidth 375, documentElement.scrollWidth 375, sideways body scroll: no
elements whose right edge escapes the page: 0
body overflow-x: visible   (not hidden, which would mask real overflow)
three wide tables scroll inside their own containers: 1797/341, 1789/341, 758/341
742 digits of results, 11 verdict labels, 7 budget-limited markers rendered
```

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

One row there is weak rather than reassuring, and it is worth saying so plainly. `qwen3.5:9b` on
deduction has n=7, because 33 of its 40 think-on responses were truncated. Its interval, -14.3
[-42.3, +13.7], is wide enough to contain almost anything, so that cell is uninformative about
reasoning quality rather than evidence for the null. The nine cells with n between 27 and 40 carry
the claim. The deduction cell carries only the truncation finding.

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

**`bash scripts/verify.sh` exits 1.** Six checks fail, and every one of them traces to a single
missing piece: the replicate run that measures the self-disagreement floor. It is waiting on
exclusive GPU time behind another builder's throughput measurement, which contention makes wrong
rather than merely slow. The failures are the gate working as designed, not a defect discovered
after the fact. Pasted from a real run:

```
[1] prerequisites
    ok: python3 Python 3.12.3
    ok: node v24.13.0
[2] cached responses are present
    ok: 800 cached raw responses across 4 condition files
[3] every cached response still belongs to its prompt
    checked 800 responses against the current item set
    models: qwen3.5:9b, qwen3:8b, both conditions, 200 items each
[4] unit suite
    ok: 110 tests passed
[5] item set is reproducible from its generator
    ok: data/items.json matches a fresh generation (200 items)
[6] summary is reproducible from the cached responses
    ok: results/summary.json matches a fresh analysis of data/raw
[7] the think flag demonstrably took effect
    qwen3:8b: reasoning payload present on 100% of think-on responses, 0% of think-off responses
    qwen3.5:9b: reasoning payload present on 100% of think-on responses, 0% of think-off responses
[8] self-disagreement floor has been measured
    COULD NOT CHECK: no replicate run present under data/replicate
    FAIL: noise floor not measured, so no effect size can be interpreted
[9] independent recomputation (node, shares no code with the analysis)
    FAIL: independent check exited 1
        MISMATCH no replicate run under data/replicate, so no effect size can be interpreted
      independent check: 800 cached responses, 200 items
      independent check: 459 comparisons, 1 mismatches
[10] published page carries the measured numbers
    ok: docs/index.html matches a fresh build
    page check: 115 assertions, 0 problems
[11] no private paths, no credential-shaped strings, no binary blind spots
    scanned 28 files as raw bytes
[12] README reflects this run
    FAIL: README Status has no line that is exactly the success line
    FAIL: README quotes a different test count than the 110 that just ran
    FAIL: README still contains TODO
[13] sabotage suite
      --- SABOTAGE noise-floor-uses-point-estimate
        (a) patch applied: src/thinktrace/analyze.py differs from the original
        (b) FAILED: the artefact is unchanged, so this attack proves nothing about the checks
      11 proven attacks caught, 1 failed
    FAIL: sabotage suite exited 1

6 check(s) failed
VERIFY OK: thinking-traces -- NOT REACHED
```

Reading the failures:

- **[8] and [9]** are the floor gate. `analyze` writes
  `noise_floor: {"measured": false, "reason": "no replicate run present"}` and every cell carries
  `clears_noise_floor: null`. Both the verify step and the independent Node checker refuse to pass
  on that, because a skipped check reports the same success as one that ran.
- **[12]** is this section and the two `TODO` markers in Unfinished. Both go once the run completes
  and the numbers below are regenerated. The success-line check is a whole-line match on purpose: a
  substring match would have been satisfied by the `-- NOT REACHED` line pasted right above, so a
  failing run could have documented itself as a passing one.
- **[13]** is the honest outcome for the twelfth attack rather than a weakness in it. That sabotage
  makes the floor read its point estimate instead of the interval's upper bound. With no replicate
  data the function returns early, so the patch changes the file and changes nothing downstream. The
  harness reported it as a no-op and failed the suite rather than crediting the detector, which is
  exactly what it is built to do. Eleven of twelve attacks are proven caught; the twelfth is
  unproven and is reported as unproven.

The other twelve steps pass from a fresh clone taken outside the source tree, which is where a
hardcoded sibling path or a stale committed artefact would show up:

```
git clone . /tmp/x && cd /tmp/x && bash scripts/verify.sh
```

### What passing will look like

Once the card frees, `--tag rep2 --per-type 8` on both models fills `data/replicate/`, and steps 8,
9 and 13 have their inputs. Nothing else changes. If the floor turns out to be large enough to
swallow the differences reported in Findings, the `clears_noise_floor` column will say so and the
prose above it will be rewritten to match rather than left standing.

## Unfinished

**The self-disagreement floor is not yet measured, and `verify.sh` fails because of it.** Temperature
0 is not determinism on this backend: another builder on this workstation demonstrated one unchanged
route returning two different answers over eight repeats of the same prompt. Without a replicate run
there is no floor to compare an effect against, so `analyze` records
`noise_floor: {"measured": false}`, every cell carries `clears_noise_floor: null`, and the verify
step for it exits nonzero rather than reporting a skip as a pass. The machinery is built and the run
is two commands:

```bash
PYTHONPATH=src python3 -m thinktrace.runner --model qwen3:8b   --think both --tag rep2 --per-type 8
PYTHONPATH=src python3 -m thinktrace.runner --model qwen3.5:9b --think both --tag rep2 --per-type 8
```

It is waiting on exclusive GPU time behind another measurement task that contention makes wrong
rather than merely slow.

**The baseline is high, which caps what any intervention could win.** Thinking off scores between 75
and 100 percent on every task type. A cell already at 97.5 percent cannot show a large gain, so
"thinking did not help here" is partly a statement about these items being too easy for these
models. Harder items in each category would be a better test of the same question, and the honest
reading of this result is narrower than "thinking traces do not help": it is that on tasks these
models already handle, the traces are pure cost.

**Only two models, both small and from the same family.** `qwen3:8b` and `qwen3.5:9b`. The catalog
task also named `qwen3.5:27b` and `qwen3.6:27b`, which are 17 GB each and did not fit in the
available headroom alongside the swarm's 14.9 GB. Nothing here should be read as applying to larger
models or to other families.

**`num_ctx` was left at the server default.** The model loaded at a 32768 context for prompts under
200 tokens, which cost roughly 10 GB of VRAM where about 6 GB would have done. Sizing it from the
longest prompt would make this cheaper to reproduce. It was deliberately not changed after the main
run began, because the replicate measurement has to use byte identical settings or it measures the
config change rather than backend nondeterminism.

**Two graded items are judgment calls rather than facts.** `recall-25` asks which planet has the most
moons, where the answer changed with observations in 2023 and a model trained earlier answers
Jupiter. That is a legitimate recall failure but it dates the item. `overthink-30` expects "meat" for
what a butcher weighs, and a model answering "cannot be determined" is arguably reading the question
more carefully than the puzzle intends. Both are graded identically in both conditions, so neither
affects the paired comparison, and both are named here rather than quietly left in.

**Grading rewards the requested answer format.** Every prompt asks for a final `Answer:` line, and
the extractor falls back to the last non-empty line when a model ignores that. A model that buries a
correct answer mid-paragraph can still be marked wrong. This is applied identically to both
conditions.

**The link check is not part of `verify.sh`.** `scripts/check_links.sh` fetches the deployed page and
resolves every outbound link, and it needs the network, which would make verify nondeterministic. It
is run separately and its output is pasted below.

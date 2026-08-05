#!/usr/bin/env node
// Independent recomputation of results/summary.json from the cached raw responses.
//
// This file imports nothing from the Python package. It re-derives the answer
// extraction, the normalisation, all four graders, the Wilson interval, the
// paired difference interval and every cost aggregate from data/items.json and
// data/raw/*.jsonl, then compares its own numbers against the published summary.
//
// The point is the one AGENTS.md makes about validators: a checker that calls the
// aggregator it is checking passes whenever the aggregator is wrong in a
// self-consistent way. Two implementations in two languages, written separately
// from the same written rules, do not fail the same way.
//
// All float comparisons are RELATIVE. Each quantity declares the natural scale it
// lives on (1.0 for a proportion, the larger of the two magnitudes for a count or
// a duration), so no tolerance is a hardcoded number of accuracy points.

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const REL = 1e-9;

let failures = 0;
let checks = 0;

function fail(msg) {
  failures += 1;
  console.error(`  MISMATCH ${msg}`);
}

function eq(name, got, want) {
  checks += 1;
  if (got !== want) fail(`${name}: independent ${got} vs summary ${want}`);
}

function closeRel(name, got, want, scale) {
  checks += 1;
  const denom = Math.max(Math.abs(got), Math.abs(want), scale);
  if (denom === 0) {
    if (got !== want) fail(`${name}: ${got} vs ${want}`);
    return;
  }
  const rel = Math.abs(got - want) / denom;
  if (!(rel <= REL)) {
    fail(`${name}: independent ${got} vs summary ${want} (relative gap ${rel.toExponential(3)})`);
  }
}

// ---------------------------------------------------------------- extraction

function stripFences(text) {
  const m = /^\s*```[a-zA-Z0-9]*\s*\n([\s\S]*?)\n\s*```\s*$/.exec(text || '');
  return m ? m[1] : (text || '');
}

function extractAnswer(content) {
  const lines = stripFences(content || '').split('\n').map((l) => l.replace(/\s+$/, ''));
  // Searched anywhere in the line, not anchored to its start: models often put
  // the whole response and its answer label on a single line.
  const label = /\*{0,2}\s*\banswer\s*\*{0,2}\s*[:\-]\s*\*{0,2}\s*(.*?)[\s*]*$/i;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const m = label.exec(lines[i]);
    if (m && m[1].trim()) return m[1].trim();
  }
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (lines[i].trim()) return lines[i].trim();
  }
  return '';
}

function normalize(s) {
  // Escapes rather than literal combining marks: the raw bytes are invisible in
  // review and one stray edit would silently change what this strips.
  let out = (s || '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  out = out.toLowerCase().replace(/\u2019/g, "'");
  out = out.replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
  if (out.startsWith('the ')) out = out.slice(4);
  return out;
}

const WORD_NUMBERS = {
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6,
  seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12,
};

function parseNumber(s) {
  const norm = normalize(s);
  if (Object.prototype.hasOwnProperty.call(WORD_NUMBERS, norm)) return WORD_NUMBERS[norm];
  const m = /-?\d[\d,]*(?:\.\d+)?/.exec(s || '');
  if (!m) {
    const first = norm.split(' ')[0];
    if (Object.prototype.hasOwnProperty.call(WORD_NUMBERS, first)) return WORD_NUMBERS[first];
    return null;
  }
  const v = Number(m[0].replace(/,/g, ''));
  return Number.isNaN(v) ? null : v;
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function containsPhrase(haystack, needle) {
  if (!needle) return false;
  return new RegExp(`(?<![a-z0-9])${escapeRe(needle)}(?![a-z0-9])`).test(haystack);
}

// ------------------------------------------------------------------- graders

function gradeNumeric(item, pred) {
  const norm = normalize(pred);
  for (const cand of item.accept || []) {
    if (norm === normalize(cand)) return true;
  }
  const got = parseNumber(pred);
  const want = parseNumber(item.answer);
  if (got === null || want === null) return false;
  return Math.abs(got - want) < 1e-9;
}

function gradeText(item, pred) {
  const got = normalize(pred);
  if (!got) return false;
  const candidates = [item.answer].concat(item.accept || []);
  for (const cand of candidates) {
    const want = normalize(cand);
    if (!want) continue;
    if (got === want) return true;
    if (got.split(' ').length <= 12 && containsPhrase(got, want)) return true;
  }
  return false;
}

function gradeChoice(item, pred) {
  const got = normalize(pred);
  const hits = new Set();
  for (const [key, syns] of Object.entries(item.options)) {
    for (const syn of syns) {
      if (containsPhrase(got, normalize(syn))) { hits.add(key); break; }
    }
  }
  if (hits.size !== 1) return false;
  return [...hits][0] === item.answer;
}

function tokens(s) {
  return (s || '').trim().split(/\s+/).filter((t) => /[A-Za-z0-9]/.test(t));
}

function orderedKeys(jsonText) {
  // JSON.parse loses nothing about key order for these inputs, but relying on
  // that is fragile, so the keys are read off the source text directly.
  const keys = [];
  const re = /"((?:[^"\\]|\\.)*)"\s*:/g;
  let m;
  while ((m = re.exec(jsonText)) !== null) keys.push(m[1]);
  return keys;
}

function gradeConstraint(item, pred) {
  const p = item.params;
  const text = stripFences(pred || '').trim();
  if (!text) return false;
  switch (item.check) {
    case 'word_count':
      return tokens(text).length === p.n;
    case 'repeat_word': {
      const want = new Array(p.n).fill(p.word).join(' ');
      return text.replace(/\.$/, '').trim().toLowerCase() === want.toLowerCase();
    }
    case 'upper_csv': {
      const parts = text.replace(/\.$/, '').split(',').map((x) => x.trim());
      return parts.length === p.n
        && parts.every((x) => /^[A-Za-z]+$/.test(x) && x === x.toUpperCase());
    }
    case 'json_keys': {
      let obj;
      try { obj = JSON.parse(text); } catch (e) { return false; }
      if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) return false;
      const keys = orderedKeys(text);
      if (keys.length !== p.keys.length) return false;
      for (let i = 0; i < keys.length; i += 1) if (keys[i] !== p.keys[i]) return false;
      return Object.values(obj).every((v) => v === 1);
    }
    case 'exact_chars':
      return text === p.char.repeat(p.n);
    case 'desc_range': {
      const want = [];
      for (let n = p.hi; n >= p.lo; n -= 1) want.push(String(n));
      return text.replace(/\.$/, '').replace(/ /g, '') === want.join(',');
    }
    case 'start_end': {
      const t = tokens(text);
      if (t.length < 2) return false;
      const first = t[0].replace(/[^A-Za-z0-9']/g, '');
      const last = t[t.length - 1].replace(/[^A-Za-z0-9']/g, '');
      return first.toLowerCase() === p.start.toLowerCase()
        && last.toLowerCase() === p.end.toLowerCase();
    }
    case 'forbidden_letter':
      return tokens(text).length >= p.min_words
        && !text.toLowerCase().includes(p.letter.toLowerCase());
    default:
      throw new Error(`unknown check ${item.check}`);
  }
}

function gradeOne(item, content) {
  const body = (content || '').trim();
  const pred = item.grader === 'constraint' ? stripFences(body).trim() : extractAnswer(body);
  if (!body || !pred) return { usable: false, correct: false };
  let correct;
  if (item.grader === 'numeric') correct = gradeNumeric(item, pred);
  else if (item.grader === 'text') correct = gradeText(item, pred);
  else if (item.grader === 'choice') correct = gradeChoice(item, pred);
  else if (item.grader === 'constraint') correct = gradeConstraint(item, pred);
  else throw new Error(`unknown grader ${item.grader}`);
  return { usable: true, correct };
}

// --------------------------------------------------------------------- stats

const Z = 1.959963984540054;

function wilson(k, n) {
  if (n <= 0) return [0, 0];
  const p = k / n;
  const denom = 1 + (Z * Z) / n;
  const centre = (p + (Z * Z) / (2 * n)) / denom;
  const half = (Z / denom) * Math.sqrt((p * (1 - p)) / n + (Z * Z) / (4 * n * n));
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

function meanOf(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function sdOf(xs) {
  if (xs.length < 2) return 0;
  const m = meanOf(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) * (x - m), 0) / (xs.length - 1));
}

function pairedDiff(diffs) {
  const n = diffs.length;
  const mean = meanOf(diffs);
  const half = n ? (Z * sdOf(diffs)) / Math.sqrt(n) : 0;
  return { n, mean, lo: mean - half, hi: mean + half, half, significant: !(mean - half <= 0 && 0 <= mean + half) };
}

// ---------------------------------------------------------------------- main

function loadJsonl(dir) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.jsonl')).sort()) {
    const text = fs.readFileSync(path.join(dir, f), 'utf8');
    for (const line of text.split('\n')) {
      if (line.trim()) out.push(JSON.parse(line));
    }
  }
  return out;
}

function main() {
  const items = new Map(
    JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'items.json'), 'utf8'))
      .map((it) => [it.id, it]),
  );
  const summary = JSON.parse(fs.readFileSync(path.join(ROOT, 'results', 'summary.json'), 'utf8'));
  const raw = loadJsonl(path.join(ROOT, 'data', 'raw'));
  if (raw.length === 0) {
    console.error('no cached responses under data/raw');
    process.exit(2);
  }
  const replicates = loadJsonl(path.join(ROOT, 'data', 'replicate'));

  const graded = raw.map((r) => {
    const item = items.get(r.item_id);
    if (!item) throw new Error(`raw response references unknown item ${r.item_id}`);
    const g = gradeOne(item, r.content);
    return {
      model: r.model,
      cond: r.think_requested ? 'on' : 'off',
      type: item.type,
      id: r.item_id,
      usable: g.usable,
      correct: g.correct,
      truncated: r.done_reason === 'length',
      genTokens: r.eval_count || 0,
      wallS: r.wall_s || 0,
      genS: (r.eval_duration_ns || 0) / 1e9,
      traceChars: (r.thinking || '').length,
    };
  });

  console.log(`independent check: ${raw.length} cached responses, ${items.size} items`);

  // The flag-effect evidence is recomputed here too. If the think flag had been
  // silently ignored, both conditions would carry identical trace evidence and
  // the whole experiment would be a null result about nothing.
  for (const [model, cell] of Object.entries(summary.flag_check)) {
    const on = graded.filter((g) => g.model === model && g.cond === 'on');
    const off = graded.filter((g) => g.model === model && g.cond === 'off');
    const onFrac = on.filter((g) => g.traceChars > 0).length / on.length;
    const offFrac = off.filter((g) => g.traceChars > 0).length / off.length;
    closeRel(`${model} flag on-trace-fraction`, onFrac, cell.on_trace_fraction, 1.0);
    closeRel(`${model} flag off-trace-fraction`, offFrac, cell.off_trace_fraction, 1.0);
    eq(`${model} flag_effective`, onFrac >= 0.9 && offFrac <= 0.02, cell.flag_effective);
    if (!(onFrac >= 0.9 && offFrac <= 0.02)) {
      fail(`${model}: the think flag did not take effect (on ${onFrac}, off ${offFrac})`);
    }
  }

  // The self-disagreement floor, recomputed from the replicate responses. This is
  // the number that decides whether an effect is real or is the backend
  // disagreeing with itself, so it gets its own derivation rather than a nod.
  const floor = summary.noise_floor || {};
  if (replicates.length === 0) {
    eq('noise_floor measured', false, Boolean(floor.measured));
    fail('no replicate run under data/replicate, so no effect size can be interpreted');
  } else {
    eq('noise_floor measured', true, Boolean(floor.measured));
    const firstRun = new Map(graded.map((g) => [`${g.model}|${g.cond}|${g.id}`, g]));
    const mine = new Map();
    for (const r of replicates) {
      const item = items.get(r.item_id);
      if (!item) throw new Error(`replicate references unknown item ${r.item_id}`);
      const cond = r.think_requested ? 'on' : 'off';
      const original = firstRun.get(`${r.model}|${cond}|${r.item_id}`);
      if (!original) continue;
      const g = gradeOne(item, r.content);
      const key = `${r.model}|${cond}`;
      const cell = mine.get(key) || { n: 0, flips: 0 };
      cell.n += 1;
      if (Boolean(g.correct) !== Boolean(original.correct)) cell.flips += 1;
      mine.set(key, cell);
    }
    let worst = 0;
    for (const [key, cell] of [...mine.entries()].sort()) {
      const pub = (floor.by_condition || {})[key];
      if (!pub) { fail(`noise floor missing published cell ${key}`); continue; }
      eq(`floor ${key} n`, cell.n, pub.n);
      eq(`floor ${key} grade_flips`, cell.flips, pub.grade_flips);
      closeRel(`floor ${key} rate`, cell.flips / cell.n, pub.grade_flip_rate, 1.0);
      const [lo, hi] = wilson(cell.flips, cell.n);
      closeRel(`floor ${key} rate_lo`, lo, pub.rate_lo, 1.0);
      closeRel(`floor ${key} rate_hi`, hi, pub.rate_hi, 1.0);
      worst = Math.max(worst, hi);
    }
    closeRel('floor upper bound', worst, floor.floor_upper_bound, 1.0);
    for (const cell of summary.cells) {
      eq(`${cell.model}/${cell.type} clears_noise_floor`,
        Math.abs(cell.paired.mean) > worst, cell.clears_noise_floor);
    }
  }

  // A published summary must cover every model and task type. An omission that
  // the aggregator recorded as "incomplete" is still an omission.
  const models = [...new Set(graded.map((g) => g.model))].sort();
  const types = [...new Set(graded.map((g) => g.type))].sort();
  eq('cell count', summary.cells.length, models.length * types.length);
  eq('incomplete cells', (summary.incomplete_cells || []).length, 0);

  for (const cell of summary.cells) {
    const tag = `${cell.model}/${cell.type}`;
    const byCond = {};
    for (const cond of ['off', 'on']) {
      const sel = graded.filter(
        (g) => g.model === cell.model && g.type === cell.type && g.cond === cond,
      );
      byCond[cond] = sel;
      const n = sel.length;
      const k = sel.filter((g) => g.correct).length;
      const [lo, hi] = wilson(k, n);
      eq(`${tag} ${cond} n`, n, cell[cond].n);
      eq(`${tag} ${cond} correct`, k, cell[cond].correct);
      eq(`${tag} ${cond} usable`, sel.filter((g) => g.usable).length, cell[cond].usable);
      eq(`${tag} ${cond} truncated`, sel.filter((g) => g.truncated).length, cell[cond].truncated);
      closeRel(`${tag} ${cond} accuracy`, k / n, cell[cond].accuracy, 1.0);
      const usable = sel.filter((g) => g.usable).length;
      if (usable > 0) {
        closeRel(`${tag} ${cond} accuracy_usable`, k / usable, cell[cond].accuracy_usable, 1.0);
      } else {
        eq(`${tag} ${cond} accuracy_usable`, null, cell[cond].accuracy_usable);
      }
      closeRel(`${tag} ${cond} acc_lo`, lo, cell[cond].acc_lo, 1.0);
      closeRel(`${tag} ${cond} acc_hi`, hi, cell[cond].acc_hi, 1.0);
      const tokTotal = sel.reduce((a, g) => a + g.genTokens, 0);
      closeRel(`${tag} ${cond} gen_tokens_total`, tokTotal, cell[cond].gen_tokens_total,
        Math.abs(cell[cond].gen_tokens_total));
      closeRel(`${tag} ${cond} gen_tokens_mean`, tokTotal / n, cell[cond].gen_tokens_mean,
        Math.abs(cell[cond].gen_tokens_mean));
      const genTotal = sel.reduce((a, g) => a + g.genS, 0);
      closeRel(`${tag} ${cond} gen_s_mean`, genTotal / n, cell[cond].gen_s_mean,
        Math.abs(cell[cond].gen_s_mean));
      const wallTotal = sel.reduce((a, g) => a + g.wallS, 0);
      closeRel(`${tag} ${cond} wall_s_mean`, wallTotal / n, cell[cond].wall_s_mean,
        Math.abs(cell[cond].wall_s_mean));
    }

    const onById = new Map(byCond.on.map((g) => [g.id, g]));
    const offById = new Map(byCond.off.map((g) => [g.id, g]));
    const shared = [...onById.keys()].filter((id) => offById.has(id)).sort();
    const diffs = shared.map((id) => Number(onById.get(id).correct) - Number(offById.get(id).correct));
    const pd = pairedDiff(diffs);
    eq(`${tag} paired n`, pd.n, cell.paired.n);
    closeRel(`${tag} paired mean`, pd.mean, cell.paired.mean, 1.0);
    closeRel(`${tag} paired lo`, pd.lo, cell.paired.lo, 1.0);
    closeRel(`${tag} paired hi`, pd.hi, cell.paired.hi, 1.0);
    eq(`${tag} paired significant`, pd.significant, cell.paired.significant);
    eq(`${tag} flips on_only`, diffs.filter((d) => d > 0).length, cell.flips.on_only);
    eq(`${tag} flips off_only`, diffs.filter((d) => d < 0).length, cell.flips.off_only);
    eq(`${tag} flips agree`, diffs.filter((d) => d === 0).length, cell.flips.agree);

    const expectedVerdict = !pd.significant ? 'indistinguishable' : (pd.mean > 0 ? 'helps' : 'hurts');
    eq(`${tag} verdict`, expectedVerdict, cell.verdict);

    // Paired comparison over items answered in both conditions, recomputed here.
    const both = shared.filter((id) => onById.get(id).usable && offById.get(id).usable);
    const diffsBoth = both.map(
      (id) => Number(onById.get(id).correct) - Number(offById.get(id).correct),
    );
    const pdBoth = pairedDiff(diffsBoth);
    eq(`${tag} paired_usable_only n`, pdBoth.n, cell.paired_usable_only.n);
    closeRel(`${tag} paired_usable_only mean`, pdBoth.mean, cell.paired_usable_only.mean, 1.0);
    closeRel(`${tag} paired_usable_only lo`, pdBoth.lo, cell.paired_usable_only.lo, 1.0);
    closeRel(`${tag} paired_usable_only hi`, pdBoth.hi, cell.paired_usable_only.hi, 1.0);
    eq(`${tag} paired_usable_only significant`, pdBoth.significant,
      cell.paired_usable_only.significant);

    const onTrunc = byCond.on.filter((g) => g.truncated).length;
    const offTrunc = byCond.off.filter((g) => g.truncated).length;
    eq(`${tag} budget_limited`,
      onTrunc / byCond.on.length >= 0.10 && onTrunc > offTrunc, cell.budget_limited);

    const onTok = meanOf(byCond.on.map((g) => g.genTokens));
    const offTok = meanOf(byCond.off.map((g) => g.genTokens));
    closeRel(`${tag} token_ratio`, onTok / offTok, cell.token_ratio, Math.abs(cell.token_ratio));
    const extraS = meanOf(byCond.on.map((g) => g.genS)) - meanOf(byCond.off.map((g) => g.genS));
    closeRel(`${tag} extra_s_per_item`, extraS, cell.extra_s_per_item, Math.abs(cell.extra_s_per_item));
    const extraTok = onTok - offTok;
    closeRel(`${tag} extra_tokens_per_item`, extraTok, cell.extra_tokens_per_item,
      Math.abs(cell.extra_tokens_per_item));
    if (cell.acc_points_per_extra_second !== null) {
      closeRel(`${tag} acc_points_per_extra_second`, (pd.mean * 100) / extraS,
        cell.acc_points_per_extra_second, Math.abs(cell.acc_points_per_extra_second));
    }
    if (cell.acc_points_per_1k_extra_tokens !== null) {
      closeRel(`${tag} acc_points_per_1k_extra_tokens`, (pd.mean * 100 * 1000) / extraTok,
        cell.acc_points_per_1k_extra_tokens, Math.abs(cell.acc_points_per_1k_extra_tokens));
    }

    // Sanity facts that hold regardless of the aggregator: the flip counts must
    // partition the items, and correct answers cannot exceed usable ones.
    eq(`${tag} flips partition`, cell.flips.on_only + cell.flips.off_only + cell.flips.agree, pd.n);
    for (const cond of ['off', 'on']) {
      if (cell[cond].correct > cell[cond].usable) {
        fail(`${tag} ${cond}: ${cell[cond].correct} correct but only ${cell[cond].usable} usable`);
      }
    }
  }

  console.log(`independent check: ${checks} comparisons, ${failures} mismatches`);
  if (failures > 0) {
    console.error('independent check FAILED');
    process.exit(1);
  }
  console.log('independent check OK');
}

main();

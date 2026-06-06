#!/usr/bin/env node
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { readFile } from "node:fs/promises";
import { type ReplaySnapshot } from "./replay.js";

export type ReplayDiffSeverity = "regression" | "warning" | "improvement";
export type ReplayDiffArea = "report" | "probe" | "recall" | "belief" | "scenario" | "routine";

export interface ReplayDiffChange {
  severity: ReplayDiffSeverity;
  area: ReplayDiffArea;
  key: string;
  before: string | null;
  after: string | null;
  message: string;
}

export interface ReplayDiffOptions {
  maxRegressions?: number;
  maxWarnings?: number;
  highConfidenceBelief?: number;
}

export interface ReplayDiffReport {
  before: { dataset: string; pipeline: string; model: string | null; generatedAt: string };
  after: { dataset: string; pipeline: string; model: string | null; generatedAt: string };
  summary: {
    passDelta: number;
    failDelta: number;
    skipDelta: number;
    regressions: number;
    warnings: number;
    improvements: number;
    changes: number;
  };
  changes: ReplayDiffChange[];
  ok: boolean;
}

export function diffReplaySnapshots(
  before: ReplaySnapshot,
  after: ReplaySnapshot,
  opts: ReplayDiffOptions = {},
): ReplayDiffReport {
  const changes: ReplayDiffChange[] = [];
  const highConfidenceBelief = opts.highConfidenceBelief ?? 0.6;

  if (before.dataset !== after.dataset)
    changes.push(change("warning", "report", "dataset", before.dataset, after.dataset, "dataset changed"));
  if (before.pipeline !== after.pipeline)
    changes.push(change("warning", "report", "pipeline", before.pipeline, after.pipeline, "pipeline changed"));
  if (before.model !== after.model)
    changes.push(change("warning", "report", "model", before.model, after.model, "model changed"));
  if (after.report.fail > before.report.fail)
    changes.push(change("regression", "report", "fail-count", String(before.report.fail), String(after.report.fail), "eval failures increased"));
  if (after.report.pass < before.report.pass)
    changes.push(change("regression", "report", "pass-count", String(before.report.pass), String(after.report.pass), "eval passes decreased"));
  if (after.report.fail < before.report.fail)
    changes.push(change("improvement", "report", "fail-count", String(before.report.fail), String(after.report.fail), "eval failures decreased"));
  if (after.report.pass > before.report.pass)
    changes.push(change("improvement", "report", "pass-count", String(before.report.pass), String(after.report.pass), "eval passes increased"));

  diffProbes(before, after, changes);
  diffRecall(before, after, changes);
  diffBeliefs(before, after, changes, highConfidenceBelief);
  diffScenarios(before, after, changes);
  diffRoutines(before, after, changes);

  const regressions = changes.filter((c) => c.severity === "regression").length;
  const warnings = changes.filter((c) => c.severity === "warning").length;
  const improvements = changes.filter((c) => c.severity === "improvement").length;
  const maxRegressions = opts.maxRegressions ?? 0;
  const maxWarnings = opts.maxWarnings ?? Number.POSITIVE_INFINITY;
  return {
    before: meta(before),
    after: meta(after),
    summary: {
      passDelta: after.report.pass - before.report.pass,
      failDelta: after.report.fail - before.report.fail,
      skipDelta: after.report.skip - before.report.skip,
      regressions,
      warnings,
      improvements,
      changes: changes.length,
    },
    changes,
    ok: regressions <= maxRegressions && warnings <= maxWarnings,
  };
}

function diffProbes(before: ReplaySnapshot, after: ReplaySnapshot, changes: ReplayDiffChange[]): void {
  const a = new Map(after.report.results.map((r) => [probeKey(r), r]));
  const seen = new Set<string>();
  for (const b of before.report.results) {
    const key = probeKey(b);
    seen.add(key);
    const next = a.get(key);
    if (!next) {
      changes.push(change("regression", "probe", key, b.status, null, "probe disappeared"));
      continue;
    }
    if (b.status === next.status) continue;
    if (b.status === "pass" && next.status !== "pass")
      changes.push(change("regression", "probe", key, b.status, next.status, `probe regressed: ${next.got}`));
    else if (b.status !== "pass" && next.status === "pass")
      changes.push(change("improvement", "probe", key, b.status, next.status, `probe recovered: ${next.got}`));
    else
      changes.push(change("warning", "probe", key, b.status, next.status, `probe status changed: ${next.got}`));
  }
  for (const next of after.report.results) {
    const key = probeKey(next);
    if (seen.has(key)) continue;
    changes.push(change(next.status === "pass" ? "improvement" : "warning", "probe", key, null, next.status, `probe added: ${next.got}`));
  }
}

function diffRecall(before: ReplaySnapshot, after: ReplaySnapshot, changes: ReplayDiffChange[]): void {
  const a = new Map(after.recall.map((r) => [r.query, r]));
  const seen = new Set<string>();
  for (const b of before.recall) {
    seen.add(b.query);
    const next = a.get(b.query);
    if (!next) {
      changes.push(change("regression", "recall", b.query, b.names.join(" > "), null, "recall query disappeared"));
      continue;
    }
    const beforeTop = b.names[0] ?? "";
    const afterTop = next.names[0] ?? "";
    if (beforeTop && !afterTop)
      changes.push(change("regression", "recall", b.query, beforeTop, null, "recall lost all results"));
    else if (beforeTop !== afterTop)
      changes.push(change("warning", "recall", b.query, beforeTop || null, afterTop || null, "top recalled place changed"));
  }
  for (const next of after.recall) {
    if (!seen.has(next.query))
      changes.push(change("improvement", "recall", next.query, null, next.names.join(" > "), "recall query added"));
  }
}

function diffBeliefs(before: ReplaySnapshot, after: ReplaySnapshot, changes: ReplayDiffChange[], highConfidenceBelief: number): void {
  const a = new Map(after.beliefs.map((b) => [beliefKey(b), b]));
  const seen = new Set<string>();
  for (const b of before.beliefs) {
    const key = beliefKey(b);
    seen.add(key);
    const next = a.get(key);
    const beforeText = `${b.confidence.toFixed(3)} ${b.source}`;
    if (!next) {
      changes.push(change(b.confidence >= highConfidenceBelief ? "regression" : "warning", "belief", key, beforeText, null, "belief disappeared"));
      continue;
    }
    if (b.confidence - next.confidence >= 0.25)
      changes.push(change(b.confidence >= highConfidenceBelief ? "regression" : "warning", "belief", key, beforeText, `${next.confidence.toFixed(3)} ${next.source}`, "belief confidence dropped"));
  }
  for (const next of after.beliefs) {
    const key = beliefKey(next);
    if (!seen.has(key) && next.confidence >= highConfidenceBelief)
      changes.push(change("improvement", "belief", key, null, `${next.confidence.toFixed(3)} ${next.source}`, "high-confidence belief added"));
  }
}

function diffScenarios(before: ReplaySnapshot, after: ReplaySnapshot, changes: ReplayDiffChange[]): void {
  const a = new Map(after.scenarios.map((s) => [s.title, s]));
  const seen = new Set<string>();
  for (const b of before.scenarios) {
    seen.add(b.title);
    const next = a.get(b.title);
    if (!next) {
      changes.push(change("warning", "scenario", b.title, scenarioText(b), null, "scenario disappeared"));
      continue;
    }
    if (scenarioText(b) !== scenarioText(next))
      changes.push(change("warning", "scenario", b.title, scenarioText(b), scenarioText(next), "scenario contents changed"));
  }
  for (const next of after.scenarios) {
    if (!seen.has(next.title))
      changes.push(change("improvement", "scenario", next.title, null, scenarioText(next), "scenario added"));
  }
}

function diffRoutines(before: ReplaySnapshot, after: ReplaySnapshot, changes: ReplayDiffChange[]): void {
  const a = new Map((after.routines ?? []).map((r) => [r.title, r]));
  const seen = new Set<string>();
  for (const b of before.routines ?? []) {
    seen.add(b.title);
    const next = a.get(b.title);
    if (!next) {
      changes.push(change("warning", "routine", b.title, routineText(b), null, "routine disappeared"));
      continue;
    }
    if (routineText(b) !== routineText(next))
      changes.push(change("warning", "routine", b.title, routineText(b), routineText(next), "routine contents changed"));
    if (next.support < b.support)
      changes.push(change("warning", "routine", `${b.title}:support`, String(b.support), String(next.support), "routine support decreased"));
  }
  for (const next of after.routines ?? []) {
    if (!seen.has(next.title))
      changes.push(change("improvement", "routine", next.title, null, routineText(next), "routine added"));
  }
}

function probeKey(r: ReplaySnapshot["report"]["results"][number]): string {
  return `${r.category}: ${r.desc}`;
}

function beliefKey(b: ReplaySnapshot["beliefs"][number]): string {
  return `${b.predicate}:${b.object}`;
}

function scenarioText(s: ReplaySnapshot["scenarios"][number]): string {
  return `${s.placeNames.join(",")} | intents=${s.intents.join(",")} | concepts=${s.concepts.join(",")} | turns=${s.turnCount}`;
}

function routineText(r: ReplaySnapshot["routines"][number]): string {
  return `${r.summary} | support=${r.support} | confidence=${r.confidence} | preferred=${r.positivePlaceNames.join(",")} | avoided=${r.negativePlaceNames.join(",")} | intents=${r.intents.join(",")} | concepts=${r.concepts.join(",")}`;
}

function meta(s: ReplaySnapshot): ReplayDiffReport["before"] {
  return { dataset: s.dataset, pipeline: s.pipeline, model: s.model, generatedAt: s.generatedAt };
}

function change(
  severity: ReplayDiffSeverity,
  area: ReplayDiffArea,
  key: string,
  before: string | null,
  after: string | null,
  message: string,
): ReplayDiffChange {
  return { severity, area, key, before, after, message };
}

async function main() {
  const args = process.argv.slice(2);
  const json = args.includes("--json");
  const maxRegressions = numArg(args, "--max-regressions=", 0);
  const maxWarnings = numArg(args, "--max-warnings=", Number.POSITIVE_INFINITY);
  const highConfidenceBelief = numArg(args, "--belief-confidence=", 0.6);
  const paths = args.filter((a) => !a.startsWith("--"));
  if (paths.length !== 2) {
    console.error("Usage: npm run eval:replay:diff -- before.json after.json [--json] [--max-regressions=0] [--max-warnings=N] [--belief-confidence=0.6]");
    process.exitCode = 2;
    return;
  }
  const before = JSON.parse(await readFile(paths[0]!, "utf-8")) as ReplaySnapshot;
  const after = JSON.parse(await readFile(paths[1]!, "utf-8")) as ReplaySnapshot;
  const diff = diffReplaySnapshots(before, after, { maxRegressions, maxWarnings, highConfidenceBelief });
  if (json) process.stdout.write(JSON.stringify(diff, null, 2) + "\n");
  else printHuman(diff);
  if (!diff.ok) process.exitCode = 1;
}

function numArg(args: string[], prefix: string, fallback: number): number {
  const raw = args.find((a) => a.startsWith(prefix))?.slice(prefix.length);
  if (raw == null) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function printHuman(diff: ReplayDiffReport): void {
  const s = diff.summary;
  console.log(`Replay diff: ${diff.before.dataset} ${diff.before.pipeline} -> ${diff.after.pipeline}`);
  console.log(`pass ${signed(s.passDelta)} · fail ${signed(s.failDelta)} · skip ${signed(s.skipDelta)}`);
  console.log(`regressions=${s.regressions} warnings=${s.warnings} improvements=${s.improvements} ok=${diff.ok}`);
  for (const c of diff.changes)
    console.log(`${c.severity.toUpperCase()} [${c.area}] ${c.key}: ${c.message} (${c.before ?? "-"} -> ${c.after ?? "-"})`);
}

const signed = (n: number) => (n > 0 ? `+${n}` : String(n));

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) main();

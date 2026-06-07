#!/usr/bin/env node
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, isAbsolute, join } from "node:path";
import { buildOpenMap } from "../src/openmap.js";
import { loadConfig, type Config } from "../src/core/config.js";
import { runEval, type EvalDataset, type EvalReport } from "./harness.js";
import { buildReplaySnapshot, type ReplaySnapshot } from "./replay.js";
import { compare } from "./compare.js";
import { type LLMRunner } from "../src/nlp/llm.js";

const here = dirname(fileURLToPath(import.meta.url));

export type SuiteSource = "synthetic" | "anonymized" | "production";

export interface EvalSuiteSpec {
  id: string;
  dataset: string;
  description?: string;
  source?: SuiteSource;
  citySlices?: string[];
  languageSlices?: string[];
  styleSlices?: string[];
  tags?: string[];
}

export interface EvalSuiteManifest {
  name: string;
  description: string;
  suites: EvalSuiteSpec[];
}

export interface EvalSuiteSlices {
  source: SuiteSource;
  citySlices: string[];
  languageSlices: string[];
  styleSlices: string[];
  tags: string[];
}

export interface AggregateCategory {
  category: string;
  pass: number;
  fail: number;
  skip: number;
  total: number;
}

export interface AggregateCoverage {
  sources: string[];
  citySlices: string[];
  languageSlices: string[];
  styleSlices: string[];
  tags: string[];
}

export interface AggregateSummary {
  suites: number;
  sessions: number;
  probes: number;
  pass: number;
  fail: number;
  skip: number;
  passRate: number;
  categories: AggregateCategory[];
  coverage: AggregateCoverage;
}

export interface EvalSuiteRun {
  manifest: string;
  description: string;
  pipeline: "llm" | "offline";
  model: string | null;
  generatedAt: string;
  summary: AggregateSummary;
  suites: Array<{
    id: string;
    dataset: string;
    description: string;
    slices: EvalSuiteSlices;
    sessions: number;
    report: EvalReport;
  }>;
}

export interface ReplaySuiteRun {
  manifest: string;
  description: string;
  pipeline: "llm" | "offline";
  model: string | null;
  generatedAt: string;
  summary: AggregateSummary;
  suites: Array<{
    id: string;
    dataset: string;
    description: string;
    slices: EvalSuiteSlices;
    sessions: number;
    snapshot: ReplaySnapshot;
  }>;
}

export interface CompareSuiteRun {
  manifest: string;
  description: string;
  live: boolean;
  model: string;
  generatedAt: string;
  offline: AggregateSummary;
  llm: AggregateSummary;
  suites: Array<{
    id: string;
    dataset: string;
    description: string;
    slices: EvalSuiteSlices;
    sessions: number;
    offline: EvalReport;
    llm: EvalReport;
    live: boolean;
    model: string;
    unlockedCategories: string[];
  }>;
}

function argValue(name: string, fallback: string): string {
  return process.argv.find((a) => a.startsWith(`${name}=`))?.slice(name.length + 1) ?? fallback;
}

function resolveEvalPath(path: string): string {
  return isAbsolute(path) ? path : join(here, path);
}

export async function loadSuiteManifest(path = "suites.json"): Promise<EvalSuiteManifest> {
  const manifest = JSON.parse(await readFile(resolveEvalPath(path), "utf-8")) as EvalSuiteManifest;
  if (!manifest.suites?.length) throw new Error(`suite manifest ${path} has no suites`);
  return manifest;
}

export async function loadSuiteDataset(spec: EvalSuiteSpec): Promise<EvalDataset> {
  return JSON.parse(await readFile(resolveEvalPath(spec.dataset), "utf-8")) as EvalDataset;
}

export async function runEvalSuites(
  manifest: EvalSuiteManifest,
  opts: { config?: Config; llmAvailable?: boolean; generatedAt?: string } = {},
): Promise<EvalSuiteRun> {
  const generatedAt = opts.generatedAt ?? new Date().toISOString();
  const { cfg, llmAvailable } = resolveSuiteConfig(opts.config, opts.llmAvailable);
  const suites: EvalSuiteRun["suites"] = [];

  for (const spec of manifest.suites) {
    const dataset = await loadSuiteDataset(spec);
    const mem = buildOpenMap(cfg, { allowHeuristicFallbackForTests: !llmAvailable });
    const report = await runEval(dataset, mem, { llmAvailable });
    suites.push({
      id: spec.id,
      dataset: dataset.name,
      description: spec.description ?? dataset.description,
      slices: suiteSlices(spec),
      sessions: dataset.sessions.length,
      report,
    });
  }

  return {
    manifest: manifest.name,
    description: manifest.description,
    pipeline: llmAvailable ? "llm" : "offline",
    model: llmAvailable ? cfg.openaiChatModel : null,
    generatedAt,
    summary: aggregateReports(manifest.suites, suites.map((s) => s.report), suites.map((s) => s.sessions)),
    suites,
  };
}

export async function buildSuiteReplay(
  manifest: EvalSuiteManifest,
  opts: { config?: Config; llmAvailable?: boolean; generatedAt?: string } = {},
): Promise<ReplaySuiteRun> {
  const generatedAt = opts.generatedAt ?? new Date().toISOString();
  const { cfg, llmAvailable } = resolveSuiteConfig(opts.config, opts.llmAvailable);
  const suites: ReplaySuiteRun["suites"] = [];

  for (const spec of manifest.suites) {
    const dataset = await loadSuiteDataset(spec);
    const mem = buildOpenMap(cfg, { allowHeuristicFallbackForTests: !llmAvailable });
    const snapshot = await buildReplaySnapshot(dataset, mem, {
      llmAvailable,
      model: llmAvailable ? cfg.openaiChatModel : null,
      generatedAt,
    });
    suites.push({
      id: spec.id,
      dataset: dataset.name,
      description: spec.description ?? dataset.description,
      slices: suiteSlices(spec),
      sessions: dataset.sessions.length,
      snapshot,
    });
  }

  return {
    manifest: manifest.name,
    description: manifest.description,
    pipeline: llmAvailable ? "llm" : "offline",
    model: llmAvailable ? cfg.openaiChatModel : null,
    generatedAt,
    summary: aggregateReports(
      manifest.suites,
      suites.map((s) => s.snapshot.report),
      suites.map((s) => s.sessions),
    ),
    suites,
  };
}

export async function compareEvalSuites(
  manifest: EvalSuiteManifest,
  opts: { runner?: LLMRunner; generatedAt?: string } = {},
): Promise<CompareSuiteRun> {
  const generatedAt = opts.generatedAt ?? new Date().toISOString();
  const suites: CompareSuiteRun["suites"] = [];

  for (const spec of manifest.suites) {
    const dataset = await loadSuiteDataset(spec);
    const result = await compare(dataset, { runner: opts.runner });
    suites.push({
      id: spec.id,
      dataset: dataset.name,
      description: spec.description ?? dataset.description,
      slices: suiteSlices(spec),
      sessions: dataset.sessions.length,
      offline: result.offline,
      llm: result.llm,
      live: result.live,
      model: result.model,
      unlockedCategories: unlockedCategories(result.offline, result.llm),
    });
  }

  return {
    manifest: manifest.name,
    description: manifest.description,
    live: suites.some((s) => s.live),
    model: unique(suites.map((s) => s.model)).join(", "),
    generatedAt,
    offline: aggregateReports(manifest.suites, suites.map((s) => s.offline), suites.map((s) => s.sessions)),
    llm: aggregateReports(manifest.suites, suites.map((s) => s.llm), suites.map((s) => s.sessions)),
    suites,
  };
}

function resolveSuiteConfig(config?: Config, llmAvailableOverride?: boolean): { cfg: Config; llmAvailable: boolean } {
  const cfg: Config = { ...(config ?? loadConfig()), dbPath: ":memory:" };
  const llmAvailable = llmAvailableOverride ?? Boolean(cfg.openaiApiKey);
  if (!llmAvailable) {
    cfg.openaiApiKey = null;
    cfg.embedder = "none";
    cfg.tagger = "lexicon";
  }
  return { cfg, llmAvailable };
}

function suiteSlices(spec: EvalSuiteSpec): EvalSuiteSlices {
  return {
    source: spec.source ?? "synthetic",
    citySlices: spec.citySlices ?? [],
    languageSlices: spec.languageSlices ?? [],
    styleSlices: spec.styleSlices ?? [],
    tags: spec.tags ?? [],
  };
}

function aggregateReports(specs: EvalSuiteSpec[], reports: EvalReport[], sessions: number[] = []): AggregateSummary {
  const pass = reports.reduce((n, r) => n + r.pass, 0);
  const fail = reports.reduce((n, r) => n + r.fail, 0);
  const skip = reports.reduce((n, r) => n + r.skip, 0);
  const probes = reports.reduce((n, r) => n + r.total, 0);
  const executable = Math.max(1, probes - skip);
  return {
    suites: reports.length,
    sessions: sessions.reduce((n, s) => n + s, 0),
    probes,
    pass,
    fail,
    skip,
    passRate: Math.round((pass / executable) * 100),
    categories: aggregateCategories(reports),
    coverage: aggregateCoverage(specs),
  };
}

function aggregateCategories(reports: EvalReport[]): AggregateCategory[] {
  const counts = new Map<string, AggregateCategory>();
  for (const report of reports) {
    for (const result of report.results) {
      const row = counts.get(result.category) ?? { category: result.category, pass: 0, fail: 0, skip: 0, total: 0 };
      row[result.status] += 1;
      row.total += 1;
      counts.set(result.category, row);
    }
  }
  return [...counts.values()].sort((a, b) => a.category.localeCompare(b.category));
}

function aggregateCoverage(specs: EvalSuiteSpec[]): AggregateCoverage {
  return {
    sources: unique(specs.map((s) => s.source ?? "synthetic")),
    citySlices: unique(specs.flatMap((s) => s.citySlices ?? [])),
    languageSlices: unique(specs.flatMap((s) => s.languageSlices ?? [])),
    styleSlices: unique(specs.flatMap((s) => s.styleSlices ?? [])),
    tags: unique(specs.flatMap((s) => s.tags ?? [])),
  };
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function unlockedCategories(offline: EvalReport, llm: EvalReport): string[] {
  const unlocked = offline.results
    .filter((o, i) => o.status !== "pass" && llm.results[i]?.status === "pass")
    .map((o) => o.category);
  return unique(unlocked);
}

function formatSummary(summary: AggregateSummary): string {
  return `${summary.pass}/${summary.probes - summary.skip} passed (${summary.passRate}%), ${summary.fail} failed, ${summary.skip} skipped`;
}

function printEvalRun(run: EvalSuiteRun): void {
  console.log(`\n  ${run.manifest} - ${run.pipeline}${run.model ? ` (${run.model})` : ""}`);
  console.log(`  ${formatSummary(run.summary)} across ${run.summary.suites} suites / ${run.summary.sessions} sessions`);
  console.log(`  coverage: cities=${run.summary.coverage.citySlices.join(", ") || "-"} languages=${run.summary.coverage.languageSlices.join(", ") || "-"} styles=${run.summary.coverage.styleSlices.join(", ") || "-"}\n`);
  for (const suite of run.suites)
    console.log(`  ${suite.id.padEnd(8)} ${String(suite.report.pass).padStart(2)}/${String(suite.report.total - suite.report.skip).padEnd(2)} pass  fail=${suite.report.fail} skip=${suite.report.skip}  ${suite.dataset}`);
  console.log("");
}

function printCompareRun(run: CompareSuiteRun): void {
  console.log(`\n  ${run.manifest} - offline vs LLM ${run.live ? `(live: ${run.model})` : "(stub)"}`);
  console.log(`  offline: ${formatSummary(run.offline)}`);
  console.log(`  llm    : ${formatSummary(run.llm)}\n`);
  for (const suite of run.suites) {
    const unlocked = suite.unlockedCategories.length ? suite.unlockedCategories.join(", ") : "-";
    console.log(`  ${suite.id.padEnd(8)} offline ${String(suite.offline.pass).padStart(2)}/${String(suite.offline.total - suite.offline.skip).padEnd(2)}  llm ${String(suite.llm.pass).padStart(2)}/${String(suite.llm.total - suite.llm.skip).padEnd(2)}  unlocks=${unlocked}`);
  }
  console.log("");
}

async function main() {
  const manifest = await loadSuiteManifest(argValue("--manifest", "suites.json"));
  const mode = argValue("--mode", "eval");
  const out = process.argv.find((a) => a.startsWith("--out="))?.slice("--out=".length);
  const forceOffline = process.argv.includes("--offline");
  const cfg = { ...loadConfig(), dbPath: ":memory:" };
  if (forceOffline) {
    cfg.openaiApiKey = null;
    cfg.embedder = "none";
    cfg.tagger = "lexicon";
  }

  if (mode === "compare") {
    const result = await compareEvalSuites(manifest);
    if (out) await writeFile(out, JSON.stringify(result, null, 2) + "\n", "utf-8");
    printCompareRun(result);
    if (result.llm.fail > 0 || result.offline.fail > 0) process.exitCode = 1;
    return;
  }

  if (mode === "replay") {
    const result = await buildSuiteReplay(manifest, { config: cfg });
    const text = JSON.stringify(result, null, 2) + "\n";
    if (out) await writeFile(out, text, "utf-8");
    else process.stdout.write(text);
    if (result.summary.fail > 0) process.exitCode = 1;
    return;
  }

  if (mode !== "eval") throw new Error(`unknown suite mode: ${mode}`);
  const result = await runEvalSuites(manifest, { config: cfg });
  if (out) await writeFile(out, JSON.stringify(result, null, 2) + "\n", "utf-8");
  printEvalRun(result);
  if (result.summary.fail > 0) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) main();

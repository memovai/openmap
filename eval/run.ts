#!/usr/bin/env node
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, isAbsolute, join } from "node:path";
import { buildOpenMap } from "../src/openmap.js";
import { loadConfig } from "../src/core/config.js";
import { runEval, type EvalDataset } from "./harness.js";

const here = dirname(fileURLToPath(import.meta.url));

function argValue(name: string, fallback: string): string {
  return process.argv.find((a) => a.startsWith(`${name}=`))?.slice(name.length + 1) ?? fallback;
}

function resolveDatasetPath(path: string): string {
  return isAbsolute(path) ? path : join(here, path);
}

async function main() {
  const datasetPath = resolveDatasetPath(argValue("--dataset", "dataset.json"));
  const dataset = JSON.parse(await readFile(datasetPath, "utf-8")) as EvalDataset;
  const cfg = { ...loadConfig(), dbPath: ":memory:" };
  const mem = buildOpenMap(cfg);
  const llmAvailable = Boolean(cfg.openaiApiKey);

  const report = await runEval(dataset, mem, { llmAvailable });

  console.log(`\n  ${dataset.name} — ${dataset.sessions.length} sessions, ${report.total} probes`);
  console.log(`  pipeline: ${llmAvailable ? "LLM" : "offline (lexicon/heuristic)"}\n`);
  const icon = (s: string) => (s === "pass" ? "✓" : s === "fail" ? "✗" : "–");
  for (const r of report.results)
    console.log(`  ${icon(r.status)} [${r.category}] ${r.desc}   →  ${r.got}`);
  const rate = report.total - report.skip > 0 ? Math.round((report.pass / (report.total - report.skip)) * 100) : 0;
  console.log(`\n  ${report.pass}/${report.total - report.skip} passed (${rate}%)  ·  ${report.skip} skipped\n`);
  if (report.fail > 0) process.exitCode = 1;
}

main();

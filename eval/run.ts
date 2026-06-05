#!/usr/bin/env node
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildOpenMap } from "../src/openmap.js";
import { loadConfig } from "../src/core/config.js";
import { runEval, type EvalDataset } from "./harness.js";

const here = dirname(fileURLToPath(import.meta.url));

async function main() {
  const dataset = JSON.parse(await readFile(join(here, "dataset.json"), "utf-8")) as EvalDataset;
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

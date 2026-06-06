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
import { getRunner } from "../src/nlp/llm.js";
import { runEval, type EvalDataset, type EvalReport } from "./harness.js";
import { StubLLMRunner } from "./stub-llm.js";
import { type LLMRunner } from "../src/nlp/llm.js";

const here = dirname(fileURLToPath(import.meta.url));

function argValue(name: string, fallback: string): string {
  return process.argv.find((a) => a.startsWith(`${name}=`))?.slice(name.length + 1) ?? fallback;
}

function resolveDatasetPath(path: string): string {
  return isAbsolute(path) ? path : join(here, path);
}

/** Run the dataset under offline vs LLM pipelines. The LLM runner is the real
 * model when a key is set, else a transparent stub — or an injected one (tests
 * pass a stub to stay deterministic). Embeddings are local in both columns so
 * the comparison isolates extraction quality. */
export async function compare(
  dataset: EvalDataset,
  opts: { runner?: LLMRunner } = {},
): Promise<{ offline: EvalReport; llm: EvalReport; live: boolean; model: string }> {
  const cfg = loadConfig();
  const base = { ...cfg, dbPath: ":memory:", embedder: "none" as const }; // hold embeddings out → compare extraction only

  const offlineMem = buildOpenMap({ ...base, openaiApiKey: null, tagger: "lexicon" });
  const offline = await runEval(dataset, offlineMem, { llmAvailable: false });

  const realKey = Boolean(cfg.openaiApiKey);
  const runner = opts.runner ?? (realKey ? getRunner(cfg)! : new StubLLMRunner());
  const live = !opts.runner && realKey;
  const llmMem = buildOpenMap({ ...base }, { llm: runner });
  const llm = await runEval(dataset, llmMem, { llmAvailable: true });

  return { offline, llm, live, model: live ? cfg.openaiChatModel : "stub" };
}

async function main() {
  const datasetPath = resolveDatasetPath(argValue("--dataset", "dataset.json"));
  const dataset = JSON.parse(await readFile(datasetPath, "utf-8")) as EvalDataset;
  const { offline, llm, live, model } = await compare(dataset);
  const icon = (s: string) => (s === "pass" ? "✓" : s === "fail" ? "✗" : "–");

  console.log(`\n  ${dataset.name} — offline vs LLM ${live ? `(live: ${model})` : "(stub — set GEMINI_API_KEY or OPENAI_API_KEY for real)"}\n`);
  console.log(`  ${"category".padEnd(16)} ${"probe".padEnd(42)} off  llm`);
  console.log(`  ${"-".repeat(72)}`);
  for (let i = 0; i < offline.results.length; i++) {
    const o = offline.results[i]!;
    const l = llm.results[i]!;
    const flag = o.status !== "pass" && l.status === "pass" ? "  ← LLM unlocks" : "";
    console.log(`  ${o.category.padEnd(16)} ${o.desc.slice(0, 42).padEnd(42)} ${icon(o.status)}    ${icon(l.status)}${flag}`);
  }
  const rate = (r: EvalReport) => Math.round((r.pass / Math.max(1, r.total - r.skip)) * 100);
  console.log(`  ${"-".repeat(72)}`);
  console.log(`  offline: ${offline.pass}/${offline.total - offline.skip} (${rate(offline)}%), ${offline.skip} skipped`);
  console.log(`  llm    : ${llm.pass}/${llm.total - llm.skip} (${rate(llm)}%), ${llm.skip} skipped`);
  const unlocked = offline.results.filter((o, i) => o.status !== "pass" && llm.results[i]!.status === "pass").map((o) => o.category);
  console.log(`  LLM unlocks: ${unlocked.length ? [...new Set(unlocked)].join(", ") : "(none)"}\n`);
}

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) main();

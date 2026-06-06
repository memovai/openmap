#!/usr/bin/env node
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadConfig } from "../src/core/config.js";
import { type EvalDataset } from "./harness.js";
import {
  createGbrainCliProvider,
  createMem0Provider,
  createOpenMapProvider,
  providerModelSummary,
  runCommandProviderEval,
  runProviderEval,
  unavailableProviderReport,
  type ProviderCompareRun,
  type ProviderEvalReport,
} from "./provider-harness.js";

const here = dirname(fileURLToPath(import.meta.url));

function argValue(name: string, fallback: string): string {
  return process.argv.find((a) => a.startsWith(`${name}=`))?.slice(name.length + 1) ?? fallback;
}

function resolveEvalPath(path: string): string {
  return isAbsolute(path) ? path : join(here, path);
}

async function loadDataset(path: string): Promise<EvalDataset> {
  return JSON.parse(await readFile(resolveEvalPath(path), "utf-8")) as EvalDataset;
}

export async function runProviderCompare(
  dataset: EvalDataset,
  opts: { providers?: string[]; generatedAt?: string; offline?: boolean } = {},
): Promise<ProviderCompareRun> {
  const generatedAt = opts.generatedAt ?? new Date().toISOString();
  const providers = expandProviders(opts.providers ?? ["openmap", "mem0", "tencentdb", "gbrain"]);
  const reports: ProviderEvalReport[] = [];
  const cfg = { ...loadConfig(), dbPath: ":memory:" };
  if (opts.offline) {
    cfg.openaiApiKey = null;
    cfg.embedder = "none";
    cfg.tagger = "lexicon";
  }

  for (const provider of providers) {
    if (provider === "openmap") {
      reports.push(await runProviderEval(dataset, createOpenMapProvider(cfg, { llmAvailable: !opts.offline && Boolean(cfg.openaiApiKey) }), { llmAvailable: !opts.offline && Boolean(cfg.openaiApiKey) }));
      continue;
    }

    if (provider === "mem0") {
      const created = await createMem0Provider(dataset, cfg);
      if ("status" in created) reports.push(created);
      else reports.push(await runProviderEval(dataset, created, { llmAvailable: true }));
      continue;
    }

    if (provider === "gbrain") {
      const command = process.env.OPENMAP_EVAL_GBRAIN_COMMAND;
      if (command) reports.push(await runCommandProviderEval(provider, command, dataset, cfg));
      else {
        const created = await createGbrainCliProvider(cfg);
        if ("status" in created) reports.push(created);
        else reports.push(await runProviderEval(dataset, created, { llmAvailable: false }));
      }
      continue;
    }

    if (provider === "tencentdb") {
      const command = process.env[`OPENMAP_EVAL_${provider.toUpperCase()}_COMMAND`];
      reports.push(
        command
          ? await runCommandProviderEval(provider, command, dataset, cfg)
          : unavailableProviderReport(provider, `set OPENMAP_EVAL_${provider.toUpperCase()}_COMMAND to an adapter command`),
      );
      continue;
    }

    if (provider.startsWith("cmd:")) {
      const [name, ...cmdParts] = provider.slice("cmd:".length).split("=");
      const command = cmdParts.join("=");
      reports.push(
        name && command
          ? await runCommandProviderEval(name, command, dataset, cfg)
          : unavailableProviderReport(provider, "custom command provider must be cmd:name=command"),
      );
      continue;
    }

    reports.push(unavailableProviderReport(provider, "unknown provider"));
  }

  return {
    dataset: dataset.name,
    description: dataset.description,
    user: dataset.user,
    generatedAt,
    model: providerModelSummary(cfg),
    providers: reports,
  };
}

function expandProviders(values: string[]): string[] {
  const out = values.flatMap((v) => v.split(",")).map((v) => v.trim()).filter(Boolean);
  if (out.includes("all")) return ["openmap", "mem0", "tencentdb", "gbrain"];
  return [...new Set(out)];
}

function printRun(run: ProviderCompareRun): void {
  console.log(`\n  ${run.dataset} - provider comparison`);
  console.log(`  ${run.description}\n`);
  console.log(`  model: chat=${run.model.chatModel} embed=${run.model.embeddingModel} dims=${run.model.embeddingDimensions}${run.model.baseUrl ? ` base=${run.model.baseUrl}` : ""}\n`);
  console.log("  provider    status       pass/supported  fail  unsupported  notes");
  console.log("  ----------  -----------  --------------  ----  -----------  ------------------------------");
  for (const p of run.providers) {
    const status = p.status.padEnd(11);
    const ratio = `${p.pass}/${p.supported}`.padEnd(14);
    const fail = String(p.fail + p.error).padStart(4);
    const unsupported = String(p.unsupported).padStart(11);
    const notes = p.reason ?? `${p.passRate}% pass · ${p.capabilities.join(", ") || "-"}`;
    console.log(`  ${p.provider.padEnd(10)}  ${status}  ${ratio}  ${fail}  ${unsupported}  ${notes}`);
  }
  console.log("");

  for (const p of run.providers.filter((r) => r.status === "ok" && r.results.some((x) => x.status !== "pass" && x.status !== "unsupported" && x.status !== "skip"))) {
    console.log(`  ${p.provider} non-pass probes:`);
    for (const r of p.results.filter((x) => x.status !== "pass" && x.status !== "unsupported" && x.status !== "skip"))
      console.log(`   ${r.status.padEnd(5)} [${r.category}] ${r.desc} -> ${r.got}`);
    console.log("");
  }
}

async function main() {
  const dataset = await loadDataset(argValue("--dataset", "dataset.json"));
  const providers = expandProviders([argValue("--providers", "all")]);
  const out = process.argv.find((a) => a.startsWith("--out="))?.slice("--out=".length);
  const offline = process.argv.includes("--offline");
  const run = await runProviderCompare(dataset, { providers, offline });
  if (out) await writeFile(out, JSON.stringify(run, null, 2) + "\n", "utf-8");
  printRun(run);
  if (run.providers.some((p) => p.status === "ok" && (p.fail > 0 || p.error > 0))) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) main();

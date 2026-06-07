#!/usr/bin/env node
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { writeFile, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, isAbsolute, join } from "node:path";
import { buildOpenMap, type OpenMap } from "../src/openmap.js";
import { loadConfig } from "../src/core/config.js";
import { runEval, type EvalDataset, type EvalReport } from "./harness.js";

const here = dirname(fileURLToPath(import.meta.url));

function argValue(name: string, fallback: string): string {
  return process.argv.find((a) => a.startsWith(`${name}=`))?.slice(name.length + 1) ?? fallback;
}

function resolveDatasetPath(path: string): string {
  return isAbsolute(path) ? path : join(here, path);
}

export interface ReplaySnapshot {
  dataset: string;
  description: string;
  user: string;
  pipeline: "llm" | "offline";
  model: string | null;
  generatedAt: string;
  report: EvalReport;
  recall: Array<{ query: string; names: string[] }>;
  beliefs: Array<{ predicate: string; object: string; confidence: number; source: string }>;
  scenarios: Array<{ title: string; placeNames: string[]; intents: string[]; concepts: string[]; turnCount: number }>;
  routines: Array<{
    title: string;
    summary: string;
    support: number;
    confidence: number;
    placeNames: string[];
    positivePlaceNames: string[];
    negativePlaceNames: string[];
    intents: string[];
    concepts: string[];
  }>;
  profile: Record<string, unknown>;
}

export async function buildReplaySnapshot(
  dataset: EvalDataset,
  mem: OpenMap,
  opts: { llmAvailable?: boolean; model?: string | null; generatedAt?: string } = {},
): Promise<ReplaySnapshot> {
  const report = await runEval(dataset, mem, { llmAvailable: opts.llmAvailable });
  const recallProbes = dataset.probes.filter((p) => p.kind === "recall");
  const recall = [];
  for (const p of recallProbes) {
    const res = await mem.recall(String(p.query), null, 5, dataset.user);
    recall.push({ query: String(p.query), names: res.map((r) => r.place.name) });
  }
  const placeName = (id: string) => mem.db.getPlace(id)?.name ?? id;
  return {
    dataset: dataset.name,
    description: dataset.description,
    user: dataset.user,
    pipeline: opts.llmAvailable ? "llm" : "offline",
    model: opts.model ?? null,
    generatedAt: opts.generatedAt ?? new Date().toISOString(),
    report,
    recall,
    beliefs: mem.beliefs(dataset.user, { minConfidence: 0.3 }).map((b) => ({
      predicate: b.predicate,
      object: b.object,
      confidence: b.confidence,
      source: b.source,
    })),
    scenarios: mem.scenarios(dataset.user, { limit: 50 }).map((s) => ({
      title: s.title,
      placeNames: s.placeIds.map(placeName),
      intents: s.intents,
      concepts: s.concepts,
      turnCount: s.turnIds.length,
    })),
    routines: mem.routines(dataset.user, { limit: 50 }).map((r) => ({
      title: r.title,
      summary: r.summary,
      support: r.support,
      confidence: r.confidence,
      placeNames: r.placeIds.map(placeName),
      positivePlaceNames: r.positivePlaceIds.map(placeName),
      negativePlaceNames: r.negativePlaceIds.map(placeName),
      intents: r.intents,
      concepts: r.concepts,
    })),
    profile: mem.tasteProfile(dataset.user),
  };
}

async function main() {
  const datasetPath = resolveDatasetPath(argValue("--dataset", "dataset.json"));
  const dataset = JSON.parse(await readFile(datasetPath, "utf-8")) as EvalDataset;
  const cfg = { ...loadConfig(), dbPath: ":memory:" };
  const llmAvailable = Boolean(cfg.openaiApiKey);
  const mem = buildOpenMap(cfg, { allowHeuristicFallbackForTests: !llmAvailable });
  const snapshot = await buildReplaySnapshot(dataset, mem, {
    llmAvailable,
    model: llmAvailable ? cfg.openaiChatModel : null,
  });
  const text = JSON.stringify(snapshot, null, 2) + "\n";
  const outArg = process.argv.find((a) => a.startsWith("--out="));
  if (outArg) await writeFile(outArg.slice("--out=".length), text, "utf-8");
  else process.stdout.write(text);
  if (snapshot.report.fail > 0) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) main();

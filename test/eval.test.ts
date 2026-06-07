import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { buildOpenMap } from "../src/openmap.js";
import { LEXICON_DISABLED_ERROR, MODEL_REQUIRED_ERROR, loadConfig, resolvedEmbedder } from "../src/core/config.js";
import { runEval, type EvalDataset } from "../eval/harness.js";
import { compare } from "../eval/compare.js";
import { StubLLMRunner } from "../eval/stub-llm.js";
import { buildReplaySnapshot, type ReplaySnapshot } from "../eval/replay.js";
import { diffReplaySnapshots } from "../eval/replay-diff.js";
import { buildSuiteReplay, loadSuiteManifest, runEvalSuites } from "../eval/suites.js";
import { runProviderCompare } from "../eval/provider-compare.js";

const loadDataset = async (name: string): Promise<EvalDataset> =>
  JSON.parse(await readFile(new URL(`../eval/${name}`, import.meta.url), "utf-8")) as EvalDataset;
const testOpenMap = () =>
  buildOpenMap({ ...loadConfig({}), dbPath: ":memory:" }, { allowHeuristicFallbackForTests: true });

test("public OpenMap builder fails fast without a model", () => {
  assert.throws(
    () => buildOpenMap({ ...loadConfig({}), dbPath: ":memory:" }),
    new RegExp(MODEL_REQUIRED_ERROR.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
  );
  assert.throws(
    () => buildOpenMap({ ...loadConfig({ OPENAI_API_KEY: "test" } as NodeJS.ProcessEnv), dbPath: ":memory:", tagger: "lexicon" }),
    new RegExp(LEXICON_DISABLED_ERROR.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
  );
});

test("place-memory mini eval: all offline probes pass", async () => {
  const dataset = await loadDataset("dataset.json");
  const mem = testOpenMap(); // explicit no-API heuristic path for tests
  const report = await runEval(dataset, mem, { llmAvailable: false });

  const failed = report.results.filter((r) => r.status === "fail");
  assert.deepEqual(
    failed.map((f) => `${f.desc} → ${f.got}`),
    [],
    "no probe should fail offline",
  );
  assert.ok(report.pass >= 9, `expected ≥9 passing probes, got ${report.pass}`);
});

test("field replay eval: broader map-assistant transcripts pass offline", async () => {
  const dataset = await loadDataset("field-dataset.json");
  const mem = testOpenMap();
  const report = await runEval(dataset, mem, { llmAvailable: false });
  const failed = report.results.filter((r) => r.status === "fail");
  assert.deepEqual(
    failed.map((f) => `${f.desc} → ${f.got}`),
    [],
    "field replay probes should pass offline",
  );
  assert.equal(report.skip, 0);
});

test("aggregate replay suite manifest gates all local suites offline", async () => {
  const manifest = await loadSuiteManifest("suites.json");
  const run = await runEvalSuites(manifest, {
    config: { ...loadConfig({}), dbPath: ":memory:" },
    llmAvailable: false,
    generatedAt: "test",
  });

  assert.equal(run.summary.suites, 2);
  assert.equal(run.summary.sessions, 17);
  assert.equal(run.summary.fail, 0);
  assert.equal(run.summary.skip, 4);
  assert.ok(run.summary.probes >= 55);
  assert.ok(run.summary.coverage.citySlices.includes("Shanghai"));
  assert.ok(run.summary.coverage.languageSlices.includes("zh"));

  const replay = await buildSuiteReplay(manifest, {
    config: { ...loadConfig({}), dbPath: ":memory:" },
    llmAvailable: false,
    generatedAt: "test",
  });
  assert.equal(replay.summary.fail, 0);
  assert.equal(replay.suites.length, run.suites.length);
  assert.ok(replay.suites.some((s) => s.snapshot.routines.length > 0));
});

test("provider comparison runs openmap and isolates unavailable external providers", async () => {
  const dataset = await loadDataset("dataset.json");
  const run = await runProviderCompare(dataset, {
    providers: ["openmap", "tencentdb"],
    generatedAt: "test",
    offline: true,
  });
  const openmap = run.providers.find((p) => p.provider === "openmap");
  const tencentdb = run.providers.find((p) => p.provider === "tencentdb");

  assert.equal(openmap?.status, "ok");
  assert.equal(openmap.fail, 0);
  assert.ok(openmap.pass >= 9);
  assert.equal(tencentdb?.status, "unavailable");
  assert.match(tencentdb.reason ?? "", /OPENMAP_EVAL_TENCENTDB_COMMAND/);
});

test("a Gemini key routes through the OpenAI-compatible endpoint", () => {
  const cfg = loadConfig({ GEMINI_API_KEY: "test-key" } as NodeJS.ProcessEnv);
  assert.equal(cfg.openaiApiKey, "test-key");
  assert.match(cfg.openaiBaseUrl ?? "", /generativelanguage\.googleapis\.com/);
  assert.equal(cfg.openaiChatModel, "gemini-2.5-flash-lite");
  assert.equal(cfg.models.extractor, "gemini-2.5-flash-lite");
  assert.equal(cfg.openaiEmbedModel, "gemini-embedding-001"); // Gemini embedding model auto-selected
  assert.equal(resolvedEmbedder(cfg), "openai"); // real embeddings on (hybrid recall)
});

test("offline vs LLM(stub): LLM unlocks region / mixed-sentiment / multi-hop", async () => {
  const dataset = await loadDataset("dataset.json");
  const { offline, llm } = await compare(dataset, { runner: new StubLLMRunner() }); // forced stub → deterministic, no network
  // the stub LLM should pass everything (incl. the llmOnly probes offline skips)
  assert.equal(llm.fail, 0, "LLM(stub) should have no failures");
  assert.ok(llm.pass > offline.pass, `LLM should unlock more probes (llm ${llm.pass} > offline ${offline.pass})`);
  // offline skips exactly the llmOnly probes
  assert.equal(offline.skip, dataset.probes.filter((p) => p.llmOnly).length);
});

test("field replay eval: LLM(stub) path also passes", async () => {
  const dataset = await loadDataset("field-dataset.json");
  const { offline, llm } = await compare(dataset, { runner: new StubLLMRunner() });
  assert.equal(offline.fail, 0);
  assert.equal(llm.fail, 0);
  assert.equal(offline.skip, 0);
});

test("eval replay snapshot captures report, recalls, scenarios, routines, and profile", async () => {
  const dataset = await loadDataset("dataset.json");
  const mem = testOpenMap();
  const snapshot = await buildReplaySnapshot(dataset, mem, { llmAvailable: false, generatedAt: "test" });
  assert.equal(snapshot.dataset, dataset.name);
  assert.equal(snapshot.generatedAt, "test");
  assert.equal(snapshot.report.fail, 0);
  assert.ok(snapshot.recall.some((r) => r.query === "a quiet place for solo work calls" && r.names.includes("Botanica")));
  assert.ok(snapshot.scenarios.some((s) => s.placeNames.includes("Botanica") && s.intents.includes("work")));
  assert.ok(snapshot.routines.some((r) => r.title.includes("focus") && r.concepts.includes("quiet") && r.positivePlaceNames.includes("Cowork Library")));
  assert.ok(snapshot.beliefs.some((b) => b.predicate === "avoids" && b.object === "loud"));
  assert.equal(snapshot.profile.userId, dataset.user);
});

test("eval replay diff gates regressions and rank drift", () => {
  const before = tinySnapshot();
  const after: ReplaySnapshot = {
    ...tinySnapshot(),
    generatedAt: "after",
    report: {
      ...before.report,
      pass: before.report.pass - 1,
      fail: before.report.fail + 1,
      results: before.report.results.map((r) =>
        r.category === "assistant-map"
          ? { ...r, status: "fail", got: "Quiet Beans > Botanica" }
          : r,
      ),
    },
    recall: [{ query: "a quiet place for solo work calls", names: ["Quiet Beans", "Botanica"] }],
    beliefs: [{ predicate: "likes", object: "quiet", confidence: 0.4, source: "inferred" }],
  };
  const diff = diffReplaySnapshots(before, after);
  assert.equal(diff.ok, false);
  assert.ok(diff.changes.some((c) => c.severity === "regression" && c.area === "probe"));
  assert.ok(diff.changes.some((c) => c.severity === "warning" && c.area === "recall"));
  assert.ok(diff.changes.some((c) => c.severity === "regression" && c.area === "belief"));
});

test("eval replay diff can fail on warnings via threshold", () => {
  const before = tinySnapshot();
  const after: ReplaySnapshot = {
    ...tinySnapshot(),
    generatedAt: "after",
    recall: [{ query: "a quiet place for solo work calls", names: ["Quiet Beans", "Botanica"] }],
  };
  assert.equal(diffReplaySnapshots(before, after).ok, true);
  assert.equal(diffReplaySnapshots(before, after, { maxWarnings: 0 }).ok, false);
});

function tinySnapshot(): ReplaySnapshot {
  return {
    dataset: "place-memory-mini",
    description: "test",
    user: "alex",
    pipeline: "offline",
    model: null,
    generatedAt: "before",
    report: {
      pass: 1,
      fail: 0,
      skip: 0,
      total: 1,
      results: [
        {
          category: "assistant-map",
          desc: 'recall "a quiet place for solo work calls" → top≈Botanica',
          status: "pass",
          got: "Botanica > Quiet Beans",
        },
      ],
    },
    recall: [{ query: "a quiet place for solo work calls", names: ["Botanica", "Quiet Beans"] }],
    beliefs: [
      { predicate: "likes", object: "quiet", confidence: 0.8, source: "inferred" },
      { predicate: "avoids", object: "loud", confidence: 0.7, source: "inferred" },
    ],
    scenarios: [
      {
        title: "work: Botanica",
        placeNames: ["Botanica"],
        intents: ["work"],
        concepts: ["quiet"],
        turnCount: 2,
      },
    ],
    routines: [
      {
        title: "focus: quiet",
        summary: "2 related focus scenarios; intents work/study; looks for quiet; preferred Botanica",
        support: 2,
        confidence: 0.73,
        placeNames: ["Botanica"],
        positivePlaceNames: ["Botanica"],
        negativePlaceNames: [],
        intents: ["work", "study"],
        concepts: ["quiet"],
      },
    ],
    profile: { userId: "alex" },
  };
}

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { buildOpenMap } from "../src/openmap.js";
import { loadConfig, resolvedEmbedder } from "../src/core/config.js";
import { runEval, type EvalDataset } from "../eval/harness.js";
import { compare } from "../eval/compare.js";
import { StubLLMRunner } from "../eval/stub-llm.js";

test("place-memory mini eval: all offline probes pass", async () => {
  const dataset = JSON.parse(
    await readFile(new URL("../eval/dataset.json", import.meta.url), "utf-8"),
  ) as EvalDataset;
  const mem = buildOpenMap({ ...loadConfig({}), dbPath: ":memory:" }); // offline (empty env)
  const report = await runEval(dataset, mem, { llmAvailable: false });

  const failed = report.results.filter((r) => r.status === "fail");
  assert.deepEqual(
    failed.map((f) => `${f.desc} → ${f.got}`),
    [],
    "no probe should fail offline",
  );
  assert.ok(report.pass >= 9, `expected ≥9 passing probes, got ${report.pass}`);
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
  const dataset = JSON.parse(
    await readFile(new URL("../eval/dataset.json", import.meta.url), "utf-8"),
  ) as EvalDataset;
  const { offline, llm } = await compare(dataset, { runner: new StubLLMRunner() }); // forced stub → deterministic, no network
  // the stub LLM should pass everything (incl. the llmOnly probes offline skips)
  assert.equal(llm.fail, 0, "LLM(stub) should have no failures");
  assert.ok(llm.pass > offline.pass, `LLM should unlock more probes (llm ${llm.pass} > offline ${offline.pass})`);
  // offline skips exactly the llmOnly probes
  assert.equal(offline.skip, dataset.probes.filter((p) => p.llmOnly).length);
});

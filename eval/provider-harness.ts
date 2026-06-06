import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { getCalibration } from "../src/memory/calibration.js";
import { buildOpenMap, type OpenMap } from "../src/openmap.js";
import { loadConfig, type Config } from "../src/core/config.js";
import { type EvalDataset, type EvalProbe } from "./harness.js";

export type ProviderProbeStatus = "pass" | "fail" | "skip" | "unsupported" | "error";
export type ProviderRunStatus = "ok" | "unavailable" | "error";

export interface ProviderHit {
  text: string;
  score?: number;
  id?: string;
  metadata?: Record<string, unknown>;
}

export interface ProviderProbeResult {
  category: string;
  desc: string;
  status: ProviderProbeStatus;
  got: string;
}

export interface ProviderEvalReport {
  provider: string;
  status: ProviderRunStatus;
  version?: string;
  reason?: string;
  capabilities: string[];
  results: ProviderProbeResult[];
  pass: number;
  fail: number;
  skip: number;
  unsupported: number;
  error: number;
  total: number;
  supported: number;
  passRate: number;
}

export interface ProviderCompareRun {
  dataset: string;
  description: string;
  user: string;
  generatedAt: string;
  model: ProviderModelSummary;
  providers: ProviderEvalReport[];
}

export interface ProviderModelSummary {
  chatModel: string;
  embeddingModel: string;
  embeddingDimensions: number;
  baseUrl: string | null;
  hasApiKey: boolean;
}

export interface ProviderAdapter {
  name: string;
  version?: string;
  capabilities: string[];
  ingest(dataset: EvalDataset): Promise<void>;
  close?(): Promise<void> | void;
  recall?(query: string, limit: number): Promise<ProviderHit[]>;
  ask?(question: string): Promise<{ likely: boolean; confidence?: number }>;
  beliefs?(predicate: string): Promise<Array<{ object: string; confidence?: number; source?: string }>>;
  calibration?(term: string, context?: string): Promise<{ value: number | null; unit?: string }>;
  memories?(place: string): Promise<Array<{ relationship?: string; text: string }>>;
  conversation?(query: string, limit: number): Promise<ProviderHit[]>;
  persona?(query: string, limit: number): Promise<{ system: string; prepend: string }>;
  graph?(): Promise<string>;
  citations?(query: string, place: string, limit: number): Promise<ProviderHit[]>;
  scenarios?(opts: { limit: number; intent?: string }): Promise<Array<{ title: string; summary: string; places: string[]; intents: string[]; concepts: string[]; turnCount: number }>>;
  routines?(opts: { limit: number; minScenarios?: number; intent?: string; concept?: string }): Promise<Array<{ title: string; summary: string; places: string[]; positivePlaces: string[]; negativePlaces: string[]; intents: string[]; concepts: string[]; support: number; confidence: number }>>;
}

export interface ProviderEvalOptions {
  llmAvailable?: boolean;
}

const lc = (s: string) => s.toLowerCase();
const arr = (x: unknown): string[] => (Array.isArray(x) ? x.map(String) : x == null ? [] : [String(x)]);

export async function runProviderEval(dataset: EvalDataset, provider: ProviderAdapter, opts: ProviderEvalOptions = {}): Promise<ProviderEvalReport> {
  const results: ProviderProbeResult[] = [];
  try {
    await provider.ingest(dataset);
    for (const p of dataset.probes) {
      if (p.llmOnly && !opts.llmAvailable) {
        results.push({ category: p.category, desc: describe(p), status: "skip", got: "needs LLM" });
        continue;
      }
      results.push(await runProbe(provider, p));
    }
  } catch (err) {
    return emptyReport(provider.name, {
      status: "error",
      version: provider.version,
      capabilities: provider.capabilities,
      reason: err instanceof Error ? err.message : String(err),
    });
  } finally {
    await provider.close?.();
  }
  return summarizeReport(provider.name, provider.capabilities, results, provider.version);
}

export function unavailableProviderReport(provider: string, reason: string): ProviderEvalReport {
  return emptyReport(provider, { status: "unavailable", reason });
}

export function providerModelSummary(config: Config): ProviderModelSummary {
  return {
    chatModel: config.openaiChatModel,
    embeddingModel: config.openaiEmbedModel,
    embeddingDimensions: embeddingDimensions(config),
    baseUrl: config.openaiBaseUrl,
    hasApiKey: Boolean(config.openaiApiKey),
  };
}

export async function runCommandProviderEval(provider: string, command: string, dataset: EvalDataset, config: Config = loadConfig()): Promise<ProviderEvalReport> {
  const payload = JSON.stringify({
    schema: "openmap-provider-eval/v1",
    provider,
    model: providerModelSummary(config),
    dataset,
  });
  try {
    const stdout = await runCommand(command, payload, providerEnv(config));
    const parsed = JSON.parse(stdout) as Partial<ProviderEvalReport>;
    const results = Array.isArray(parsed.results) ? parsed.results : [];
    return summarizeReport(
      parsed.provider ?? provider,
      Array.isArray(parsed.capabilities) ? parsed.capabilities : ["external-command"],
      results as ProviderProbeResult[],
      parsed.version,
    );
  } catch (err) {
    return emptyReport(provider, {
      status: "error",
      reason: err instanceof Error ? err.message : String(err),
      capabilities: ["external-command"],
    });
  }
}

export function createOpenMapProvider(config?: Config, opts: { llmAvailable?: boolean } = {}): ProviderAdapter {
  const cfg: Config = { ...(config ?? loadConfig()), dbPath: ":memory:" };
  const llmAvailable = opts.llmAvailable ?? Boolean(cfg.openaiApiKey);
  if (!llmAvailable) {
    cfg.openaiApiKey = null;
    cfg.embedder = "none";
    cfg.tagger = "lexicon";
  }
  const mem = buildOpenMap(cfg);
  return new OpenMapProvider(mem);
}

export async function createMem0Provider(dataset: EvalDataset, config: Config = loadConfig()): Promise<ProviderAdapter | ProviderEvalReport> {
  const spec = "mem0ai/oss";
  let mod: any;
  try {
    mod = await import(spec);
  } catch {
    return unavailableProviderReport("mem0", "install optional package first: npm install mem0ai");
  }

  const apiKey = config.openaiApiKey;
  if (!apiKey) return unavailableProviderReport("mem0", "set OPENAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY in .env.local");

  const baseURL = process.env.MEM0_OPENAI_BASE_URL ?? config.openaiBaseUrl ?? undefined;
  const embedModel = process.env.MEM0_EMBED_MODEL ?? config.openaiEmbedModel;
  const chatModel = process.env.MEM0_CHAT_MODEL ?? config.openaiChatModel;
  const dimension = Number(process.env.MEM0_EMBED_DIMS ?? process.env.OPENMAP_OPENAI_EMBED_DIMS ?? embeddingDimensions(config));
  const Memory = mod.Memory;
  if (!Memory?.fromConfig) return unavailableProviderReport("mem0", "mem0ai/oss does not expose Memory.fromConfig");

  const memory = Memory.fromConfig({
    embedder: {
      provider: "openai",
      config: { apiKey, baseURL, model: embedModel, embeddingDims: dimension },
    },
    vectorStore: {
      provider: "memory",
      config: { collectionName: `openmap_eval_${dataset.name.replace(/[^a-z0-9_]+/gi, "_")}_${Date.now()}`, dimension },
    },
    llm: {
      provider: "openai",
      config: { apiKey, baseURL, model: chatModel },
    },
    disableHistory: true,
    customInstructions:
      "Extract durable personal memories from map-assistant conversations. Preserve place names, preferences, dislikes, companions, and intent words.",
  });
  return new Mem0Provider(memory, dataset.user);
}

export async function createGbrainCliProvider(config: Config = loadConfig()): Promise<ProviderAdapter | ProviderEvalReport> {
  const available = await commandOk("gbrain", ["--version"]);
  if (!available) return unavailableProviderReport("gbrain", "gbrain CLI not found on PATH");
  return new GbrainCliProvider(config);
}

class OpenMapProvider implements ProviderAdapter {
  name = "openmap";
  capabilities = ["recall", "ask", "belief", "calibration", "memory", "conversation", "persona", "graph", "citation", "scenario", "routine"];
  private userId = "default";
  constructor(private readonly mem: OpenMap) {}

  async ingest(dataset: EvalDataset): Promise<void> {
    this.userId = dataset.user;
    for (const s of dataset.sessions) await this.mem.capture(s.turns, { userId: dataset.user });
    this.mem.consolidate(dataset.user);
  }

  async recall(query: string, limit: number): Promise<ProviderHit[]> {
    const rows = await this.mem.recall(query, null, limit, this.userId);
    return rows.map((r) => ({ id: r.place.id, text: r.place.name, score: r.score, metadata: { reasons: r.reasons } }));
  }

  async ask(question: string): Promise<{ likely: boolean; confidence?: number }> {
    const res = this.mem.ask(question, this.userId);
    return { likely: res.likely, confidence: res.confidence };
  }

  async beliefs(predicate: string): Promise<Array<{ object: string; confidence?: number; source?: string }>> {
    return this.mem.beliefs(this.userId, { predicate: predicate as any }).map((b) => ({ object: b.object, confidence: b.confidence, source: b.source }));
  }

  async calibration(term: string, context?: string): Promise<{ value: number | null; unit?: string }> {
    const c = getCalibration(this.mem.db, this.userId, term, context);
    return { value: c.value, unit: c.unit };
  }

  async memories(place: string): Promise<Array<{ relationship?: string; text: string }>> {
    return this.mem.listMemories(this.userId, { limit: 1000 })
      .filter((it) => it.place && lc(it.place.name) === lc(place))
      .map((it) => ({ relationship: it.memory.relationship, text: `${it.place?.name}: ${it.memory.note}` }));
  }

  async conversation(query: string, limit: number): Promise<ProviderHit[]> {
    return this.mem.searchConversation(query, { userId: this.userId, limit }).map((t) => ({ id: String(t.id), text: t.content }));
  }

  async persona(query: string, limit: number): Promise<{ system: string; prepend: string }> {
    const ctx = await this.mem.recallContext(query, { userId: this.userId, limit });
    return { system: ctx.system, prepend: ctx.prepend };
  }

  async graph(): Promise<string> {
    return this.mem.graphMermaid(this.userId);
  }

  async citations(query: string, place: string, limit: number): Promise<ProviderHit[]> {
    const ctx = await this.mem.recallContext(query, { userId: this.userId, limit });
    const hit = ctx.places.find((r) => lc(r.place.name).includes(lc(place)));
    if (!hit) return [];
    return (ctx.sources[hit.place.id] ?? []).map((s) => ({ id: String(s.turnId), text: s.snippet }));
  }

  async scenarios(opts: { limit: number; intent?: string }): Promise<Array<{ title: string; summary: string; places: string[]; intents: string[]; concepts: string[]; turnCount: number }>> {
    const places = this.mem.listPlaces(this.userId, { limit: 1000 });
    const placeName = (id: string) => places.find((pl) => pl.id === id)?.name ?? id;
    return this.mem.scenarios(this.userId, opts).map((s) => ({
      title: s.title,
      summary: s.summary,
      places: s.placeIds.map(placeName),
      intents: s.intents,
      concepts: s.concepts,
      turnCount: s.turnIds.length,
    }));
  }

  async routines(opts: { limit: number; minScenarios?: number; intent?: string; concept?: string }): Promise<Array<{ title: string; summary: string; places: string[]; positivePlaces: string[]; negativePlaces: string[]; intents: string[]; concepts: string[]; support: number; confidence: number }>> {
    const places = this.mem.listPlaces(this.userId, { limit: 1000 });
    const placeName = (id: string) => places.find((pl) => pl.id === id)?.name ?? id;
    return this.mem.routines(this.userId, opts).map((r) => ({
      title: r.title,
      summary: r.summary,
      places: r.placeIds.map(placeName),
      positivePlaces: r.positivePlaceIds.map(placeName),
      negativePlaces: r.negativePlaceIds.map(placeName),
      intents: r.intents,
      concepts: r.concepts,
      support: r.support,
      confidence: r.confidence,
    }));
  }
}

class Mem0Provider implements ProviderAdapter {
  name = "mem0";
  capabilities = ["recall"];
  constructor(private readonly memory: any, private readonly userId: string) {}

  async ingest(dataset: EvalDataset): Promise<void> {
    let sessionIndex = 0;
    for (const s of dataset.sessions) {
      sessionIndex += 1;
      const messages = s.turns.map((t) => ({ role: t.role ?? "user", content: t.content }));
      await this.memory.add(messages, {
        userId: this.userId,
        infer: true,
        metadata: { dataset: dataset.name, session: s.title, sessionIndex },
      });
    }
  }

  async recall(query: string, limit: number): Promise<ProviderHit[]> {
    const res = await this.memory.search(query, { topK: limit, filters: { user_id: this.userId } });
    const rows = Array.isArray(res?.results) ? res.results : [];
    return rows.map((r: any) => ({
      id: String(r.id ?? ""),
      text: String(r.memory ?? r.content ?? ""),
      score: typeof r.score === "number" ? r.score : undefined,
      metadata: typeof r.metadata === "object" && r.metadata ? r.metadata : undefined,
    }));
  }

  async close(): Promise<void> {
    await this.memory.reset?.();
  }
}

class GbrainCliProvider implements ProviderAdapter {
  name = "gbrain";
  capabilities = ["recall", "conversation", "citation"];
  private readonly runId = `openmap-eval-${Date.now()}-${randomUUID().slice(0, 8)}`;
  private readonly slugs: string[] = [];
  private tmpDir: string | null = null;
  constructor(private readonly config: Config) {}

  async ingest(dataset: EvalDataset): Promise<void> {
    this.tmpDir = await mkdtemp(join(tmpdir(), "openmap-gbrain-eval-"));
    let idx = 0;
    for (const session of dataset.sessions) {
      idx += 1;
      const slug = `${this.runId}-${idx}`;
      this.slugs.push(slug);
      const body = [
        `# ${dataset.name}: ${session.title}`,
        "",
        `user: ${dataset.user}`,
        `session: ${session.title}`,
        "",
        ...session.turns.map((t, i) => `turn ${i + 1} ${t.role ?? "user"}: ${t.content}`),
      ].join("\n");
      await writeFile(join(this.tmpDir, `${slug}.md`), body + "\n", "utf-8");
    }
    await runCommand(`gbrain import ${shellQuote(this.tmpDir)} --no-embed --workers 1 --json`, "", providerEnv(this.config), 60_000);
  }

  async recall(query: string, limit: number): Promise<ProviderHit[]> {
    return this.search(query, limit);
  }

  async conversation(query: string, limit: number): Promise<ProviderHit[]> {
    return this.search(query, limit);
  }

  async citations(query: string, place: string, limit: number): Promise<ProviderHit[]> {
    const hits = await this.search(`${query} ${place}`, limit);
    return hits.filter((h) => h.text.toLowerCase().includes(place.toLowerCase()));
  }

  async close(): Promise<void> {
    for (const slug of this.slugs) {
      try {
        await runCommand(`gbrain delete ${shellQuote(slug)}`, "", providerEnv(this.config), 10_000);
      } catch {
        // Best-effort cleanup only.
      }
    }
    if (this.tmpDir) await rm(this.tmpDir, { recursive: true, force: true });
  }

  private async search(query: string, limit: number): Promise<ProviderHit[]> {
    const stdout = await runCommand(`gbrain search ${shellQuote(query)} --limit ${Number(limit) || 5}`, "", providerEnv(this.config), 15_000);
    return stdout.split("\n").map(parseGbrainSearchLine).filter((x): x is ProviderHit => x != null);
  }
}

function parseGbrainSearchLine(line: string): ProviderHit | null {
  const m = line.match(/^\[([0-9.]+)\]\s+(\S+)\s+--\s+([\s\S]*)$/);
  if (!m) return null;
  return { score: Number(m[1]), id: m[2], text: m[3] };
}

function commandOk(command: string, args: string[]): Promise<boolean> {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: "ignore" });
    child.on("error", () => resolve(false));
    child.on("close", (code) => resolve(code === 0));
  });
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

async function runProbe(provider: ProviderAdapter, p: EvalProbe): Promise<ProviderProbeResult> {
  try {
    switch (p.kind) {
      case "recall": {
        if (!provider.recall) return unsupported(p, "recall");
        const res = await provider.recall(String(p.query), 5);
        const texts = res.map((r) => r.text);
        const top = texts[0] ?? "";
        let pass = p.expectTop == null || lc(top).includes(lc(String(p.expectTop)));
        if (p.expectIncludes != null) pass = pass && texts.some((text) => lc(text).includes(lc(String(p.expectIncludes))));
        if (p.expectNotTop != null) pass = pass && !lc(top).includes(lc(String(p.expectNotTop)));
        return result(p, pass, texts.join(" > ") || "(none)");
      }
      case "ask": {
        if (!provider.ask) return unsupported(p, "ask");
        const a = await provider.ask(String(p.question));
        return result(p, a.likely === Boolean(p.expectLikely), `likely=${a.likely} conf=${a.confidence ?? "-"}`);
      }
      case "belief": {
        if (!provider.beliefs) return unsupported(p, "belief");
        const beliefs = await provider.beliefs(String(p.predicate));
        const has = beliefs.some((b) => lc(b.object) === lc(String(p.object)));
        return result(p, has === (p.expectPresent == null ? true : Boolean(p.expectPresent)), has ? "present" : "absent");
      }
      case "calibration": {
        if (!provider.calibration) return unsupported(p, "calibration");
        const c = await provider.calibration(String(p.term), p.context ? String(p.context) : undefined);
        let pass = false;
        if (p.expect != null) pass = c.value === Number(p.expect);
        else if (p.min != null || p.max != null)
          pass = c.value != null && (p.min == null || c.value >= Number(p.min)) && (p.max == null || c.value <= Number(p.max));
        else pass = c.value != null;
        return result(p, pass, `${c.value} ${c.unit ?? ""}`.trim());
      }
      case "reconcile":
      case "memory": {
        if (!provider.memories) return unsupported(p, "memory");
        const items = await provider.memories(String(p.place));
        const rel = items[0]?.relationship;
        const pass = (p.expectCount == null || items.length === Number(p.expectCount)) && rel === p.expectRelationship;
        return result(p, pass, `count=${items.length} rel=${rel ?? "-"}`);
      }
      case "conversation": {
        if (!provider.conversation) return unsupported(p, "conversation");
        const hits = await provider.conversation(String(p.query), Number(p.limit ?? 10));
        return result(p, hits.some((h) => lc(h.text).includes(lc(String(p.expectIncludes)))), hits.map((h) => h.text).join(" | ") || "(none)");
      }
      case "persona": {
        if (!provider.persona) return unsupported(p, "persona");
        const ctx = await provider.persona(String(p.query ?? ""), Number(p.limit ?? 5));
        const system = Array.isArray(p.expectSystemIncludes) ? p.expectSystemIncludes.map(String) : [];
        const prepend = Array.isArray(p.expectPrependIncludes) ? p.expectPrependIncludes.map(String) : [];
        const excludes = Array.isArray(p.expectSystemExcludes) ? p.expectSystemExcludes.map(String) : [];
        const pass = system.every((s) => lc(ctx.system).includes(lc(s))) && prepend.every((s) => lc(ctx.prepend).includes(lc(s))) && excludes.every((s) => !lc(ctx.system).includes(lc(s)));
        return result(p, pass, `system=${ctx.system.replace(/\s+/g, " ").slice(0, 160)} prepend=${ctx.prepend.replace(/\s+/g, " ").slice(0, 160)}`);
      }
      case "graph": {
        if (!provider.graph) return unsupported(p, "graph");
        const graph = await provider.graph();
        const includes = Array.isArray(p.expectIncludes) ? p.expectIncludes.map(String) : [String(p.expectIncludes)];
        return result(p, includes.every((s) => lc(graph).includes(lc(s))), graph.replace(/\s+/g, " ").slice(0, 220));
      }
      case "citation": {
        if (!provider.citations) return unsupported(p, "citation");
        const sources = await provider.citations(String(p.query ?? ""), String(p.place), Number(p.limit ?? 5));
        return result(p, sources.some((s) => lc(s.text).includes(lc(String(p.expectIncludes)))), sources.map((s) => s.text).join(" | ") || "(none)");
      }
      case "scenario": {
        if (!provider.scenarios) return unsupported(p, "scenario");
        const scenarios = await provider.scenarios({ limit: Number(p.limit ?? 20), intent: p.intent ? String(p.intent) : undefined });
        const found = scenarios.find((s) =>
          (p.expectPlace == null || s.places.some((n) => lc(n).includes(lc(String(p.expectPlace))))) &&
          (p.expectOtherPlace == null || s.places.some((n) => lc(n).includes(lc(String(p.expectOtherPlace))))) &&
          (p.expectIntent == null || s.intents.includes(String(p.expectIntent))) &&
          (p.expectNotIntent == null || !s.intents.includes(String(p.expectNotIntent))) &&
          (p.expectConcept == null || s.concepts.includes(String(p.expectConcept))) &&
          (p.expectNotConcept == null || !s.concepts.includes(String(p.expectNotConcept))) &&
          arr(p.expectConceptsInclude).every((c) => s.concepts.includes(c)) &&
          arr(p.expectConceptsExclude).every((c) => !s.concepts.includes(c)) &&
          arr(p.expectTitleIncludes).every((t) => lc(s.title).includes(lc(t))) &&
          arr(p.expectTitleExcludes).every((t) => !lc(s.title).includes(lc(t))) &&
          arr(p.expectSummaryIncludes).every((t) => lc(s.summary).includes(lc(t))) &&
          arr(p.expectSummaryExcludes).every((t) => !lc(s.summary).includes(lc(t))) &&
          (p.minTurns == null || s.turnCount >= Number(p.minTurns))
        );
        return result(p, !!found, found ? `${found.title} places=${found.places.join(",")} intents=${found.intents.join(",")} concepts=${found.concepts.join(",")} turns=${found.turnCount}` : scenarios.map((s) => `${s.title}:${s.places.join(",")}`).join(" | ") || "(none)");
      }
      case "routine": {
        if (!provider.routines) return unsupported(p, "routine");
        const routines = await provider.routines({
          limit: Number(p.limit ?? 20),
          minScenarios: p.minScenarios == null ? undefined : Number(p.minScenarios),
          intent: p.intent ? String(p.intent) : undefined,
          concept: p.concept ? String(p.concept) : undefined,
        });
        const found = routines.find((r) =>
          (p.expectPlace == null || r.places.some((n) => lc(n).includes(lc(String(p.expectPlace))))) &&
          (p.expectOtherPlace == null || r.places.some((n) => lc(n).includes(lc(String(p.expectOtherPlace))))) &&
          (p.expectPositivePlace == null || r.positivePlaces.some((n) => lc(n).includes(lc(String(p.expectPositivePlace))))) &&
          (p.expectNegativePlace == null || r.negativePlaces.some((n) => lc(n).includes(lc(String(p.expectNegativePlace))))) &&
          arr(p.expectIntentsInclude).every((i) => r.intents.includes(i)) &&
          arr(p.expectIntentsExclude).every((i) => !r.intents.includes(i)) &&
          arr(p.expectConceptsInclude).every((c) => r.concepts.includes(c)) &&
          arr(p.expectConceptsExclude).every((c) => !r.concepts.includes(c)) &&
          arr(p.expectTitleIncludes).every((t) => lc(r.title).includes(lc(t))) &&
          arr(p.expectTitleExcludes).every((t) => !lc(r.title).includes(lc(t))) &&
          arr(p.expectSummaryIncludes).every((t) => lc(r.summary).includes(lc(t))) &&
          arr(p.expectSummaryExcludes).every((t) => !lc(r.summary).includes(lc(t))) &&
          (p.minScenarios == null || r.support >= Number(p.minScenarios)) &&
          (p.minConfidence == null || r.confidence >= Number(p.minConfidence))
        );
        return result(p, !!found, found ? `${found.title} support=${found.support} intents=${found.intents.join(",")} concepts=${found.concepts.join(",")} preferred=${found.positivePlaces.join(",")} avoided=${found.negativePlaces.join(",")}` : routines.map((r) => `${r.title}:support=${r.support}:${r.summary}`).join(" | ") || "(none)");
      }
    }
  } catch (err) {
    return { category: p.category, desc: describe(p), status: "error", got: err instanceof Error ? err.message : String(err) };
  }
}

function result(p: EvalProbe, pass: boolean, got: string): ProviderProbeResult {
  return { category: p.category, desc: describe(p), status: pass ? "pass" : "fail", got };
}

function unsupported(p: EvalProbe, capability: string): ProviderProbeResult {
  return { category: p.category, desc: describe(p), status: "unsupported", got: `${capability} not supported by provider` };
}

function summarizeReport(provider: string, capabilities: string[], results: ProviderProbeResult[], version?: string): ProviderEvalReport {
  const pass = results.filter((r) => r.status === "pass").length;
  const fail = results.filter((r) => r.status === "fail").length;
  const skip = results.filter((r) => r.status === "skip").length;
  const unsupported = results.filter((r) => r.status === "unsupported").length;
  const error = results.filter((r) => r.status === "error").length;
  const supported = Math.max(0, results.length - skip - unsupported);
  const executable = Math.max(1, supported);
  return {
    provider,
    status: error > 0 ? "error" : "ok",
    version,
    capabilities,
    results,
    pass,
    fail,
    skip,
    unsupported,
    error,
    total: results.length,
    supported,
    passRate: Math.round((pass / executable) * 100),
  };
}

function emptyReport(provider: string, opts: { status: ProviderRunStatus; reason?: string; capabilities?: string[]; version?: string }): ProviderEvalReport {
  return {
    provider,
    status: opts.status,
    version: opts.version,
    reason: opts.reason,
    capabilities: opts.capabilities ?? [],
    results: [],
    pass: 0,
    fail: 0,
    skip: 0,
    unsupported: 0,
    error: opts.status === "error" ? 1 : 0,
    total: 0,
    supported: 0,
    passRate: 0,
  };
}

function describe(p: EvalProbe): string {
  switch (p.kind) {
    case "recall": {
      const target = p.expectTop != null ? `top≈${p.expectTop}` : p.expectIncludes != null ? `includes ${p.expectIncludes}` : `not top ${p.expectNotTop}`;
      return `recall "${p.query}" -> ${target}`;
    }
    case "ask": return `ask "${p.question}" -> likely=${p.expectLikely}`;
    case "belief": return `belief user ${p.predicate} ${p.object} ${p.expectPresent === false ? "absent" : "present"}`;
    case "calibration": return `calibration ${p.term}${p.context ? "@" + p.context : ""}`;
    case "reconcile": return `reconcile ${p.place} -> ${p.expectCount}x ${p.expectRelationship}`;
    case "memory": return `memory ${p.place} -> ${p.expectRelationship}`;
    case "conversation": return `conversation "${p.query}" contains ${p.expectIncludes}`;
    case "persona": return `persona context for "${p.query ?? ""}"`;
    case "graph": return `graph contains ${Array.isArray(p.expectIncludes) ? p.expectIncludes.join("+") : p.expectIncludes}`;
    case "citation": return `citation ${p.place} for "${p.query ?? ""}" contains ${p.expectIncludes}`;
    case "scenario": return `scenario ${p.expectPlace ?? ""}/${p.expectOtherPlace ?? ""} intent=${p.expectIntent ?? ""}`;
    case "routine": return `routine ${p.expectPlace ?? ""}/${p.expectOtherPlace ?? ""} concepts=${arr(p.expectConceptsInclude).join("+")}`;
    default: return p.kind;
  }
}

function embeddingDimensions(config: Config): number {
  const explicit = process.env.OPENMAP_OPENAI_EMBED_DIMS;
  if (explicit && Number.isFinite(Number(explicit))) return Number(explicit);
  const model = config.openaiEmbedModel.toLowerCase();
  if (model.includes("gemini")) return 768;
  if (model.includes("3-large")) return 3072;
  if (model.includes("ada-002")) return 1536;
  return 1536;
}

function providerEnv(config: Config): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    OPENMAP_OPENAI_CHAT_MODEL: config.openaiChatModel,
    OPENMAP_OPENAI_EMBED_MODEL: config.openaiEmbedModel,
    OPENMAP_OPENAI_EMBED_DIMS: String(embeddingDimensions(config)),
  };
  if (config.openaiApiKey) env.OPENAI_API_KEY = config.openaiApiKey;
  else {
    delete env.OPENAI_API_KEY;
    delete env.GEMINI_API_KEY;
    delete env.GOOGLE_API_KEY;
  }
  if (config.openaiBaseUrl) env.OPENMAP_OPENAI_BASE_URL = config.openaiBaseUrl;
  else delete env.OPENMAP_OPENAI_BASE_URL;
  return env;
}

function runCommand(command: string, stdin: string, env: NodeJS.ProcessEnv = process.env, timeoutMs = 120_000): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, { shell: true, stdio: ["pipe", "pipe", "pipe"], env });
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`command timed out after ${timeoutMs}ms: ${command}`));
    }, timeoutMs);
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf-8");
    child.stderr.setEncoding("utf-8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(stdout.trim());
      else reject(new Error(`command exited ${code}: ${stderr || stdout}`.trim()));
    });
    child.stdin.end(stdin);
  });
}

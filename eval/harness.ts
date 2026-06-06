import { type OpenMap } from "../src/openmap.js";
import { getCalibration } from "../src/memory/calibration.js";

export interface EvalProbe {
  kind: "recall" | "ask" | "belief" | "calibration" | "reconcile" | "memory" | "conversation" | "persona" | "graph" | "citation" | "scenario" | "routine";
  category: string;
  llmOnly?: boolean;
  [k: string]: unknown;
}
export interface EvalDataset {
  name: string;
  description: string;
  user: string;
  sessions: Array<{ title: string; turns: Array<{ role?: string; content: string }> }>;
  probes: EvalProbe[];
}
export interface ProbeResult {
  category: string;
  desc: string;
  status: "pass" | "fail" | "skip";
  got: string;
}
export interface EvalReport {
  results: ProbeResult[];
  pass: number;
  fail: number;
  skip: number;
  total: number;
}

const lc = (s: string) => s.toLowerCase();
const arr = (x: unknown): string[] => (Array.isArray(x) ? x.map(String) : x == null ? [] : [String(x)]);

/** Ingest the dataset's sessions into `mem` (one capture() per session), then
 * consolidate beliefs. Conversation is the only input; capture records raw L0
 * turns for grounding and extracts structured map memory from user turns. */
export async function ingest(dataset: EvalDataset, mem: OpenMap): Promise<void> {
  for (const s of dataset.sessions) await mem.capture(s.turns, { userId: dataset.user });
  mem.consolidate(dataset.user);
}

export async function runEval(dataset: EvalDataset, mem: OpenMap, opts: { llmAvailable?: boolean } = {}): Promise<EvalReport> {
  await ingest(dataset, mem);
  const u = dataset.user;
  const results: ProbeResult[] = [];

  for (const p of dataset.probes) {
    if (p.llmOnly && !opts.llmAvailable) {
      results.push({ category: p.category, desc: describe(p), status: "skip", got: "needs LLM" });
      continue;
    }
    let pass = false;
    let got = "";
    switch (p.kind) {
      case "recall": {
        const res = await mem.recall(String(p.query), null, 5, u);
        const names = res.map((r) => r.place.name);
        got = names.join(" > ") || "(none)";
        pass = p.expectTop == null || lc(names[0] ?? "").includes(lc(String(p.expectTop)));
        if (p.expectIncludes != null)
          pass = pass && names.some((name) => lc(name).includes(lc(String(p.expectIncludes))));
        if (p.expectNotTop != null)
          pass = pass && !lc(names[0] ?? "").includes(lc(String(p.expectNotTop)));
        break;
      }
      case "ask": {
        const a = mem.ask(String(p.question), u);
        got = `likely=${a.likely} conf=${a.confidence}`;
        pass = a.likely === Boolean(p.expectLikely);
        break;
      }
      case "belief": {
        const has = mem.beliefs(u, { predicate: p.predicate as any }).some((b) => lc(b.object) === lc(String(p.object)));
        got = has ? "present" : "absent";
        pass = has === (p.expectPresent == null ? true : Boolean(p.expectPresent));
        break;
      }
      case "calibration": {
        const c = getCalibration(mem.db, u, String(p.term), p.context ? String(p.context) : undefined);
        got = `${c.value} ${c.unit}`;
        if (p.expect != null) pass = c.value === Number(p.expect);
        else if (p.min != null || p.max != null)
          pass =
            c.value != null &&
            (p.min == null || c.value >= Number(p.min)) &&
            (p.max == null || c.value <= Number(p.max));
        else pass = c.value != null;
        break;
      }
      case "reconcile":
      case "memory": {
        const items = mem.listMemories(u, { limit: 1000 }).filter((it) => it.place && lc(it.place.name) === lc(String(p.place)));
        const rel = items[0]?.memory.relationship;
        got = `count=${items.length} rel=${rel ?? "-"}`;
        pass = (p.expectCount == null || items.length === Number(p.expectCount)) && rel === p.expectRelationship;
        break;
      }
      case "conversation": {
        const hits = mem.searchConversation(String(p.query), { userId: u, limit: Number(p.limit ?? 10) });
        got = hits.map((h) => h.content).join(" | ") || "(none)";
        pass = hits.some((h) => lc(h.content).includes(lc(String(p.expectIncludes))));
        break;
      }
      case "persona": {
        const ctx = await mem.recallContext(String(p.query ?? ""), { userId: u, limit: Number(p.limit ?? 5) });
        got = `system=${ctx.system.replace(/\s+/g, " ").slice(0, 160)} prepend=${ctx.prepend.replace(/\s+/g, " ").slice(0, 160)}`;
        const system = Array.isArray(p.expectSystemIncludes) ? p.expectSystemIncludes.map(String) : [];
        const prepend = Array.isArray(p.expectPrependIncludes) ? p.expectPrependIncludes.map(String) : [];
        const excludes = Array.isArray(p.expectSystemExcludes) ? p.expectSystemExcludes.map(String) : [];
        pass =
          system.every((s) => lc(ctx.system).includes(lc(s))) &&
          prepend.every((s) => lc(ctx.prepend).includes(lc(s))) &&
          excludes.every((s) => !lc(ctx.system).includes(lc(s)));
        break;
      }
      case "graph": {
        const mermaid = mem.graphMermaid(u);
        got = mermaid.replace(/\s+/g, " ").slice(0, 220);
        const includes = Array.isArray(p.expectIncludes) ? p.expectIncludes.map(String) : [String(p.expectIncludes)];
        pass = includes.every((s) => lc(mermaid).includes(lc(s)));
        break;
      }
      case "citation": {
        const ctx = await mem.recallContext(String(p.query ?? ""), { userId: u, limit: Number(p.limit ?? 5) });
        const place = ctx.places.find((r) => lc(r.place.name).includes(lc(String(p.place))));
        const sources = place ? ctx.sources[place.place.id] ?? [] : [];
        got = sources.map((s) => `turn#${s.turnId}:${s.snippet}`).join(" | ") || "(none)";
        pass =
          !!place &&
          sources.some((s) => lc(s.snippet).includes(lc(String(p.expectIncludes)))) &&
          sources.some((s) => ctx.prepend.includes(`turn#${s.turnId}`));
        break;
      }
      case "scenario": {
        const scenarios = mem.scenarios(u, { limit: Number(p.limit ?? 20), intent: p.intent ? String(p.intent) : undefined });
        const places = mem.listPlaces(u, { limit: 1000 });
        const placeName = (id: string) => places.find((pl) => pl.id === id)?.name ?? id;
        const found = scenarios.find((s) => {
          const names = s.placeIds.map(placeName).map(lc);
          return (
            (p.expectPlace == null || names.some((n) => n.includes(lc(String(p.expectPlace))))) &&
            (p.expectOtherPlace == null || names.some((n) => n.includes(lc(String(p.expectOtherPlace))))) &&
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
            (p.minTurns == null || s.turnIds.length >= Number(p.minTurns))
          );
        });
        got = found
          ? `${found.title} places=${found.placeIds.map(placeName).join(",")} intents=${found.intents.join(",")} concepts=${found.concepts.join(",")} turns=${found.turnIds.length}`
          : scenarios.map((s) => `${s.title}:${s.placeIds.map(placeName).join(",")}`).join(" | ") || "(none)";
        pass = !!found;
        break;
      }
      case "routine": {
        const routines = mem.routines(u, {
          limit: Number(p.limit ?? 20),
          minScenarios: p.minScenarios == null ? undefined : Number(p.minScenarios),
          intent: p.intent ? String(p.intent) : undefined,
          concept: p.concept ? String(p.concept) : undefined,
        });
        const places = mem.listPlaces(u, { limit: 1000 });
        const placeName = (id: string) => places.find((pl) => pl.id === id)?.name ?? id;
        const found = routines.find((r) => {
          const allNames = r.placeIds.map(placeName).map(lc);
          const positiveNames = r.positivePlaceIds.map(placeName).map(lc);
          const negativeNames = r.negativePlaceIds.map(placeName).map(lc);
          return (
            (p.expectPlace == null || allNames.some((n) => n.includes(lc(String(p.expectPlace))))) &&
            (p.expectOtherPlace == null || allNames.some((n) => n.includes(lc(String(p.expectOtherPlace))))) &&
            (p.expectPositivePlace == null || positiveNames.some((n) => n.includes(lc(String(p.expectPositivePlace))))) &&
            (p.expectNegativePlace == null || negativeNames.some((n) => n.includes(lc(String(p.expectNegativePlace))))) &&
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
        });
        got = found
          ? `${found.title} support=${found.support} intents=${found.intents.join(",")} concepts=${found.concepts.join(",")} preferred=${found.positivePlaceIds.map(placeName).join(",")} avoided=${found.negativePlaceIds.map(placeName).join(",")}`
          : routines.map((r) => `${r.title}:support=${r.support}:${r.summary}`).join(" | ") || "(none)";
        pass = !!found;
        break;
      }
    }
    results.push({ category: p.category, desc: describe(p), status: pass ? "pass" : "fail", got });
  }

  return {
    results,
    pass: results.filter((r) => r.status === "pass").length,
    fail: results.filter((r) => r.status === "fail").length,
    skip: results.filter((r) => r.status === "skip").length,
    total: results.length,
  };
}

function describe(p: EvalProbe): string {
  switch (p.kind) {
    case "recall": {
      const target = p.expectTop != null ? `top≈${p.expectTop}` : p.expectIncludes != null ? `includes ${p.expectIncludes}` : `not top ${p.expectNotTop}`;
      return `recall "${p.query}" → ${target}`;
    }
    case "ask": return `ask "${p.question}" → likely=${p.expectLikely}`;
    case "belief": return `belief user ${p.predicate} ${p.object} ${p.expectPresent === false ? "absent" : "present"}`;
    case "calibration": {
      const bound = p.expect != null
        ? "=" + p.expect
        : [p.min != null ? "≥" + p.min : "", p.max != null ? "≤" + p.max : ""].filter(Boolean).join(" ");
      return `calibration ${p.term}${p.context ? "@" + p.context : ""} ${bound}`;
    }
    case "reconcile": return `reconcile ${p.place} → ${p.expectCount}× ${p.expectRelationship}`;
    case "memory": return `memory ${p.place} → ${p.expectRelationship}`;
    case "conversation": return `conversation "${p.query}" contains ${p.expectIncludes}`;
    case "persona": return `persona context for "${p.query ?? ""}"`;
    case "graph": return `graph contains ${Array.isArray(p.expectIncludes) ? p.expectIncludes.join("+") : p.expectIncludes}`;
    case "citation": return `citation ${p.place} for "${p.query ?? ""}" contains ${p.expectIncludes}`;
    case "scenario": return `scenario ${p.expectPlace ?? ""}/${p.expectOtherPlace ?? ""} intent=${p.expectIntent ?? ""}`;
    case "routine": return `routine ${p.expectPlace ?? ""}/${p.expectOtherPlace ?? ""} concepts=${arr(p.expectConceptsInclude).join("+")}`;
    default: return p.kind;
  }
}

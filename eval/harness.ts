import { type OpenMap } from "../src/openmap.js";
import { getCalibration } from "../src/memory/calibration.js";

export interface EvalProbe {
  kind: "recall" | "ask" | "belief" | "calibration" | "reconcile" | "memory";
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

/** Ingest the dataset's sessions into `mem` (one observe() per session), then
 * consolidate beliefs. Conversation is the only input. */
export async function ingest(dataset: EvalDataset, mem: OpenMap): Promise<void> {
  for (const s of dataset.sessions) await mem.observe(s.turns, { userId: dataset.user });
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
        got = res[0]?.place.name ?? "(none)";
        pass = lc(got).includes(lc(String(p.expectTop)));
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
        pass = has;
        break;
      }
      case "calibration": {
        const c = getCalibration(mem.db, u, String(p.term), p.context ? String(p.context) : undefined);
        got = `${c.value} ${c.unit}`;
        if (p.expect != null) pass = c.value === Number(p.expect);
        else if (p.min != null) pass = c.value != null && c.value >= Number(p.min);
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
    case "recall": return `recall "${p.query}" → top≈${p.expectTop}`;
    case "ask": return `ask "${p.question}" → likely=${p.expectLikely}`;
    case "belief": return `belief user ${p.predicate} ${p.object}`;
    case "calibration": return `calibration ${p.term}${p.context ? "@" + p.context : ""} ${p.expect != null ? "=" + p.expect : "≥" + p.min}`;
    case "reconcile": return `reconcile ${p.place} → ${p.expectCount}× ${p.expectRelationship}`;
    case "memory": return `memory ${p.place} → ${p.expectRelationship}`;
    default: return p.kind;
  }
}

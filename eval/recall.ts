#!/usr/bin/env node
process.removeAllListeners("warning");
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && /SQLite/i.test(w.message)) return;
  console.error(w.stack ?? w.message);
});

import { buildOpenMap } from "../src/openmap.js";
import { loadConfig } from "../src/core/config.js";
import { type RawPlace } from "../src/core/types.js";

// Places with descriptive tags; queries are PARAPHRASES that deliberately share
// no trigger words with the tags — so recall is driven by embedding semantics,
// not lexical/lexicon overlap. This isolates "hash bag" vs real embeddings.
const places: Array<{ name: string; tags: string[] }> = [
  { name: "Quiet Beans", tags: ["cafe", "wifi", "study", "calm"] },
  { name: "Roar House", tags: ["sports", "tv", "beer", "crowd"] },
  { name: "Lumiere", tags: ["fine dining", "intimate", "candlelit", "wine"] },
  { name: "Green Bowl", tags: ["vegan", "salad", "healthy", "light"] },
  { name: "Nonna's", tags: ["italian", "pasta", "hearty", "family"] },
  { name: "Brew Lab", tags: ["coffee", "roastery", "espresso", "pour-over"] },
];
const queries: Array<{ q: string; expect: string }> = [
  { q: "somewhere I can focus and get work done", expect: "Quiet Beans" },
  { q: "a place to catch the match with the guys", expect: "Roar House" },
  { q: "somewhere special to impress my partner", expect: "Lumiere" },
  { q: "meat-free clean eating", expect: "Green Bowl" },
  { q: "old-school comfort food", expect: "Nonna's" },
  { q: "a morning caffeine fix", expect: "Brew Lab" },
];

const raw = (name: string, tags: string[]): RawPlace => ({
  name, lat: null, lng: null, category: null, address: null, source: "agent", sourceId: name, tags, raw: {},
});

async function seedAndScore(embedder: "none" | "openai", embedModel?: string) {
  const cfg = loadConfig();
  const mem = buildOpenMap(
    {
      ...cfg,
      dbPath: ":memory:",
      embedder, // "none" = keyword-only (FTS/BM25); "openai" = hybrid (keyword + vector RRF)
      tagger: "lexicon", // hold extraction constant — vary ONLY the embedding arm
      ...(embedModel ? { openaiEmbedModel: embedModel } : {}),
    },
    { allowHeuristicFallbackForTests: true },
  );
  for (const p of places) await mem.rememberPlace(raw(p.name, p.tags), { userId: "u", relationship: "mentioned" });

  const rows: Array<{ q: string; expect: string; top: string; rank: number }> = [];
  for (const { q, expect } of queries) {
    const res = await mem.recall(q, null, 6, "u");
    const names = res.map((r) => r.place.name);
    rows.push({ q, expect, top: names[0] ?? "(none)", rank: names.indexOf(expect) });
  }
  const r1 = rows.filter((r) => r.top === r.expect).length;
  const mrr = rows.reduce((s, r) => s + (r.rank >= 0 ? 1 / (r.rank + 1) : 0), 0) / rows.length;
  return { rows, r1, mrr };
}

async function main() {
  const cfg = loadConfig();
  if (!cfg.openaiApiKey) {
    console.log("\n  No embedding key found (set GOOGLE_API_KEY / OPENAI_API_KEY in .env.local). Skipping the real-embedding column.\n");
  }
  console.log("\n  recall quality — keyword-only vs hybrid (paraphrase queries)\n");
  const hash = await seedAndScore("none");
  const real = cfg.openaiApiKey ? await seedAndScore("openai", cfg.openaiEmbedModel) : null;

  const tag = (top: string, expect: string) => (top === expect ? "✓" : "✗");
  console.log(`  ${"query".padEnd(44)} ${"expected".padEnd(13)} keyword      ${real ? "hybrid" : ""}`);
  console.log(`  ${"-".repeat(86)}`);
  for (let i = 0; i < queries.length; i++) {
    const h = hash.rows[i]!;
    const r = real?.rows[i];
    console.log(
      `  ${h.q.slice(0, 44).padEnd(44)} ${h.expect.padEnd(13)} ${tag(h.top, h.expect)} ${h.top.slice(0, 11).padEnd(11)} ${r ? `${tag(r.top, r.expect)} ${r.top}` : ""}`,
    );
  }
  console.log(`  ${"-".repeat(86)}`);
  console.log(`  keyword: recall@1 ${hash.r1}/${queries.length}  MRR ${hash.mrr.toFixed(2)}`);
  if (real) console.log(`  hybrid : recall@1 ${real.r1}/${queries.length}  MRR ${real.mrr.toFixed(2)}  (kw + ${cfg.openaiEmbedModel})`);
  console.log("");
}

main();

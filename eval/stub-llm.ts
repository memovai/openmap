import { type LLMRunner } from "../src/nlp/llm.js";
import { extractConcepts, extractMeasures, inferCompanion, inferIntents, inferRelationship } from "../src/nlp/extract.js";
import { lexiconFrame } from "../src/nlp/tagger.js";

/**
 * A transparent, regex-backed stand-in for an LLM runner — used by the compare
 * harness when no real key is set, so the offline-vs-LLM comparison is runnable
 * and deterministic. It emulates the things an LLM does that the lexicon can't:
 * **per-place sentiment**, **named regions**, and **per-place concept tags**.
 * With OPENAI_API_KEY set, the harness uses the real model instead.
 */
export class StubLLMRunner implements LLMRunner {
  async run(opts: { system?: string; prompt: string; json?: boolean; model?: string }): Promise<string> {
    const p = opts.prompt;

    if (p.includes("Resolve the latent intent")) {
      const query = p.split("Query:\n").pop() ?? "";
      return JSON.stringify(lexiconFrame(query.trim()));
    }

    if (p.includes("Extract the physical places")) {
      const user = (p.split("User:").pop() ?? "").trim();
      const ctx = (p.split("Assistant (context):").pop() ?? "").split("User:")[0] ?? "";
      const full = `${ctx} ${user}`;
      const regionMatch = full.match(/\bin ([A-Z][A-Za-z]+)\b/);
      const region = regionMatch ? regionMatch[1]! : null;
      const measures = extractMeasures(full);
      const goal = inferIntents(user)[0]?.purpose ?? null;
      const companion = inferCompanion(user);
      // split into clauses so each place gets ITS OWN sentiment, not the sentence's
      const clauses = user.split(/\s*(?:,?\s*but\s|;|,\s*and\s|\.\s)\s*/i);
      const places = [...user.matchAll(/"([^"]{2,60})"/g)].map((m) => {
        const name = m[1]!;
        const clause = clauses.find((c) => c.includes(name)) ?? user;
        return {
          name,
          relationship: inferRelationship(clause),
          companions: companion ? [companion] : [],
          region,
          measures,
          goal,
          concepts: extractConcepts(clause),
        };
      });
      return JSON.stringify({ places });
    }

    if (p.includes("Extract names of physical places")) {
      const text = p.split("Text:\n").pop() ?? "";
      return [...text.matchAll(/"([^"]{2,60})"/g)].map((m) => m[1]).join("\n");
    }

    return "";
  }
}

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export interface Config {
  dbPath: string;
  embedder: string; // auto | openai | none
  tagger: string; // auto | llm (lexicon is test-only and disabled for public builders)
  openaiApiKey: string | null;
  /** OpenAI-compatible base URL (BYOC) — DeepSeek / Gemini / local / gateway. */
  openaiBaseUrl: string | null;
  openaiEmbedModel: string;
  openaiChatModel: string;
  /** Per-component models — each LLM-using layer can run a different model. */
  models: { tagger: string; extractor: string; persona: string };
  beliefHalfLifeDays: number; // recency decay for inferred beliefs (0 = no decay)
}

const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/";
export const MODEL_REQUIRED_ERROR =
  "OpenMap requires an LLM. Set OPENAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY, or inject buildOpenMap(cfg, { llm }). Offline lexicon/heuristic mode is disabled.";
export const LEXICON_DISABLED_ERROR =
  "OPENMAP_TAGGER=lexicon is disabled for public OpenMap builders. Configure an LLM instead.";

let envFilesLoaded = false;
/** Load `.env.local` then `.env` from cwd into process.env (without overriding
 * already-set vars). No dependency — a tiny KEY=VALUE parser. */
function loadEnvFiles(): void {
  if (envFilesLoaded) return;
  envFilesLoaded = true;
  for (const f of [".env.local", ".env"]) {
    let txt: string;
    try {
      txt = readFileSync(join(process.cwd(), f), "utf-8");
    } catch {
      continue;
    }
    for (const line of txt.split("\n")) {
      if (/^\s*#/.test(line) || !line.includes("=")) continue;
      const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*$/);
      if (!m) continue;
      const key = m[1]!;
      let val = m[2]!;
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) val = val.slice(1, -1);
      if (process.env[key] === undefined) process.env[key] = val;
    }
  }
}

/**
 * openmap is a map-aware *memory* layer — its only input is conversation. Config
 * covers how that conversation is understood (extraction/embeddings/models) and
 * remembered. Reads env (auto-loading `.env.local`/`.env`). Natively recognizes a
 * Gemini key (`GEMINI_API_KEY`/`GOOGLE_API_KEY`) and routes via Google's
 * OpenAI-compatible endpoint. Public builders fail fast when no model is
 * configured; the key-free heuristic components are only for explicit tests.
 */
export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  if (env === process.env) loadEnvFiles();

  const geminiKey = env.GEMINI_API_KEY ?? env.GOOGLE_API_KEY ?? null;
  const usingGemini = !env.OPENAI_API_KEY && !!geminiKey;
  const apiKey = env.OPENAI_API_KEY ?? geminiKey ?? null;
  const chatModel = env.OPENMAP_OPENAI_CHAT_MODEL ?? (usingGemini ? "gemini-2.5-flash-lite" : "gpt-4o-mini");

  return {
    dbPath: env.OPENMAP_DB ?? join(homedir(), ".openmap", "openmap.db"),
    embedder: env.OPENMAP_EMBEDDER ?? "auto", // auto → real embeddings when a key is set
    tagger: env.OPENMAP_TAGGER ?? "auto",
    openaiApiKey: apiKey,
    openaiBaseUrl: env.OPENMAP_OPENAI_BASE_URL ?? (usingGemini ? GEMINI_BASE_URL : null),
    // Gemini's OpenAI-compat endpoint serves embeddings as "gemini-embedding-001"
    // (text-embedding-004 404s on that path).
    openaiEmbedModel: env.OPENMAP_OPENAI_EMBED_MODEL ?? (usingGemini ? "gemini-embedding-001" : "text-embedding-3-small"),
    openaiChatModel: chatModel,
    models: {
      tagger: env.OPENMAP_MODEL_TAGGER ?? chatModel,
      extractor: env.OPENMAP_MODEL_EXTRACTOR ?? chatModel,
      persona: env.OPENMAP_MODEL_PERSONA ?? chatModel,
    },
    beliefHalfLifeDays: env.OPENMAP_BELIEF_HALFLIFE_DAYS ? Number(env.OPENMAP_BELIEF_HALFLIFE_DAYS) : 60,
  };
}

export function resolvedEmbedder(cfg: Config): "openai" | "none" {
  if (cfg.embedder === "none") return "none";
  if (cfg.embedder === "openai") return cfg.openaiApiKey ? "openai" : "none";
  return cfg.openaiApiKey ? "openai" : "none"; // auto
}

export function resolvedTagger(cfg: Config): "llm" | "missing" | "disabled" {
  if (cfg.tagger === "lexicon") return "disabled";
  if (cfg.tagger === "llm") return cfg.openaiApiKey ? "llm" : "missing";
  return cfg.openaiApiKey ? "llm" : "missing"; // auto
}

export function assertModelConfigured(
  cfg: Config,
  runner?: unknown,
  opts: { allowHeuristicFallbackForTests?: boolean } = {},
): void {
  if (opts.allowHeuristicFallbackForTests) return;
  if (cfg.tagger === "lexicon") throw new Error(LEXICON_DISABLED_ERROR);
  if (!runner && !cfg.openaiApiKey) throw new Error(MODEL_REQUIRED_ERROR);
}

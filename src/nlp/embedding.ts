import { type Config, resolvedEmbedder } from "../core/config.js";

export interface Embedder {
  dim: number;
  embed(texts: string[]): Promise<Float32Array[]>;
  embedOne(text: string): Promise<Float32Array>;
}

function l2normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i]! * v[i]!;
  const norm = Math.sqrt(sum) || 1;
  for (let i = 0; i < v.length; i++) v[i]! /= norm;
  return v;
}

/** Real embeddings over any OpenAI-compatible endpoint (BYOC: OpenAI / Gemini /
 * local server). `openai` is imported lazily so it stays an optional dep. There
 * is no built-in offline embedder; test/eval code may explicitly opt into
 * keyword-only recall, but public builders require a model. */
export class OpenAIEmbedder implements Embedder {
  dim = 1536;
  private client: unknown;

  constructor(
    private apiKey: string,
    private model = "text-embedding-3-small",
    private baseURL: string | null = null,
  ) {}

  private async getClient(): Promise<any> {
    if (!this.client) {
      const { default: OpenAI } = await import("openai");
      this.client = new OpenAI({ apiKey: this.apiKey, ...(this.baseURL ? { baseURL: this.baseURL } : {}) });
    }
    return this.client;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const client = await this.getClient();
    const resp = await client.embeddings.create({ model: this.model, input: texts });
    return resp.data.map((d: { embedding: number[] }) => {
      this.dim = d.embedding.length;
      return l2normalize(Float32Array.from(d.embedding));
    });
  }
  async embedOne(text: string): Promise<Float32Array> {
    return (await this.embed([text]))[0]!;
  }
}

/** The embedder, or null when no provider is configured (→ keyword-only recall). */
export function getEmbedder(cfg: Config): Embedder | null {
  if (resolvedEmbedder(cfg) === "openai") return new OpenAIEmbedder(cfg.openaiApiKey!, cfg.openaiEmbedModel, cfg.openaiBaseUrl);
  return null;
}

/** Cosine of a query against many vectors (all L2-normalized → dot product). */
export function cosineMatrix(query: Float32Array, mats: Array<Float32Array | null>): number[] {
  return mats.map((m) => {
    if (!m || m.length !== query.length) return 0;
    let s = 0;
    for (let i = 0; i < query.length; i++) s += query[i]! * m[i]!;
    return s;
  });
}

/** L2-normalized average of two vectors (taste centroid + persona embedding). */
export function blendVectors(a: Float32Array | null, b: Float32Array | null): Float32Array | null {
  if (!a && !b) return null;
  if (!a) return b;
  if (!b || a.length !== b.length) return a;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i]! + b[i]!) / 2;
  return l2normalize(out);
}

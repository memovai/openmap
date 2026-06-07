import { type Config } from "../core/config.js";

/**
 * The seam for "who runs the LLM". openmap's extraction (tagger / mention /
 * memory extractor) calls this instead of holding its own client, so it can:
 *  - **borrow the host agent's model** (inject your own LLMRunner — like an
 *    OpenClaw/Hermes runtime), or
 *  - use a **BYOC** OpenAI-compatible endpoint (OpenAILLMRunner + baseURL).
 * Public OpenMap builders require this runner; key-free heuristics are test-only.
 */
export interface LLMRunner {
  run(opts: { system?: string; prompt: string; json?: boolean; model?: string }): Promise<string>;
}

/** Runner over any OpenAI-compatible endpoint (BYOC via baseURL). */
export class OpenAILLMRunner implements LLMRunner {
  private client: unknown;
  constructor(
    private apiKey: string,
    private baseURL: string | null,
    private model: string,
  ) {}

  private async getClient(): Promise<any> {
    if (!this.client) {
      const { default: OpenAI } = await import("openai");
      this.client = new OpenAI({ apiKey: this.apiKey, ...(this.baseURL ? { baseURL: this.baseURL } : {}) });
    }
    return this.client;
  }

  async run(opts: { system?: string; prompt: string; json?: boolean; model?: string }): Promise<string> {
    const client = await this.getClient();
    const messages: Array<{ role: string; content: string }> = [];
    if (opts.system) messages.push({ role: "system", content: opts.system });
    messages.push({ role: "user", content: opts.prompt });
    const resp = await client.chat.completions.create({
      model: opts.model ?? this.model,
      messages,
      temperature: 0,
      ...(opts.json ? { response_format: { type: "json_object" } } : {}),
    });
    return resp.choices[0]?.message?.content ?? "";
  }
}

/** Pull the first JSON object out of a model response (strips ```json fences,
 * prose, etc.) so parsing survives models that don't honor json-mode strictly. */
export function extractJson(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1]! : text;
  const start = body.indexOf("{");
  const end = body.lastIndexOf("}");
  return start >= 0 && end > start ? body.slice(start, end + 1) : body.trim();
}

/** Resolve a runner from config: an OpenAI-compatible one when a key is set,
 * else null. A host can bypass this by injecting its own runner. */
export function getRunner(cfg: Config): LLMRunner | null {
  return cfg.openaiApiKey ? new OpenAILLMRunner(cfg.openaiApiKey, cfg.openaiBaseUrl, cfg.openaiChatModel) : null;
}

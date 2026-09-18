import type { Message, ToolResult, ToolUse } from '../src/index.js';

const SECRET =
  /\b(?:sk|ops|ghp|gho|ghs|xox[bp])[-_][A-Za-z0-9_-]{16,}|\bAIza[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{16,}/g;

/** Replaces common key shapes; `op://` references carry no secret and pass through. */
export function redact(text: string): string {
  return text.replace(SECRET, '[REDACTED]');
}

function redactInput(input: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    out[key] = typeof value === 'string' ? redact(value) : value;
  }
  return out;
}

type Block = {
  type?: string;
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: unknown;
  is_error?: boolean;
};

function resultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return (content as Block[])
    .filter((block) => block.type === 'text')
    .map((block) => block.text ?? '')
    .join('\n');
}

/**
 * A Claude Code session JSONL as the library's Message[]: user and assistant
 * records only, thinking dropped, tool_use paired with tool_result, tool
 * inputs redacted (results never leave the machine, inputs do).
 */
export function transcriptFromJsonl(raw: string): Message[] {
  const messages: Message[] = [];
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    let rec: { type?: string; message?: { content?: unknown } };
    try {
      rec = JSON.parse(line);
    } catch {
      continue;
    }
    if ((rec.type !== 'user' && rec.type !== 'assistant') || !rec.message) continue;
    const content = rec.message.content;
    const message: Message = { role: rec.type, text: '', toolUses: [] };
    const texts: string[] = [];
    if (typeof content === 'string') texts.push(content);
    else if (Array.isArray(content)) {
      for (const block of content as Block[]) {
        if (block.type === 'text') texts.push(block.text ?? '');
        else if (block.type === 'tool_use' && block.id && block.name) {
          message.toolUses.push({
            tool_use_id: block.id,
            tool: block.name,
            input: redactInput(block.input ?? {}),
          });
        } else if (block.type === 'tool_result' && block.tool_use_id) {
          const result: ToolResult = {
            tool_use_id: block.tool_use_id,
            text: resultText(block.content),
            isError: block.is_error === true,
          };
          (message.toolResults ??= []).push(result);
        }
      }
    }
    message.text = texts.join('\n');
    if (!message.text && message.toolUses.length === 0 && !message.toolResults?.length) continue;
    messages.push(message);
  }
  const results = new Map<string, ToolResult>();
  for (const message of messages) {
    for (const result of message.toolResults ?? []) results.set(result.tool_use_id, result);
  }
  for (const message of messages) {
    for (const tool of message.toolUses as ToolUse[]) {
      const result = results.get(tool.tool_use_id);
      if (result) {
        tool.text = result.text;
        tool.isError = result.isError;
      }
    }
  }
  return messages;
}

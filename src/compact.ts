import { noulAnswer } from './request.js';
import { collectToolCalls, estimateTokens, fitState } from './state.js';
import type {
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  CompactionState,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  keepCallThreshold: 0.5,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  truncateHeadChars: 300,
  truncateTailChars: 0,
  truncateInputChars: 0,
  alwaysKeepCall: [],
  alwaysKeepResult: [],
};

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function patterns(list: readonly string[] | undefined, name: string): RegExp[] {
  return (list ?? []).map((source) => {
    try {
      return new RegExp(source);
    } catch (error) {
      throw new Error(
        `invalid ${name} pattern ${JSON.stringify(source)}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  });
}

/** Whether a tool name matches any of the floor patterns. */
export function matchesAny(patterns: readonly RegExp[], tool: string): boolean {
  return patterns.some((pattern) => pattern.test(tool));
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  const keepThreshold = finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold);
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold,
    keepCallThreshold: finite(options.keepCallThreshold, keepThreshold),
    truncateInputChars: Math.max(
      0,
      Math.floor(finite(options.truncateInputChars, DEFAULT_OPTIONS.truncateInputChars)),
    ),
    alwaysKeepCall: patterns(options.alwaysKeepCall, 'alwaysKeepCall'),
    alwaysKeepResult: patterns(options.alwaysKeepResult, 'alwaysKeepResult'),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
    truncateTailChars: Math.max(
      0,
      Math.floor(finite(options.truncateTailChars, DEFAULT_OPTIONS.truncateTailChars)),
    ),
  };
}

const WRITERS = new Set(['Write', 'Edit', 'MultiEdit', 'NotebookEdit']);

function str(input: Record<string, unknown>, key: string): string | undefined {
  const value = input[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function filePath(input: Record<string, unknown>): string | undefined {
  return str(input, 'file_path') ?? str(input, 'notebook_path');
}

/** What makes two calls the same call, per tool; undefined when the tool has no such key. */
export function identityKey(tool: string, input: Record<string, unknown>): string | undefined {
  switch (tool) {
    case 'Read': {
      const path = filePath(input);
      return path && `${path}|${String(input['offset'] ?? '')}|${String(input['limit'] ?? '')}`;
    }
    case 'Bash':
    case 'PowerShell':
      return str(input, 'command');
    case 'Grep': {
      const pattern = str(input, 'pattern');
      return pattern && `${pattern}|${str(input, 'path') ?? ''}|${str(input, 'glob') ?? ''}`;
    }
    case 'Glob': {
      const pattern = str(input, 'pattern');
      return pattern && `${pattern}|${str(input, 'path') ?? ''}`;
    }
    case 'ToolSearch':
      return str(input, 'query');
    default:
      return undefined;
  }
}

/**
 * Calls a later call made stale, decided without Jev: a Read whose file was
 * later written or edited, or read again the same way; a Write whose file was
 * later written or edited; an Edit whose file was later rewritten whole; any
 * other call repeated identically later. Pinned calls and floored-result tools
 * are never superseded.
 */
export function supersededCalls(
  calls: readonly ToolCall[],
  alwaysKeepResult: readonly RegExp[],
): Set<string> {
  const stale = new Set<string>();
  calls.forEach((call, i) => {
    if (call.pinned || matchesAny(alwaysKeepResult, call.tool)) return;
    const path = filePath(call.input);
    const key = identityKey(call.tool, call.input);
    for (let j = i + 1; j < calls.length; j += 1) {
      const later = calls[j]!;
      const laterPath = filePath(later.input);
      const sameFile = path !== undefined && laterPath === path;
      let hit = false;
      if (call.tool === 'Read') {
        hit = (sameFile && WRITERS.has(later.tool)) || (later.tool === 'Read' && key !== undefined && identityKey('Read', later.input) === key);
      } else if (call.tool === 'Write') {
        hit = sameFile && WRITERS.has(later.tool);
      } else if (WRITERS.has(call.tool)) {
        hit = sameFile && later.tool === 'Write';
      } else if (key !== undefined) {
        hit = later.tool === call.tool && identityKey(later.tool, later.input) === key;
      }
      if (hit) {
        stale.add(call.id);
        return;
      }
    }
  });
  return stale;
}

export type QuestionOptions = Partial<Pick<ResolvedCompactOptions, 'truncateInputChars'>>;

/** Whether the input of this call is long enough to be asked about. */
export function asksInput(call: Pick<ToolCall, 'inputChars'>, options: QuestionOptions): boolean {
  const limit = options.truncateInputChars ?? 0;
  return limit > 0 && call.inputChars > limit;
}

/**
 * The `noul` questions asked about one call: keep the call, keep its result,
 * and (long inputs only, when `truncateInputChars` is set) keep its input.
 */
export function questionsFor(call: ToolCall, options: QuestionOptions = {}): JevQuestions {
  const questions: JevQuestions = {
    [`call_${call.id}`]: {
      type: 'noul',
      instructions: `Tool call ${call.id} (${call.tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next`,
      criteria: {
        true: 'A later step depends on remembering this call happened: it changed something, its input records a decision, or the goal refers back to it',
        false: 'It only looked something up, or the work it belonged to is finished and nothing in the goal refers back to it',
      },
    },
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The full output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
      criteria: {
        true: 'The output holds something the assistant will need again and cannot get back by re-running the tool or reading a file: a one-off error, a number or fact it has not written down, a listing that changes over time',
        false: "The output is a file's current content, routine command output, or something the assistant already restated in its own text",
      },
    },
  };
  if (asksInput(call, options)) {
    questions[`input_${call.id}`] = {
      type: 'noul',
      instructions: `The full input of tool call ${call.id} (${call.tool}, ${call.inputChars} chars) should stay in the history verbatim: the assistant still needs exactly what was written or edited and re-reading the file would not do`,
    };
  }
  return questions;
}

/** Cuts every string field longer than `limit` to its head plus a note; other fields and keys are untouched. */
export function truncateInput(
  input: Record<string, unknown>,
  limit: number,
): { input: Record<string, unknown>; changed: boolean } {
  let changed = false;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (typeof value === 'string' && value.length > limit) {
      out[key] = `${value.slice(0, limit)}\n[fast-jev-compaction truncated ${value.length - limit} chars of this input; re-read the file if needed]`;
      changed = true;
    } else out[key] = value;
  }
  return { input: changed ? out : input, changed };
}

/**
 * Splits the candidate calls into batches whose questions, together with the
 * (always complete) state, fit one request.
 */
export function batchCalls(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens'> & QuestionOptions,
): ToolCall[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const batches: ToolCall[][] = [];
  let current: ToolCall[] = [];
  let currentTokens = 0;
  for (const call of calls) {
    const tokens = estimateTokens(JSON.stringify(questionsFor(call, options)));
    if (current.length > 0 && currentTokens + tokens > budget) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(call);
    currentTokens += tokens;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export type DecisionOptions = Pick<ResolvedCompactOptions, 'keepThreshold'> &
  Partial<Pick<ResolvedCompactOptions, 'keepCallThreshold' | 'alwaysKeepCall' | 'alwaysKeepResult'>>;

/**
 * Keep, drop the result, or drop the call. A result stays at `keepThreshold`;
 * a call stays at `keepCallThreshold` (defaults to `keepThreshold`) or when its
 * tool is floored; a floored-result tool keeps everything. A long input is
 * cut when Jev's `keepInput` is below `keepThreshold` and the call stays.
 */
export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  answer: CallAnswer,
  options: DecisionOptions,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer, inputTruncated: false };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (matchesAny(options.alwaysKeepResult ?? [], call.tool)) {
    return { ...base, action: 'keep', reason: 'floor' };
  }
  const inputTruncated = answer.keepInput !== undefined && answer.keepInput < options.keepThreshold;
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, inputTruncated, action: 'keep', reason: 'kept' };
  }
  const callThreshold = options.keepCallThreshold ?? options.keepThreshold;
  if (answer.keepCall >= callThreshold || matchesAny(options.alwaysKeepCall ?? [], call.tool)) {
    return { ...base, inputTruncated, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

/** The decision for a superseded call: no Jev, the call line survives only on a floor, a long input is cut. */
export function decideSuperseded(
  call: Pick<ToolCall, 'id' | 'tool' | 'inputChars'>,
  options: Pick<ResolvedCompactOptions, 'truncateInputChars' | 'alwaysKeepCall'>,
): CallDecision {
  const base = {
    id: call.id,
    tool: call.tool,
    keepCall: 0,
    keepResult: 0,
    inputTruncated: asksInput(call, options),
  };
  if (matchesAny(options.alwaysKeepCall, call.tool)) {
    return { ...base, action: 'drop_result', reason: 'superseded' };
  }
  return { ...base, inputTruncated: false, action: 'drop_call', reason: 'superseded' };
}

async function askBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
  options: QuestionOptions,
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map((call) => questionsFor(call, options)));
  const { answers } = await asker.ask(state, questions);
  return new Map(
    batch.map((call) => {
      const answer: CallAnswer = {
        keepCall: noulAnswer(answers, `call_${call.id}`),
        keepResult: noulAnswer(answers, `result_${call.id}`),
      };
      if (asksInput(call, options)) answer.keepInput = noulAnswer(answers, `input_${call.id}`);
      return [call.id, answer];
    }),
  );
}

function truncatedResultText(
  text: string,
  isError: boolean,
  headChars: number,
  tailChars: number = 0,
): string {
  if (text.length <= headChars + tailChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  const tail = tailChars > 0 ? `\n${text.slice(-tailChars)}` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - headChars - tailChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]${tail}`;
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result; a dropped result keeps a bounded head and note.
 * Messages that lose all their content are removed; untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
  inputChars: number = 0,
  tailChars: number = 0,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const plans = new Map<string, { action: CallDecision['action']; inputTruncated: boolean }>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && (decision.action !== 'keep' || decision.inputTruncated)) {
      plans.set(call.tool_use_id, { action: decision.action, inputTruncated: decision.inputTruncated });
    }
  }
  const actionOf = (id: string): CallDecision['action'] => plans.get(id)?.action ?? 'keep';
  const kept: Message[] = [];
  for (const message of messages) {
    const touched =
      message.toolUses.some((tool) => plans.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => plans.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actionOf(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        const plan = plans.get(tool.tool_use_id);
        if (!plan) return tool;
        const text =
          plan.action === 'drop_result'
            ? truncatedResultText(tool.text ?? '', tool.isError ?? false, headChars, tailChars)
            : tool.text;
        const textChanged = plan.action === 'drop_result' && text !== (tool.text ?? '');
        const cut =
          plan.inputTruncated && inputChars > 0
            ? truncateInput(tool.input, inputChars)
            : { input: tool.input, changed: false };
        if (!textChanged && !cut.changed) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: cut.input,
        };
        if (text !== undefined) copy.text = text;
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actionOf(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actionOf(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars, tailChars);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !message.toolUses.some((tool) => actionOf(tool.tool_use_id) === 'drop_call') &&
      !(message.toolResults ?? []).some((result) => actionOf(result.tool_use_id) === 'drop_call') &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/**
 * Compacts a transcript by asking Jev, for every tool call outside the pinned
 * first and newest messages, whether the call and whether its result must
 * stay. The whole history (results omitted, fitted into `maxStateTokens`) is
 * sent as state with every batch of questions. Throws when Jev fails or the
 * history cannot be fitted; the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const stale = supersededCalls(calls, resolved.alwaysKeepResult);
  const candidates = calls.filter(
    (call) => !call.pinned && !stale.has(call.id) && !matchesAny(resolved.alwaysKeepResult, call.tool),
  );
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let fitted: { tokens: number; stage: string } = { tokens: 0, stage: '' };
  let batches: ToolCall[][] = [];
  const answers = new Map<string, CallAnswer>();
  if (candidates.length > 0) {
    const state = fitState(messages, calls, resolved);
    fitted = state;
    batches = batchCalls(candidates, state.tokens, resolved);
    const answered = await Promise.all(
      batches.map((batch) => askBatch(asker, state.state, batch, resolved)),
    );
    for (const map of answered) for (const [id, answer] of map) answers.set(id, answer);
  }

  const decisions = calls.map((call) =>
    stale.has(call.id)
      ? decideSuperseded(call, resolved)
      : decideCall(call, answers.get(call.id) ?? { keepCall: 1, keepResult: 1 }, resolved),
  );
  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
    resolved.truncateInputChars,
    resolved.truncateTailChars,
  );
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      floored: count(decisions, 'floor'),
      superseded: count(decisions, 'superseded'),
      inputsTruncated: decisions.filter((decision) => decision.inputTruncated).length,
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests: batches.length,
      ms: Date.now() - started,
    },
  };
}

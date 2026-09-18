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
  };
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
    },
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The full output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
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

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
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
            ? truncatedResultText(tool.text ?? '', tool.isError ?? false, headChars)
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
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
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
  const candidates = calls.filter(
    (call) => !call.pinned && !matchesAny(resolved.alwaysKeepResult, call.tool),
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
    decideCall(call, answers.get(call.id) ?? { keepCall: 1, keepResult: 1 }, resolved),
  );
  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
    resolved.truncateInputChars,
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
      inputsTruncated: decisions.filter((decision) => decision.inputTruncated).length,
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests: batches.length,
      ms: Date.now() - started,
    },
  };
}

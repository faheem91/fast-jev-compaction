/**
 * Offline eval: run the compaction over a real Claude Code session transcript
 * with real Jev and print every decision that changes the history, so the
 * thresholds and floors can be judged by a human before the plugin goes live.
 *
 *   TYPESAFE_API_KEY=... npx tsx eval/run.ts <session.jsonl> [upstream|ours|ours-call04]
 */
import { readFileSync } from 'node:fs';
import {
  collectToolCalls,
  compact,
  JevClient,
  reductionRatio,
  resolveOptions,
  type CompactOptions,
  type JevAsker,
  type JevQuestions,
  type JevState,
} from '../src/index.js';
import { transcriptFromJsonl } from './transcript.js';

const MODEL = 'jev-1.13.0';
const PRICE_PER_MTOK = 0.042;
const OURS: CompactOptions = {
  keepThreshold: 0.5,
  keepCallThreshold: 0.3,
  truncateInputChars: 1500,
  truncateHeadChars: 400,
  truncateTailChars: 200,
  preserveRecentMessages: 8,
  alwaysKeepCall: [
    '^(Write|Edit|MultiEdit|NotebookEdit|Bash|PowerShell|Agent|SendMessage|Artifact|Skill)$',
    '^mcp__.*__.*(create|update|delete|send|set_|write|publish|post|move|add_|remove|merge|upload|execute|run_).*',
  ],
  alwaysKeepResult: [
    '^(Agent|TaskOutput|WebFetch|WebSearch|AskUserQuestion)$',
    '^mcp__(firecrawl|claude_ai_Ahrefs|fathom|claude_ai_Apollo_io|parallel|seo-research|higgsfield|claude_ai_Clay|qmd)__',
  ],
};
const CONFIGS: Record<string, CompactOptions> = {
  upstream: { preserveRecentMessages: 6 },
  ours: OURS,
  'ours-call04': { ...OURS, keepCallThreshold: 0.4 },
};
const MUTATION = /^(Write|Edit|MultiEdit|NotebookEdit|Bash|PowerShell|Agent|SendMessage|Artifact|Skill)$/;

const [path, only, limit] = process.argv.slice(2);
if (!path) {
  console.error('usage: tsx eval/run.ts <session.jsonl> [upstream|ours|ours-call04] [maxMessages]');
  process.exit(2);
}
const whole = transcriptFromJsonl(readFileSync(path, 'utf8'));
const messages = limit ? whole.slice(0, Number(limit)) : whole;
console.log(
  `transcript: ${messages.length} of ${whole.length} messages, ${messages.reduce((n, m) => n + m.toolUses.length, 0)} tool uses`,
);

for (const [name, options] of Object.entries(CONFIGS)) {
  if (only && name !== only) continue;
  const client = new JevClient({ model: MODEL });
  let inputTokens = 0;
  const asker: JevAsker = {
    async ask(state: JevState, questions: JevQuestions) {
      const response = await client.ask(state, questions);
      inputTokens += response.usage?.input_tokens ?? 0;
      return response;
    },
  };
  const started = Date.now();
  try {
    const result = await compact(messages, asker, options);
    const s = result.stats;
    console.log(
      `\n## ${name}: ${(reductionRatio(result) * 100).toFixed(1)}% reduction | messages ${s.messagesBefore} -> ${s.messagesAfter} | chars ${s.charsBefore.toLocaleString()} -> ${s.charsAfter.toLocaleString()} | calls ${s.calls}: pinned ${s.pinned}, floored ${s.floored}, superseded ${s.superseded}, kept ${s.kept}, results truncated ${s.resultsDropped}, calls dropped ${s.callsDropped}, inputs truncated ${s.inputsTruncated} | ${s.requests} requests, state ~${s.stateTokens} tokens (${s.stateStage}) | ${inputTokens.toLocaleString()} input tokens = $${((inputTokens / 1e6) * PRICE_PER_MTOK).toFixed(4)} | ${Date.now() - started} ms`,
    );
    const resolved = resolveOptions(options);
    const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
    const byId = new Map(calls.map((call) => [call.id, call]));
    for (const d of result.decisions) {
      if (d.action === 'keep' && !d.inputTruncated) continue;
      const call = byId.get(d.id);
      const head = call ? JSON.stringify(call.input).slice(0, 90) : '';
      const flag = d.action === 'drop_call' && MUTATION.test(d.tool) ? ' !MUTATION' : '';
      const input = d.keepInput !== undefined ? ` input=${d.keepInput.toFixed(2)}` : '';
      const why = d.reason === 'superseded' ? ' superseded' : '';
      console.log(
        `${d.id} ${d.tool} ${d.action}${d.inputTruncated ? '+input' : ''}${why} call=${d.keepCall.toFixed(2)} result=${d.keepResult.toFixed(2)}${input}${flag} ${head}`,
      );
    }
  } catch (error) {
    console.log(
      `\n## ${name}: FAILED ${error instanceof Error ? error.message : String(error)} (${inputTokens} input tokens spent)`,
    );
  }
}

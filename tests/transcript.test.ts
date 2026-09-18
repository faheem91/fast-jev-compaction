import { describe, expect, it } from 'vitest';
import { redact, transcriptFromJsonl } from '../eval/transcript.js';

const lines = [
  JSON.stringify({ type: 'system', content: 'ignored' }),
  JSON.stringify({ type: 'user', message: { role: 'user', content: 'Fix the test. Token sk-abcdefghijklmnopqrstuvwxyz12' } }),
  JSON.stringify({
    type: 'assistant',
    message: {
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'private' },
        { type: 'text', text: 'Reading.' },
        { type: 'tool_use', id: 'toolu_1', name: 'Bash', input: { command: 'echo ghp_abcdefghijklmnopqrstuvwxyz1234' } },
      ],
    },
  }),
  JSON.stringify({
    type: 'user',
    message: {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: [{ type: 'text', text: 'out' }], is_error: false }],
    },
  }),
  JSON.stringify({ type: 'attachment', attachment: {} }),
  '',
].join('\n');

describe('transcriptFromJsonl', () => {
  it('keeps user and assistant records, drops thinking, pairs tools, redacts inputs', () => {
    const messages = transcriptFromJsonl(lines);
    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(messages[0]?.text).toBe('Fix the test. Token sk-abcdefghijklmnopqrstuvwxyz12');
    expect(messages[1]?.text).toBe('Reading.');
    expect(messages[1]?.toolUses[0]).toMatchObject({
      tool_use_id: 'toolu_1',
      tool: 'Bash',
      input: { command: 'echo [REDACTED]' },
      text: 'out',
      isError: false,
    });
    expect(messages[2]?.toolResults?.[0]).toEqual({ tool_use_id: 'toolu_1', text: 'out', isError: false });
  });

  it('redacts common key shapes and leaves op:// references alone', () => {
    expect(redact('op://FlowScale-Dev/abc/credential')).toBe('op://FlowScale-Dev/abc/credential');
    expect(redact('Bearer abcdefghijklmnop.qrstuv')).toBe('[REDACTED]');
    expect(redact('xoxb-1234567890123456789')).toBe('[REDACTED]');
  });
});

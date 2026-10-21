import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { TestModelProvider } from '../../__fixtures__/model-test-helpers.js'
import { TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'

describe('malformed tool input JSON', () => {
  it('skips the tool and returns an error tool result with the parse error', async () => {
    let calls = 0
    const model = new TestModelProvider(async function* () {
      calls++
      yield { type: 'modelMessageStartEvent', role: 'assistant' }
      if (calls === 1) {
        yield {
          type: 'modelContentBlockStartEvent',
          start: { type: 'toolUseStart', toolUseId: 'tool-1', name: 'getWeather' },
        }
        yield {
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'toolUseInputDelta', input: '{"city": "Par' },
        }
        yield { type: 'modelContentBlockStopEvent' }
        yield { type: 'modelMessageStopEvent', stopReason: 'toolUse' }
      } else {
        yield { type: 'modelContentBlockStartEvent' }
        yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'Let me fix that JSON.' } }
        yield { type: 'modelContentBlockStopEvent' }
        yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
      }
    })

    // No tools are registered: any execution attempt would fail loudly,
    // so a passing run proves the tool was never invoked.
    const agent = new Agent({ model, printer: false })
    const result = await agent.invoke('weather?')

    expect(result.stopReason).toBe('endTurn')

    const toolUse = agent.messages
      .flatMap((message) => message.content)
      .find((block) => block instanceof ToolUseBlock) as ToolUseBlock
    expect(toolUse.toolUseId).toBe('tool-1')
    expect(toolUse.inputParseError).toBe('{"city": "Par')

    const toolResult = agent.messages
      .flatMap((message) => message.content)
      .find((block) => block instanceof ToolResultBlock) as ToolResultBlock
    expect(toolResult.toolUseId).toBe('tool-1')
    expect(toolResult.status).toBe('error')
    expect((toolResult.content[0] as TextBlock).text).toContain('Invalid JSON in tool input')
    expect((toolResult.content[0] as TextBlock).text).toContain('{"city": "Par')
  })
})

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { queryAgent, type AgentResult } from '../api';
import { Brain, Wrench } from 'lucide-react';

export default function Agent() {
  const [query, setQuery] = useState('');

  const mutation = useMutation({
    mutationFn: (q: string) => queryAgent(q),
  });

  const result = mutation.data as AgentResult | undefined;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Reef Decision-Support Agent</h1>
      <p className="text-gray-500 mb-6">
        Multi-step reasoning agent with tools: reef state queries, simulations, knowledge search, and stress analysis.
      </p>

      <div className="flex gap-2 mb-6">
        <div className="relative flex-1">
          <Brain size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && query.trim() && mutation.mutate(query)}
            placeholder="Compare bleaching risk across all reefs and recommend interventions"
            className="w-full bg-gray-900 border border-gray-700 rounded-lg pl-10 pr-4 py-2.5 text-sm focus:outline-none focus:border-cyan-600"
          />
        </div>
        <button
          onClick={() => query.trim() && mutation.mutate(query)}
          disabled={mutation.isPending || !query.trim()}
          className="bg-purple-700 hover:bg-purple-600 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50"
        >
          {mutation.isPending ? 'Thinking...' : 'Ask Agent'}
        </button>
      </div>

      {result && (
        <div className="space-y-4">
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <h2 className="text-lg font-semibold mb-3">Answer</h2>
            <div className="text-gray-300 leading-relaxed whitespace-pre-wrap">{result.answer}</div>
            <div className="mt-3 text-xs text-gray-600">
              Iterations: {result.iterations} | Tokens: {result.tokens.input} in / {result.tokens.output} out
            </div>
          </div>

          {result.tool_calls.length > 0 && (
            <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
              <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Wrench size={18} /> Tool Calls ({result.tool_calls.length})
              </h2>
              <div className="space-y-3">
                {result.tool_calls.map((tc, i) => (
                  <div key={i} className="bg-gray-800 rounded p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-mono text-purple-400">{tc.tool}</span>
                      <span className="text-xs text-gray-600">
                        ({JSON.stringify(tc.input).slice(0, 80)})
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 font-mono">
                      {tc.result_summary.slice(0, 150)}...
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

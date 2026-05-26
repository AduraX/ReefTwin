import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { queryRAG, type RAGResult } from '../api';
import { Search } from 'lucide-react';

export default function KnowledgeBase() {
  const [question, setQuestion] = useState('');

  const mutation = useMutation({
    mutationFn: (q: string) => queryRAG(q),
  });

  const result = mutation.data as RAGResult | undefined;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Reef Knowledge Base</h1>
      <p className="text-gray-500 mb-6">
        Search reef science literature using hybrid RAG (BM25 + dense + Reciprocal Rank Fusion).
      </p>

      <div className="flex gap-2 mb-6">
        <div className="relative flex-1">
          <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && question.trim() && mutation.mutate(question)}
            placeholder="What causes coral bleaching?"
            className="w-full bg-gray-900 border border-gray-700 rounded-lg pl-10 pr-4 py-2.5 text-sm focus:outline-none focus:border-cyan-600"
          />
        </div>
        <button
          onClick={() => question.trim() && mutation.mutate(question)}
          disabled={mutation.isPending || !question.trim()}
          className="bg-cyan-700 hover:bg-cyan-600 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50"
        >
          {mutation.isPending ? 'Searching...' : 'Search'}
        </button>
      </div>

      {result && (
        <div className="space-y-4">
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <h2 className="text-lg font-semibold mb-3">Answer</h2>
            <div className="text-gray-300 leading-relaxed whitespace-pre-wrap">{result.answer}</div>
            <div className="mt-3 text-xs text-gray-600">
              Model: {result.model} | Retrieval: {result.retrieval_method}
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <h2 className="text-lg font-semibold mb-3">Sources ({result.sources.length})</h2>
            <div className="space-y-3">
              {result.sources.map((src, i) => (
                <div key={i} className="border-l-2 border-cyan-800 pl-4">
                  <div className="text-sm font-medium text-cyan-400">
                    [{src.metadata?.source}] {src.metadata?.topic}
                  </div>
                  <div className="text-sm text-gray-500 mt-1">
                    {src.content.slice(0, 200)}...
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

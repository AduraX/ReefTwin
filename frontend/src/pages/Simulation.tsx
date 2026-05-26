import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { fetchReefs, runSimulation, type SimulationResult } from '../api';

export default function Simulation() {
  const { data: reefs } = useQuery({ queryKey: ['reefs'], queryFn: fetchReefs });
  const [reefId, setReefId] = useState('');
  const [tempDelta, setTempDelta] = useState(1.5);
  const [duration, setDuration] = useState(21);
  const [turbDelta, setTurbDelta] = useState(0);
  const [phDelta, setPhDelta] = useState(0);

  const mutation = useMutation({
    mutationFn: () =>
      runSimulation({
        reef_id: reefId || reefs?.[0]?.reef_id || '',
        temperature_delta_c: tempDelta,
        duration_days: duration,
        turbidity_delta_pct: turbDelta,
        ph_delta: phDelta,
      }),
  });

  const result = mutation.data as SimulationResult | undefined;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Scenario Simulation</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Parameters</h2>

          <label className="block mb-3">
            <span className="text-sm text-gray-400">Reef</span>
            <select
              className="block w-full mt-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
              value={reefId || reefs?.[0]?.reef_id || ''}
              onChange={e => setReefId(e.target.value)}
            >
              {reefs?.map(r => (
                <option key={r.reef_id} value={r.reef_id}>
                  {r.reef_id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </option>
              ))}
            </select>
          </label>

          <label className="block mb-3">
            <span className="text-sm text-gray-400">Temperature change: {tempDelta > 0 ? '+' : ''}{tempDelta}°C</span>
            <input type="range" min={-2} max={5} step={0.1} value={tempDelta}
              onChange={e => setTempDelta(Number(e.target.value))}
              className="block w-full mt-1" />
          </label>

          <label className="block mb-3">
            <span className="text-sm text-gray-400">Duration: {duration} days</span>
            <input type="range" min={1} max={180} step={1} value={duration}
              onChange={e => setDuration(Number(e.target.value))}
              className="block w-full mt-1" />
          </label>

          <label className="block mb-3">
            <span className="text-sm text-gray-400">Turbidity change: {turbDelta > 0 ? '+' : ''}{turbDelta}%</span>
            <input type="range" min={-50} max={200} step={5} value={turbDelta}
              onChange={e => setTurbDelta(Number(e.target.value))}
              className="block w-full mt-1" />
          </label>

          <label className="block mb-4">
            <span className="text-sm text-gray-400">pH change: {phDelta >= 0 ? '+' : ''}{phDelta}</span>
            <input type="range" min={-0.5} max={0.2} step={0.01} value={phDelta}
              onChange={e => setPhDelta(Number(e.target.value))}
              className="block w-full mt-1" />
          </label>

          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="w-full bg-cyan-700 hover:bg-cyan-600 text-white font-medium py-2 px-4 rounded transition disabled:opacity-50"
          >
            {mutation.isPending ? 'Running...' : 'Run Simulation'}
          </button>
        </div>

        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Results</h2>
          {!result ? (
            <p className="text-gray-500">Configure parameters and run a simulation.</p>
          ) : (
            <div>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <Metric label="Baseline Risk" value={`${(result.baseline_risk * 100).toFixed(1)}%`} />
                <Metric
                  label="Projected Risk"
                  value={`${(result.projected_bleaching_risk * 100).toFixed(1)}%`}
                  delta={result.projected_bleaching_risk - result.baseline_risk}
                />
                <Metric label="Status" value={result.projected_ecosystem_status} />
              </div>
              <RiskBar risk={result.projected_bleaching_risk} baseline={result.baseline_risk} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, delta }: { label: string; value: string; delta?: number }) {
  return (
    <div className="text-center">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {delta !== undefined && (
        <div className={`text-sm mt-1 ${delta > 0 ? 'text-red-400' : 'text-green-400'}`}>
          {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}

function RiskBar({ risk, baseline }: { risk: number; baseline: number }) {
  const pct = Math.min(risk * 100, 100);
  const basePct = Math.min(baseline * 100, 100);
  const color = risk >= 0.85 ? 'bg-red-600' : risk >= 0.7 ? 'bg-orange-500' : risk >= 0.5 ? 'bg-yellow-500' : 'bg-green-500';
  return (
    <div className="mt-4">
      <div className="text-xs text-gray-500 mb-1">Risk Scale</div>
      <div className="relative h-6 bg-gray-800 rounded-full overflow-hidden">
        <div className={`absolute inset-y-0 left-0 ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
        <div className="absolute inset-y-0 border-l-2 border-white/60" style={{ left: `${basePct}%` }} title="Baseline" />
      </div>
      <div className="flex justify-between text-xs text-gray-600 mt-1">
        <span>0%</span><span>50% Watch</span><span>70% Warning</span><span>85% Alert</span><span>100%</span>
      </div>
    </div>
  );
}

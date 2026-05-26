import { useQuery } from '@tanstack/react-query';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { fetchReefs, type ReefState } from '../api';

// Client-side stress scoring (mirrors models/stress_scoring.py)
function sigmoid(x: number, midpoint: number, steepness: number) {
  return 1 / (1 + Math.exp(-steepness * (x - midpoint)));
}

function computeStress(state: ReefState) {
  const thermal =
    sigmoid(state.sst_celsius ? state.sst_celsius - 28.2 : 0, 1.0, 2.0) * 0.5 +
    sigmoid(state.degree_heating_weeks > 0 ? Math.max(0, state.sst_celsius - 28.2 - 0.7) : 0, 1.0, 3.0) * 0.5;
  const wq =
    sigmoid(state.turbidity_ntu, 1.5, 2.0) * 0.6 +
    sigmoid(Math.abs(state.ph - 8.1), 0.15, 10.0) * 0.4;
  const bio = sigmoid(Math.max(0, 5.0 - state.dissolved_oxygen_mg_l), 1.0, 3.0);
  const cumulative = sigmoid(state.degree_heating_weeks, 4.0, 0.5);

  return [
    { dimension: 'Thermal', value: thermal },
    { dimension: 'Water Quality', value: wq },
    { dimension: 'Biological', value: bio },
    { dimension: 'Cumulative', value: cumulative },
  ];
}

export default function StressAnalysis() {
  const { data: reefs, isLoading } = useQuery({ queryKey: ['reefs'], queryFn: fetchReefs });

  if (isLoading) return <div className="text-gray-400">Loading...</div>;
  if (!reefs?.length) return <div className="text-gray-400">No reef data available.</div>;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Multi-Factor Stress Analysis</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {reefs.map((state: ReefState) => {
          const stress = computeStress(state);
          const total = stress.reduce((s, d) => s + d.value, 0) / stress.length;
          const dominant = stress.reduce((a, b) => (b.value > a.value ? b : a)).dimension;
          const name = state.reef_id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

          return (
            <div key={state.reef_id} className="bg-gray-900 rounded-lg border border-gray-800 p-6">
              <div className="flex justify-between items-start mb-2">
                <h2 className="text-lg font-semibold">{name}</h2>
                <span className="text-sm bg-gray-800 px-2 py-1 rounded">
                  Total: {(total * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-gray-500 mb-4">Dominant stressor: <strong className="text-gray-300">{dominant}</strong></p>

              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={stress}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis dataKey="dimension" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                  <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 10 }} />
                  <Radar
                    dataKey="value"
                    stroke={total > 0.5 ? '#ef4444' : total > 0.3 ? '#f97316' : '#22c55e'}
                    fill={total > 0.5 ? '#ef444440' : total > 0.3 ? '#f9731640' : '#22c55e40'}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}

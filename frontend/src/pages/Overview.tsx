import { useQuery } from '@tanstack/react-query';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import { fetchReefs, type ReefState } from '../api';
import ReefCard from '../components/ReefCard';

function riskColor(risk: number) {
  if (risk >= 0.85) return '#dc2626';
  if (risk >= 0.7) return '#ea580c';
  if (risk >= 0.5) return '#eab308';
  return '#22c55e';
}

export default function Overview() {
  const { data: reefs, isLoading, error } = useQuery({
    queryKey: ['reefs'],
    queryFn: fetchReefs,
    refetchInterval: 30000,
  });

  if (isLoading) return <div className="text-gray-400">Loading reef states...</div>;
  if (error) return <div className="text-red-400">Error loading reef data. Run `make update-twin` first.</div>;
  if (!reefs?.length) return <div className="text-gray-400">No reef data available.</div>;

  const chartData = reefs.map((r: ReefState) => ({
    name: r.reef_id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    risk: r.bleaching_risk_score,
  }));

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Reef State Overview</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {reefs.map((s: ReefState) => (
          <ReefCard key={s.reef_id} state={s} />
        ))}
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Bleaching Risk Comparison</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af' }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
              formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`, 'Risk']}
            />
            <ReferenceLine y={0.5} stroke="#eab308" strokeDasharray="5 5" label={{ value: 'Watch', fill: '#eab308', fontSize: 11 }} />
            <ReferenceLine y={0.85} stroke="#dc2626" strokeDasharray="5 5" label={{ value: 'Alert', fill: '#dc2626', fontSize: 11 }} />
            <Bar dataKey="risk" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={riskColor(entry.risk)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

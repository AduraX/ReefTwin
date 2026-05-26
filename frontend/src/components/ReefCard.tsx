import type { ReefState } from '../api';

const STATUS_COLORS: Record<string, string> = {
  stable: 'bg-green-900/50 border-green-700 text-green-300',
  watch: 'bg-yellow-900/50 border-yellow-700 text-yellow-300',
  stressed: 'bg-orange-900/50 border-orange-700 text-orange-300',
  critical: 'bg-red-900/50 border-red-700 text-red-300',
};

export default function ReefCard({ state }: { state: ReefState }) {
  const name = state.reef_id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const risk = (state.bleaching_risk_score * 100).toFixed(1);
  const colorClass = STATUS_COLORS[state.ecosystem_status] ?? 'bg-gray-800 border-gray-700';

  return (
    <div className={`rounded-lg border p-4 ${colorClass}`}>
      <h3 className="font-semibold text-lg mb-2">{name}</h3>
      <div className="text-3xl font-bold mb-1">{risk}%</div>
      <div className="text-sm uppercase tracking-wide mb-3">{state.ecosystem_status}</div>
      <div className="grid grid-cols-2 gap-2 text-sm opacity-80">
        <div>Temp: {state.water_temperature_c}°C</div>
        <div>DHW: {state.degree_heating_weeks}</div>
        <div>pH: {state.ph}</div>
        <div>Turbidity: {state.turbidity_ntu}</div>
      </div>
    </div>
  );
}

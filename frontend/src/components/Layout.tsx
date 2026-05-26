import { NavLink, Outlet } from 'react-router-dom';
import { Activity, FlaskConical, Brain, Search, BarChart3, Upload } from 'lucide-react';

const NAV = [
  { to: '/', label: 'Overview', icon: Activity },
  { to: '/simulate', label: 'Simulation', icon: FlaskConical },
  { to: '/stress', label: 'Stress Analysis', icon: BarChart3 },
  { to: '/knowledge', label: 'Knowledge Base', icon: Search },
  { to: '/agent', label: 'Agent', icon: Brain },
  { to: '/upload', label: 'Upload', icon: Upload },
];

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <nav className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 flex items-center h-14 gap-6">
          <span className="text-lg font-semibold text-cyan-400 mr-4">ReefTwin</span>
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 text-sm px-3 py-1.5 rounded transition ${
                  isActive
                    ? 'bg-cyan-900/40 text-cyan-300'
                    : 'text-gray-400 hover:text-gray-200'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}

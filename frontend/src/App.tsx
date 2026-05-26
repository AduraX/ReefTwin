import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import Simulation from './pages/Simulation';
import StressAnalysis from './pages/StressAnalysis';
import KnowledgeBase from './pages/KnowledgeBase';
import Agent from './pages/Agent';
import Upload from './pages/Upload';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 15000 } },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="simulate" element={<Simulation />} />
            <Route path="stress" element={<StressAnalysis />} />
            <Route path="knowledge" element={<KnowledgeBase />} />
            <Route path="agent" element={<Agent />} />
            <Route path="upload" element={<Upload />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

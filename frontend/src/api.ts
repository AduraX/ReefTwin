const BASE = '/api';

export interface ReefState {
  reef_id: string;
  timestamp: string;
  sst_celsius: number;
  water_temperature_c: number;
  ph: number;
  salinity_psu: number;
  turbidity_ntu: number;
  dissolved_oxygen_mg_l: number;
  degree_heating_weeks: number;
  bleaching_risk_score: number;
  risk_category: string;
  ecosystem_status: string;
}

export interface SimulationRequest {
  reef_id: string;
  temperature_delta_c: number;
  duration_days: number;
  turbidity_delta_pct: number;
  ph_delta: number;
}

export interface SimulationResult {
  reef_id: string;
  baseline_risk: number;
  projected_bleaching_risk: number;
  projected_ecosystem_status: string;
  scenario: SimulationRequest;
}

export interface RAGResult {
  answer: string;
  sources: { id: string; content: string; metadata: Record<string, string>; rrf_score?: number }[];
  model: string;
  retrieval_method: string;
}

export interface AgentResult {
  answer: string;
  tool_calls: { tool: string; input: Record<string, unknown>; result_summary: string }[];
  iterations: number;
  tokens: { input: number; output: number };
}

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

export async function fetchReefs(): Promise<ReefState[]> {
  const res = await fetch(`${BASE}/reefs`);
  const data = await res.json();
  return data.reefs ?? [];
}

export async function fetchReefState(reefId: string): Promise<ReefState> {
  const res = await fetch(`${BASE}/reefs/${reefId}/state`);
  if (!res.ok) throw new Error(`Reef not found: ${reefId}`);
  return res.json();
}

export async function runSimulation(req: SimulationRequest): Promise<SimulationResult> {
  const res = await fetch(`${BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function queryRAG(question: string, k = 3): Promise<RAGResult> {
  const res = await fetch(`${BASE}/rag`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, k }),
  });
  return res.json();
}

export async function queryAgent(query: string): Promise<AgentResult> {
  const res = await fetch(`${BASE}/agent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  return res.json();
}

export interface UploadResult {
  status: string;
  dataset_type: string;
  filename: string;
  format: string;
  target: string;
  rows: number;
}

export interface IngestResult {
  status: string;
  backend: string;
  total: number;
  valid: number;
  rejected: number;
  success_rate: number;
}

export async function ingestStream(
  events: Record<string, unknown>[],
): Promise<IngestResult> {
  const res = await fetch(`${BASE}/ingest/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? 'Ingest failed');
  }
  return res.json();
}

export async function uploadDataset(
  file: File,
  datasetType: 'iot' | 'noaa',
): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/datasets/upload?dataset_type=${datasetType}`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? 'Upload failed');
  }
  return res.json();
}

import { useState, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Upload as UploadIcon, FileUp, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadDataset, type UploadResult } from '../api';

const IOT_COLUMNS = [
  'reef_id', 'timestamp', 'water_temperature_c', 'ph',
  'salinity_psu', 'turbidity_ntu', 'dissolved_oxygen_mg_l',
];
const NOAA_COLUMNS = [
  'reef_id', 'date', 'sst_celsius', 'sst_anomaly_c',
  'hotspot_c', 'degree_heating_weeks', 'bleaching_alert_area',
];

export default function UploadPage() {
  const [datasetType, setDatasetType] = useState<'iot' | 'noaa'>('iot');
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string[][] | null>(null);

  const mutation = useMutation({
    mutationFn: (f: File) => uploadDataset(f, datasetType),
  });

  const handleFile = useCallback((f: File) => {
    const ext = f.name.split('.').pop()?.toLowerCase();
    if (!['csv', 'parquet', 'json'].includes(ext ?? '')) return;
    setFile(f);
    mutation.reset();

    if (ext === 'csv') {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const lines = text.split('\n').filter(Boolean).slice(0, 11);
        setPreview(lines.map((l) => l.split(',')));
      };
      reader.readAsText(f.slice(0, 8192));
    } else if (ext === 'json') {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          if (Array.isArray(data) && data.length > 0) {
            const keys = Object.keys(data[0]);
            const rows = data.slice(0, 10).map((r: Record<string, unknown>) => keys.map((k) => String(r[k] ?? '')));
            setPreview([keys, ...rows]);
          }
        } catch { setPreview(null); }
      };
      reader.readAsText(f.slice(0, 65536));
    } else {
      setPreview(null);
    }
  }, [datasetType, mutation]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const columns = datasetType === 'iot' ? IOT_COLUMNS : NOAA_COLUMNS;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Upload Reef Datasets</h1>
      <p className="text-gray-400">
        Upload CSV files to the <span className="text-cyan-400">bronze</span> data layer.
        After uploading, re-run the pipeline to update the twin state.
      </p>

      {/* Dataset type selector */}
      <div className="flex gap-3">
        {(['iot', 'noaa'] as const).map((t) => (
          <button
            key={t}
            onClick={() => { setDatasetType(t); setFile(null); setPreview(null); mutation.reset(); }}
            className={`px-4 py-2 rounded text-sm transition ${
              datasetType === t
                ? 'bg-cyan-900/50 text-cyan-300 border border-cyan-700'
                : 'bg-gray-800 text-gray-400 border border-gray-700 hover:text-gray-200'
            }`}
          >
            {t === 'iot' ? 'IoT Sensor Readings' : 'NOAA CRW Satellite Data'}
          </button>
        ))}
      </div>

      {/* Expected schema */}
      <div className="bg-gray-900 border border-gray-800 rounded p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">Expected columns</h3>
        <div className="flex flex-wrap gap-2">
          {columns.map((col) => (
            <span key={col} className="text-xs bg-gray-800 text-cyan-400 px-2 py-1 rounded font-mono">
              {col}
            </span>
          ))}
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-10 text-center transition ${
          dragOver
            ? 'border-cyan-500 bg-cyan-950/20'
            : 'border-gray-700 hover:border-gray-500'
        }`}
      >
        <UploadIcon className="mx-auto mb-3 text-gray-500" size={40} />
        <p className="text-gray-400 mb-2">Drag and drop a CSV, Parquet, or JSON file here, or</p>
        <label className="inline-flex items-center gap-2 px-4 py-2 bg-cyan-900/40 text-cyan-300 rounded cursor-pointer hover:bg-cyan-900/60 transition">
          <FileUp size={16} />
          Browse files
          <input
            type="file"
            accept=".csv,.parquet,.json"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          />
        </label>
        {file && (
          <p className="mt-3 text-sm text-gray-300">
            Selected: <span className="font-mono text-cyan-400">{file.name}</span>
            {' '}({(file.size / 1024).toFixed(1)} KB)
          </p>
        )}
      </div>

      {/* Parquet notice */}
      {file && file.name.endsWith('.parquet') && !preview && (
        <div className="bg-gray-900 border border-gray-800 rounded p-4 text-sm text-gray-400">
          Parquet files cannot be previewed in the browser. The server will validate columns on upload
          and convert to CSV for the bronze layer.
        </div>
      )}

      {/* Streaming info */}
      <div className="bg-gray-900 border border-gray-800 rounded p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">Streaming ingestion</h3>
        <p className="text-sm text-gray-400">
          For real-time IoT data, push JSON events directly to{' '}
          <span className="font-mono text-cyan-400">POST /ingest/stream</span>{' '}
          which publishes to Kafka/Redpanda topic{' '}
          <span className="font-mono text-cyan-400">reef.iot.readings</span>.
          Events are schema-validated; rejects go to the dead-letter queue.
        </p>
      </div>

      {/* Preview */}
      {preview && preview.length > 1 && (
        <div className="bg-gray-900 border border-gray-800 rounded overflow-x-auto">
          <h3 className="text-sm font-semibold text-gray-300 px-4 pt-3">Preview (first 10 rows)</h3>
          <table className="w-full text-xs font-mono mt-2">
            <thead>
              <tr className="border-b border-gray-800">
                {preview[0].map((h, i) => (
                  <th key={i} className="px-3 py-2 text-left text-cyan-400 font-medium">{h.trim()}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.slice(1).map((row, i) => (
                <tr key={i} className="border-b border-gray-800/50">
                  {row.map((cell, j) => (
                    <td key={j} className="px-3 py-1.5 text-gray-300">{cell.trim()}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Upload button */}
      {file && (
        <button
          onClick={() => mutation.mutate(file)}
          disabled={mutation.isPending}
          className="px-6 py-2.5 bg-cyan-700 text-white rounded font-medium hover:bg-cyan-600 disabled:opacity-50 transition"
        >
          {mutation.isPending ? 'Uploading...' : 'Upload to bronze layer'}
        </button>
      )}

      {/* Result */}
      {mutation.isSuccess && (
        <div className="flex items-start gap-3 bg-green-950/30 border border-green-800 rounded p-4">
          <CheckCircle className="text-green-400 mt-0.5 shrink-0" size={20} />
          <div>
            <p className="text-green-300 font-medium">Upload successful</p>
            <p className="text-sm text-gray-400 mt-1">
              {(mutation.data as UploadResult).rows} rows written to{' '}
              <span className="font-mono text-cyan-400">{(mutation.data as UploadResult).target}</span>
            </p>
            <p className="text-xs text-gray-500 mt-2">
              Next: re-run the pipeline to update the twin state.
            </p>
          </div>
        </div>
      )}

      {mutation.isError && (
        <div className="flex items-start gap-3 bg-red-950/30 border border-red-800 rounded p-4">
          <AlertCircle className="text-red-400 mt-0.5 shrink-0" size={20} />
          <div>
            <p className="text-red-300 font-medium">Upload failed</p>
            <p className="text-sm text-gray-400 mt-1">{(mutation.error as Error).message}</p>
          </div>
        </div>
      )}
    </div>
  );
}

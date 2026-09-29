import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');

const STATUS_COLORS = {
  OK: '#2ecc71',
  READY: '#2ecc71',
  MODEL_UNAVAILABLE: '#e67e22',
  DISABLED: '#7f8c8d',
  UNLOADED: '#7f8c8d',
  ERROR: '#e74c3c',
  QC_REJECTED: '#e74c3c',
  UNKNOWN: '#95a5a6',
};

function Badge({ status }) {
  const color = STATUS_COLORS[status] || '#95a5a6';
  return <span className="badge" style={{ background: color }}>{status}</span>;
}

async function fetchJSON(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function App() {
  const [models, setModels] = useState(null);
  const [pipeline, setPipeline] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [m, p] = await Promise.all([
        fetchJSON('/models'),
        fetchJSON('/pipeline/status'),
      ]);
      setModels(m);
      setPipeline(p);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { refresh(); }, []);

  return (
    <main dir="rtl">
      <h1>GoalReel</h1>
      <p>Virtual Cinema Engine — Model &amp; Pipeline Dashboard</p>

      <section>
        <div><b>THE EVENT IS IMMUTABLE</b><span>ONLY THE CAMERA VIEWPOINT MAY CHANGE</span></div>
        {models && (
          <div className="device">
            <span>Device: <b>{models.device}</b></span>
            <span>CUDA: <b>{models.cuda && models.cuda.available ? `yes (${models.cuda.name})` : 'no'}</b></span>
            <span>Torch: <b>{models.torch || 'n/a'}</b></span>
          </div>
        )}
        <button onClick={refresh} disabled={loading}>
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </section>

      {error && <p className="error">API error: {error}. Assurez-vous que <code>python server.py</code> tourne.</p>}

      <h2>Models</h2>
      <div className="grid">
        {models && models.models.map((m) => (
          <article key={m.name}>
            <h3>{m.name}<Badge status={m.status} /></h3>
            <small>stage: {m.stage}</small><br />
            <small>device: {m.device}</small><br />
            <small>checkpoint: {m.checkpoint_present ? 'present' : 'missing'}</small><br />
            <small>avg infer: {m.avg_infer_ms} ms</small><br />
            <small>enabled: {String(m.enabled)}</small>
          </article>
        ))}
      </div>

      <h2>Pipeline</h2>
      <div className="grid">
        {pipeline && pipeline.stages.map((s) => (
          <article key={s.id}>
            <h3>{s.label}<Badge status={s.status} /></h3>
            <small>model: {s.model || '—'}</small><br />
            {s.avg_infer_ms > 0 && <small>avg infer: {s.avg_infer_ms} ms</small>}
          </article>
        ))}
      </div>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);

import React, { useEffect, useState } from 'react';
import { fetchStatus } from '../api.js';

// Pastille de statut ComfyUI : vert = connecté, orange = hors ligne.
// L'application reste utilisable même si ComfyUI est indisponible.
export default function StatusPill() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  async function load() {
    try {
      const data = await fetchStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  const available = status && status.comfy && status.comfy.available;
  const cls = error ? 'pill error' : available ? 'pill ok' : 'pill warn';
  const label = error
    ? 'API indisponible'
    : available
    ? 'ComfyUI connecté'
    : 'ComfyUI hors ligne';

  return (
    <div className={cls} title={(status && status.comfy && status.comfy.message) || error || ''}>
      <span className="dot" />
      {label}
    </div>
  );
}

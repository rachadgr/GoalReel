// Client API du GoalReel Reel Studio.
// En dev, Vite proxifie /api vers le backend FastAPI.
// En prod, définir VITE_API_BASE si l'API est sur un autre domaine.

const API = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');

async function handle(res) {
  const contentType = res.headers.get('content-type') || '';
  let body = null;
  if (contentType.includes('application/json')) {
    body = await res.json().catch(() => null);
  }
  if (!res.ok) {
    const detail = body && (body.detail || body.message);
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return body;
}

export function apiUrl(path) {
  return `${API}${path}`;
}

export async function fetchHealth() {
  return handle(await fetch(apiUrl('/api/health')));
}

export async function fetchStatus() {
  return handle(await fetch(apiUrl('/api/status')));
}

export async function fetchConfig() {
  return handle(await fetch(apiUrl('/api/config')));
}

export async function fetchPresets() {
  return handle(await fetch(apiUrl('/api/presets')));
}

export async function fetchJobs(limit = 20) {
  return handle(await fetch(apiUrl(`/api/jobs?limit=${limit}`)));
}

export async function fetchJob(jobId) {
  return handle(await fetch(apiUrl(`/api/status/${jobId}`)));
}

export async function uploadImage(file) {
  const form = new FormData();
  form.append('image', file);
  const res = await fetch(apiUrl('/api/upload'), { method: 'POST', body: form });
  return handle(res);
}

export async function generate(params) {
  const form = new FormData();
  const add = (k, v) => {
    if (v !== undefined && v !== null && v !== '') form.append(k, v);
  };
  add('upload_id', params.uploadId);
  add('preset', params.preset);
  add('prompt', params.prompt);
  add('negative_prompt', params.negativePrompt);
  add('width', params.width);
  add('height', params.height);
  add('length', params.length);
  add('fps', params.fps);
  add('steps', params.steps);
  add('cfg', params.cfg);
  add('seed', params.seed);
  const res = await fetch(apiUrl('/api/generate'), { method: 'POST', body: form });
  return handle(res);
}

export function resultUrl(jobId) {
  return apiUrl(`/api/result/${jobId}`);
}

// Polling de progression avec backoff doux et annulation.
export function pollJob(jobId, onUpdate, { intervalMs = 2000, timeoutMs = 1800000 } = {}) {
  let cancelled = false;
  const started = Date.now();

  async function tick() {
    if (cancelled) return;
    try {
      const job = await fetchJob(jobId);
      onUpdate(job);
      if (job.state === 'COMPLETED' || job.state === 'FAILED') return;
      if (Date.now() - started > timeoutMs) {
        onUpdate({ ...job, state: 'FAILED', error: 'Timeout du suivi de génération.' });
        return;
      }
    } catch (err) {
      onUpdate({ job_id: jobId, state: 'FAILED', error: String(err.message || err) });
      return;
    }
    setTimeout(tick, intervalMs);
  }

  tick();
  return () => {
    cancelled = true;
  };
}

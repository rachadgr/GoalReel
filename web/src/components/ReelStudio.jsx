import React, { useEffect, useRef, useState } from 'react';
import {
  fetchConfig,
  fetchPresets,
  fetchStatus,
  generate,
  pollJob,
  resultUrl,
  uploadImage,
} from '../api.js';

const STATE_LABELS = {
  QUEUED: 'En file',
  UPLOADING: 'Envoi',
  SUBMITTING: 'Soumission',
  GENERATING: 'Génération',
  PROCESSING: 'Traitement',
  COMPLETED: 'Terminé',
  FAILED: 'Échec',
};

export default function ReelStudio() {
  const [presets, setPresets] = useState([]);
  const [defaultPreset, setDefaultPreset] = useState('');
  const [config, setConfig] = useState(null);
  const [comfy, setComfy] = useState(null);

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadInfo, setUploadInfo] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [presetId, setPresetId] = useState('');
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [params, setParams] = useState({
    width: '', height: '', length: '', fps: '', steps: '', cfg: '', seed: '',
  });

  const [job, setJob] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const stopPollRef = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        const [p, c, s] = await Promise.all([fetchPresets(), fetchConfig(), fetchStatus()]);
        setPresets(p.presets || []);
        setDefaultPreset(p.default || '');
        setPresetId(p.default || '');
        setConfig(c);
        setComfy(s.comfy || null);
        if (c.defaults) {
          setParams({
            width: c.defaults.width,
            height: c.defaults.height,
            length: c.defaults.length,
            fps: c.defaults.fps,
            steps: c.defaults.steps,
            cfg: c.defaults.cfg,
            seed: c.defaults.seed,
          });
        }
      } catch (err) {
        setError(`Impossible de charger la configuration : ${err.message || err}`);
      }
    })();
    return () => {
      if (stopPollRef.current) stopPollRef.current();
    };
  }, []);

  function onPickFile(e) {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    setFile(f);
    setUploadInfo(null);
    setJob(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function doUpload() {
    if (!file) return null;
    setUploading(true);
    setError(null);
    try {
      const info = await uploadImage(file);
      setUploadInfo(info);
      return info;
    } catch (err) {
      setError(`Upload échoué : ${err.message || err}`);
      return null;
    } finally {
      setUploading(false);
    }
  }

  async function doGenerate() {
    setError(null);
    if (!file && !uploadInfo) {
      setError('Sélectionnez d’abord une image.');
      return;
    }
    let info = uploadInfo;
    if (!info) {
      info = await doUpload();
      if (!info) return;
    }
    setGenerating(true);
    setJob({ state: 'QUEUED', progress: 0, message: 'Envoi de la requête…' });
    try {
      const res = await generate({
        uploadId: info.upload_id,
        preset: presetId,
        prompt,
        negativePrompt,
        width: params.width || undefined,
        height: params.height || undefined,
        length: params.length || undefined,
        fps: params.fps || undefined,
        steps: params.steps || undefined,
        cfg: params.cfg || undefined,
        seed: params.seed || undefined,
      });
      if (stopPollRef.current) stopPollRef.current();
      stopPollRef.current = pollJob(res.job_id, (j) => {
        setJob(j);
        if (j.state === 'COMPLETED' || j.state === 'FAILED') setGenerating(false);
      });
    } catch (err) {
      setError(`Génération impossible : ${err.message || err}`);
      setGenerating(false);
      setJob(null);
    }
  }

  const progress = job ? job.progress || 0 : 0;
  const stateLabel = job ? STATE_LABELS[job.state] || job.state : '';

  return (
    <div className="studio">
      {comfy && !comfy.available && (
        <div className="notice warn">
          <b>ComfyUI hors ligne.</b> Le studio reste accessible, mais la génération
          échouera jusqu’à ce que ComfyUI soit joignable sur{' '}
          <code>{config ? config.comfy_url : 'COMFY_URL'}</code>.
        </div>
      )}

      {/* 1. Image de départ */}
      <section className="card">
        <h2>1 · Image de départ</h2>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={onPickFile}
          hidden
        />
        <div className="dropzone" onClick={() => fileInputRef.current && fileInputRef.current.click()}>
          {previewUrl ? (
            <img src={previewUrl} alt="aperçu" className="preview" />
          ) : (
            <div className="dropzone-empty">
              <span className="dropzone-icon">📷</span>
              <p>Touchez pour choisir une photo</p>
              <small>PNG · JPEG · WEBP — max {config ? config.max_upload_mb : 20} Mo</small>
            </div>
          )}
        </div>
        {file && (
          <div className="row">
            <small className="muted">{file.name}</small>
            {!uploadInfo && (
              <button className="btn ghost" onClick={doUpload} disabled={uploading}>
                {uploading ? 'Envoi…' : 'Valider l’image'}
              </button>
            )}
            {uploadInfo && <span className="ok-tag">✓ Image prête</span>}
          </div>
        )}
      </section>

      {/* 2. Style / preset */}
      <section className="card">
        <h2>2 · Style football</h2>
        <div className="preset-grid">
          {presets.map((p) => (
            <button
              key={p.id}
              className={presetId === p.id ? 'preset active' : 'preset'}
              onClick={() => setPresetId(p.id)}
              title={p.description}
            >
              {p.label}
            </button>
          ))}
        </div>

        <label className="field">
          <span>Prompt (optionnel — override)</span>
          <textarea
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="ex. ralenti héroïque, stade illuminé, pluie fine…"
          />
        </label>

        <button className="btn ghost small" onClick={() => setAdvanced((v) => !v)}>
          {advanced ? 'Masquer les réglages avancés' : 'Réglages avancés'}
        </button>

        {advanced && (
          <div className="advanced">
            <label className="field">
              <span>Prompt négatif (optionnel)</span>
              <textarea
                rows={2}
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="éléments à éviter…"
              />
            </label>
            <div className="param-grid">
              {[
                ['width', 'Largeur'],
                ['height', 'Hauteur'],
                ['length', 'Frames'],
                ['fps', 'FPS'],
                ['steps', 'Steps'],
                ['cfg', 'CFG'],
                ['seed', 'Seed'],
              ].map(([key, label]) => (
                <label key={key} className="field">
                  <span>{label}</span>
                  <input
                    type="number"
                    value={params[key]}
                    onChange={(e) => setParams({ ...params, [key]: e.target.value })}
                  />
                </label>
              ))}
            </div>
            <small className="muted">
              Défaut T4 : 832×480 · 49 frames · 16 fps · 20 steps · cfg 5.0. Un rendu
              9:16 final est obtenu en post-traitement.
            </small>
          </div>
        )}
      </section>

      {/* 3. Génération */}
      <section className="card">
        <h2>3 · Génération</h2>
        <button className="btn primary big" onClick={doGenerate} disabled={generating || !file}>
          {generating ? 'Génération en cours…' : 'Générer le clip'}
        </button>

        {error && <p className="error">{error}</p>}

        {job && (
          <div className="progress-block">
            <div className="progress-head">
              <span className={`state state-${job.state}`}>{stateLabel}</span>
              <span>{progress}%</span>
            </div>
            <div className="progress-track">
              <div
                className={job.state === 'FAILED' ? 'progress-fill failed' : 'progress-fill'}
                style={{ width: `${progress}%` }}
              />
            </div>
            <small className="muted">{job.message}</small>
            {job.error && <p className="error">{job.error}</p>}
          </div>
        )}
      </section>

      {/* 4. Résultat */}
      {job && job.state === 'COMPLETED' && (
        <section className="card">
          <h2>4 · Résultat</h2>
          <video
            className="video"
            controls
            playsInline
            src={resultUrl(job.job_id)}
            poster={previewUrl || undefined}
          />
          <div className="row">
            <a className="btn primary" href={resultUrl(job.job_id)} download={`goalreel_${job.job_id}.mp4`}>
              ⬇ Télécharger le MP4
            </a>
            <button className="btn ghost" onClick={doGenerate} disabled={generating}>
              Régénérer
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

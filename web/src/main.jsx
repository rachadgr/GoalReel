import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

import ReelStudio from './components/ReelStudio.jsx';
import ModelDashboard from './components/ModelDashboard.jsx';
import StatusPill from './components/StatusPill.jsx';

function App() {
  const [tab, setTab] = useState('studio');

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">⚽</span>
          <div>
            <h1>GoalReel</h1>
            <p>AI Football Reel Studio</p>
          </div>
        </div>
        <StatusPill />
      </header>

      <nav className="tabs" role="tablist" aria-label="Navigation">
        <button
          role="tab"
          aria-selected={tab === 'studio'}
          className={tab === 'studio' ? 'tab active' : 'tab'}
          onClick={() => setTab('studio')}
        >
          Studio
        </button>
        <button
          role="tab"
          aria-selected={tab === 'models'}
          className={tab === 'models' ? 'tab active' : 'tab'}
          onClick={() => setTab('models')}
        >
          Modèles
        </button>
      </nav>

      <main className="app-main">
        {tab === 'studio' ? <ReelStudio /> : <ModelDashboard />}
      </main>

      <footer className="app-footer">
        <span>Wan 2.2 TI2V 5B · ComfyUI</span>
        <span>Mobile-first · 832×480 → 9:16</span>
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);

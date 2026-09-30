"""Pipeline d'analyse legacy (API conservée) — branché sur le ModelManager.

``run_analysis`` conserve sa signature historique mais utilise désormais le
ModelManager central et la couche ``services.ai`` (détection, suivi, reid,
depth, segmentation, pose). ``render_baseline`` est inchangé.
"""
import os
from pathlib import Path

from .core.ffprobe import probe
from .core.io import write_json
from .core.types import StageResult
from .source.analysis import analyze_video
from .scene.camera import CameraEstimator
from .events.football import FootballEventEngine
from .events.hero import score_hero, track_stats
from .source.ffmpeg import render_vertical, validate_output
from .qc.final import final_qc
from .models.manager import ModelManager
from .services.analysis_pipeline import AnalysisPipeline


def run_analysis(video, out, models='models', every=15, manager=None,
                 stages=None, max_frames=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manager = manager or ModelManager()

    info = probe(video)
    analysis = analyze_video(video, out / 'frames', every)
    stages_results = [
        StageResult('source_analysis', 'OK', 'Source analyzed',
                    evidence='VISIBLE_FROM_SOURCE',
                    metrics={k: v for k, v in analysis.items() if k != 'sampled_frames'},
                    artifacts=analysis['sampled_frames'][:10])
    ]

    # Analyse réelle par modèles via la couche unifiée.
    report = AnalysisPipeline(manager).run(video, every=every, max_frames=max_frames,
                                           stages=stages)
    for s in report['stages']:
        stages_results.append(StageResult(s['name'], s['status'], s['message'],
                                          evidence=s.get('evidence', 'UNKNOWN'),
                                          metrics=s.get('metrics', {})))

    camera = CameraEstimator().estimate(video)

    # Trajectoires réelles -> événements (sinon aucun événement inventé).
    tracks = {}
    rec_tracks = report.get('records', {}).get('tracks') or {}
    for frame_idx, tracked in rec_tracks.items():
        for t in tracked:
            tracks.setdefault(t['track_id'], []).append({'frame': frame_idx, **t})
    events = FootballEventEngine().infer(tracks=tracks)
    hero = score_hero(events, track_stats=track_stats(tracks))

    final_report = {
        'schema': 'goalreel.report.v2',
        'mode': 'FOOTAGE',
        'source': info.to_dict(),
        'stages': [s.to_dict() for s in stages_results],
        'model_status': manager.statuses(),
        'camera_estimation': camera.to_dict(),
        'event_timeline': events,
        'hero_moment': hero,
        'novel_view': {'status': manager.statuses().get('virtual_camera', {}).get('status', 'MODEL_UNAVAILABLE')},
        'identity': {'status': manager.statuses().get('reid', {}).get('status', 'UNKNOWN')},
        'warnings': [
            'Les statuts reflètent l\'environnement réel (aucun modèle simulé).',
        ],
    }
    write_json(out / 'report.json', final_report)
    return final_report


def render_baseline(video, out):
    p = render_vertical(video, out)
    return final_qc(p)

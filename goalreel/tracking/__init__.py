"""Couche de suivi GoalReel — continuité de sujet (Subject Continuity V2.1).

Ce paquet relie des fragments de suivi bruts (ByteTrack) en **identités
canoniques** lorsque des preuves mesurées le permettent. Il ne remplace ni le
détecteur ni le traqueur existants : c'est une couche d'analyse en aval.
"""
from .continuity import (
    FALLBACK_UNRESOLVED,
    POLICY,
    SCHEMA,
    Fragment,
    active_track_at,
    analyze_continuity,
    build_fragments,
    hero_identity,
    identity_containing,
    identity_frames,
    link_fragments,
    score_link,
)

__all__ = [
    "SCHEMA",
    "POLICY",
    "FALLBACK_UNRESOLVED",
    "Fragment",
    "build_fragments",
    "score_link",
    "link_fragments",
    "analyze_continuity",
    "hero_identity",
    "identity_containing",
    "identity_frames",
    "active_track_at",
]

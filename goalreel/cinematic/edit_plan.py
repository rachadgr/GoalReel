"""Cinematic Director V2 — artefact de plan de montage (machine-readable).

Ce module définit le **schéma** du plan de montage cinématique et les
constantes de contrat partagées par le director (``director.py``), le renderer
(:mod:`goalreel.cinematic.render_plan`) et la QA finale.

Principe de vérité
------------------
Le plan est **entièrement dérivé de la preuve réelle** disponible :

  * trajectoires suivies (ByteTrack sur détections YOLO réelles) ;
  * statistiques factuelles de trajectoire (persistance, amplitude, continuité) ;
  * chronologie d'événements inférés du mouvement réel ;
  * sélection héros multi-preuves (``multi_evidence_v2``) ;
  * détections de ballon RÉELLES (classe COCO ``sports ball``) — jamais simulées ;
  * frontières de plan RÉELLES mesurées dans la source (pic de différence
    inter-frame) ;
  * géométrie réelle du groupe de sujets (étendue horizontale) et échelle de
    plan réellement mesurée (taille des bbox dominantes).

Aucun événement (but, tir, passe, identité joueur, possession, arbitre) n'est
inventé. Quand la preuve est absente ou ambiguë, la phase est **omise** (avec la
raison enregistrée) ou sa confiance est **dégradée** — jamais fabriquée.

Vitesse
-------
``SpeedCurve`` exprime une **vitesse de lecture** (``factor``) : ``1.0`` = temps
réel, ``< 1.0`` = ralenti, ``> 1.0`` = accélération. Le rendu conserve la durée
totale et la synchronisation audio en normalisant le budget temporel
(les accélérations « paient » les ralentis) — voir ``render_plan.py``.

Déterminisme
------------
Pour une même entrée de preuve, ``build_edit_plan`` produit **toujours** le même
plan : toutes les itérations sur des dictionnaires sont triées, les clés de
tri/classement sont totales et stables, les valeurs flottantes sont arrondies de
façon reproductible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Phases narratives (vocabulaire cinématique).
# ---------------------------------------------------------------------------
HOOK = "HOOK"
BUILD_UP = "BUILD_UP"
ANTICIPATION = "ANTICIPATION"
ACTION = "ACTION"
HERO = "HERO"
REACTION = "REACTION"
CELEBRATION = "CELEBRATION"
SECONDARY = "SECONDARY"
CLIMAX = "CLIMAX"
FINAL_HERO = "FINAL_HERO"
OUTRO = "OUTRO"

# Ordre canonique de la structure cible (documentation / tests / audit).
PHASE_ORDER: tuple[str, ...] = (
    HOOK, BUILD_UP, ANTICIPATION, ACTION, HERO, REACTION, CELEBRATION,
    SECONDARY, CLIMAX, FINAL_HERO, OUTRO,
)

# ---------------------------------------------------------------------------
# Comportements caméra (sous-ensemble *contrôlé* de la caméra HERO_TRACK).
# Chaque comportement est une consigne de cadrage 9:16 exprimée en fractions de
# la hauteur source ; il n'invente aucun pixel (simple recadrage virtuel).
# ---------------------------------------------------------------------------
CAM_STABLE_FOLLOW = "STABLE_FOLLOW"
CAM_ANTICIPATION_FOLLOW = "ANTICIPATION_FOLLOW"
CAM_PUSH_IN = "PUSH_IN"
CAM_PULL_BACK = "PULL_BACK"
CAM_REACTION_FRAMING = "REACTION_FRAMING"
CAM_CELEBRATION_FRAMING = "CELEBRATION_FRAMING"
CAM_FINAL_HERO_FRAMING = "FINAL_HERO_FRAMING"

CAMERA_BEHAVIORS: tuple[str, ...] = (
    CAM_STABLE_FOLLOW, CAM_ANTICIPATION_FOLLOW, CAM_PUSH_IN, CAM_PULL_BACK,
    CAM_REACTION_FRAMING, CAM_CELEBRATION_FRAMING, CAM_FINAL_HERO_FRAMING,
)

# ---------------------------------------------------------------------------
# Types de plans (variété de cadrage issue de la MÊME source réelle).
# ---------------------------------------------------------------------------
SHOT_WIDE = "WIDE"
SHOT_MEDIUM = "MEDIUM"
SHOT_HERO_MEDIUM = "HERO_MEDIUM"
SHOT_CLOSE = "CLOSE"
SHOT_LOW_FEELING = "LOW_FEELING"
SHOT_REACTION = "REACTION"
SHOT_CELEBRATION = "CELEBRATION"
SHOT_FINAL_HERO = "FINAL_HERO"
SHOT_OUTRO = "OUTRO"
SHOT_HOOK = "HOOK"

# ---------------------------------------------------------------------------
# Transitions (restreintes, motivées par les frontières événementielles).
# ---------------------------------------------------------------------------
CUT_HARD = "HARD_CUT"
CUT_DISSOLVE = "DISSOLVE"          # fondu court uniquement si justifié
CUT_FADE_FROM_BLACK = "FADE_FROM_BLACK"
CUT_FADE_TO_BLACK = "FADE_TO_BLACK"

SUPPORTED_TRANSITIONS: tuple[str, ...] = (
    CUT_HARD, CUT_DISSOLVE, CUT_FADE_FROM_BLACK, CUT_FADE_TO_BLACK,
)

# ---------------------------------------------------------------------------
# Vitesses (comportement par phase).
# ---------------------------------------------------------------------------
SPEED_NORMAL = "NORMAL"
SPEED_SLIGHT_FAST = "SLIGHT_FAST"          # légère accélération (rythme)
SPEED_ANTICIPATION = "ANTICIPATION_SLOW"
SPEED_HERO_SLOW = "HERO_SLOW"              # ralenti héros (justifié par l'événement)
SPEED_RECOVERY = "RECOVERY"

# Bornes de vitesse de lecture : (factor_min, factor_max) par comportement.
# ``1.0`` = temps réel ; ``< 1`` = ralenti ; ``> 1`` = accélération.
SPEED_RANGE: dict[str, tuple[float, float]] = {
    SPEED_NORMAL: (1.0, 1.0),
    SPEED_SLIGHT_FAST: (1.04, 1.12),
    SPEED_ANTICIPATION: (0.76, 0.90),
    SPEED_HERO_SLOW: (0.58, 0.76),
    SPEED_RECOVERY: (0.88, 1.0),
}

# Bornes dures de sécurité (aucune phase ne peut sortir de ces bornes).
SPEED_MIN = 0.55
SPEED_MAX = 1.15

# ---------------------------------------------------------------------------
# Cadrage 9:16 : fraction de la HAUTEUR source utilisée par la fenêtre de crop.
# ``1.0`` = toute la hauteur (plan large) ; ``< 1`` = plan plus serré.
#
# Borné volontairement à ``CROP_FRACTION_MIN`` pour éviter le sur-zoom et
# préserver le détail réel de la source (pas de sur-agrandissement).
# ---------------------------------------------------------------------------
CROP_FRACTION: dict[str, float] = {
    SHOT_HOOK: 0.98,
    SHOT_WIDE: 1.0,
    SHOT_MEDIUM: 0.90,
    SHOT_HERO_MEDIUM: 0.84,
    SHOT_CLOSE: 0.74,
    SHOT_LOW_FEELING: 0.86,
    SHOT_REACTION: 0.88,
    SHOT_CELEBRATION: 0.80,
    SHOT_FINAL_HERO: 0.86,
    SHOT_OUTRO: 0.95,
}
CROP_FRACTION_MIN = 0.72
CROP_FRACTION_MAX = 1.0

# Position verticale cible du sujet dans le cadre (fraction de la hauteur du
# cadre). ``0.5`` = centré ; ``> 0.5`` = sujet plus haut (cadrage « low-feeling »).
SUBJECT_Y_BIAS: dict[str, float] = {
    SHOT_HOOK: 0.55,
    SHOT_WIDE: 0.55,
    SHOT_MEDIUM: 0.55,
    SHOT_HERO_MEDIUM: 0.52,
    SHOT_CLOSE: 0.50,
    SHOT_LOW_FEELING: 0.62,
    SHOT_REACTION: 0.48,
    SHOT_CELEBRATION: 0.50,
    SHOT_FINAL_HERO: 0.50,
    SHOT_OUTRO: 0.52,
}

# Delta de crop (signé) appliqué de ``crop_start`` vers ``crop_end`` selon le
# comportement caméra. Positif => le plan se resserre (push-in).
CAMERA_CROP_DELTA: dict[str, float] = {
    CAM_STABLE_FOLLOW: 0.0,
    CAM_ANTICIPATION_FOLLOW: 0.02,
    CAM_PUSH_IN: 0.08,
    CAM_PULL_BACK: -0.08,
    CAM_REACTION_FRAMING: -0.04,
    CAM_CELEBRATION_FRAMING: 0.02,
    CAM_FINAL_HERO_FRAMING: 0.02,
}

# ---------------------------------------------------------------------------
# Détection de coupe (preuve réelle mesurée sur la source).
# ---------------------------------------------------------------------------
CUT_DIFF_THRESHOLD = 18.0

# Une trajectoire « taille personne » : le sujet suivi en plan large est un
# joueur, pas un blob large de premier plan.
PERSON_MAX_WIDTH = 90.0
PERSON_MAX_HEIGHT = 140.0

# ---------------------------------------------------------------------------
# Bornes de montage (durées en frames à 30 fps).
# ---------------------------------------------------------------------------
MIN_SHOT_FRAMES = 12           # 0.4 s — aucun plan plus court n'est conservé
HOOK_FRAMES = 30               # 1.0 s
ACTION_LEN = 36                # 1.2 s
ANTICIPATION_LEN = 30          # 1.0 s
HERO_PRE = 29                  # ~1.0 s avant le pic héros
HERO_POST = 67                 # ~2.2 s après le pic héros
REACTION_MIN = 24
CLIMAX_HALF = 18               # demi-fenêtre du climax autour du pic de mouvement
OUTRO_FRAMES = 18              # 0.6 s
MIN_SEGMENT_FRAMES = 24        # segments plus courts fusionnés avec le précédent
BEAT_SPLIT_MIN_SEGMENT = 60    # au-delà, un segment est scindé sur un vrai beat


def round_scale(value: float, digits: int = 6) -> float:
    """Arrondi reproductible (stabilité du déterminisme JSON)."""
    return float(round(float(value), digits))


@dataclass
class SpeedCurve:
    """Courbe de vitesse d'un plan : facteurs bornés aux frontières.

    ``start_factor`` / ``end_factor`` encadrent une interpolation lisse interne
    (``easing``) ; ``behavior`` documente l'intention. Les facteurs sont
    **bornés** dans ``[SPEED_MIN, SPEED_MAX]``.
    """

    behavior: str = SPEED_NORMAL
    start_factor: float = 1.0
    end_factor: float = 1.0
    easing: str = "linear"
    reason: str = "NO_SPEED_EVIDENCE"
    fallback: str | None = None

    def __post_init__(self) -> None:
        self.start_factor = float(min(SPEED_MAX, max(SPEED_MIN, self.start_factor)))
        self.end_factor = float(min(SPEED_MAX, max(SPEED_MIN, self.end_factor)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior": self.behavior,
            "start_factor": round_scale(self.start_factor),
            "end_factor": round_scale(self.end_factor),
            "easing": self.easing,
            "reason": self.reason,
            "fallback": self.fallback,
        }


@dataclass
class Shot:
    """Un plan du montage cinématique (phase + timing + caméra + vitesse)."""

    shot_id: str
    phase: str
    shot_type: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    duration_frames: int
    selected_track: int | None
    camera_behavior: str
    transition_in: str
    transition_out: str
    crop_start: float
    crop_end: float
    crop_easing: str
    subject_y_bias: float
    speed_curve: dict[str, Any]
    priority: int
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    fallback: str | None = None
    boundary: str = "CONTINUOUS"
    # Cibles caméra réelles (frame -> centre source réel suivi), preuve suivie.
    targets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("crop_start", "crop_end", "subject_y_bias", "confidence",
                    "start_time_s", "end_time_s", "duration_s"):
            d[key] = round_scale(d[key])
        return d

    @property
    def frame_span(self) -> tuple[int, int]:
        return int(self.start_frame), int(self.end_frame)


@dataclass
class EditPlan:
    """Plan de montage complet, déterministe et explicable."""

    schema: str = "goalreel.cinematic_edit_plan.v1"
    source_video: str = ""
    source_width: int = 0
    source_height: int = 0
    fps: float = 30.0
    total_frames: int = 0
    total_duration_s: float = 0.0
    hero_track: int | None = None
    hero_event_id: str | None = None
    hero_method: str | None = None
    followed_track: int | None = None
    policy: str = "NO_EVENT_WITHOUT_EVIDENCE"
    # Contrat événement -> héros -> caméra (hero_track == followed_track).
    hero_camera_contract: bool = False
    duration_preserved: bool = True
    claims: dict[str, Any] = field(default_factory=dict)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    evidence_used: dict[str, Any] = field(default_factory=dict)
    time_map: dict[str, Any] = field(default_factory=dict)
    phases_present: list[str] = field(default_factory=list)
    phases_absent: list[dict[str, Any]] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["shots"] = [s.to_dict() if isinstance(s, Shot) else s for s in self.shots]
        return d

    @property
    def shot_count(self) -> int:
        return len(self.shots)

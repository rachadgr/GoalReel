"""Presets de génération football pour le GoalReel Reel Studio.

Chaque preset produit une paire ``(positive, negative)`` de prompts adaptée au
modèle Wan 2.2 TI2V 5B. La base positive commune et la base négative commune
sont fusionnées avec la description spécifique du preset, afin de garantir :

* un rendu **football réaliste** (stade pro, anatomie naturelle, kit cohérent,
  mouvement de caméra cinématographique, éclairage naturel) ;
* l'évitement systématique des artefacts classiques (membres en trop, joueurs
  dupliqués, visage déformé, maillot déformé, scintillement, fond instable,
  aspect CGI, texte, watermark).

Le module ne dépend d'aucune bibliothèque lourde et est donc testable hors GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Bases communes
# ---------------------------------------------------------------------------
BASE_POSITIVE = (
    "realistic football footage, professional stadium, natural anatomy, "
    "consistent player appearance, realistic kit, cinematic camera movement, "
    "natural lighting, realistic motion"
)

BASE_NEGATIVE = (
    "extra limbs, duplicate players, distorted face, warped jersey, flicker, "
    "unstable background, artificial CGI look, text, watermark"
)


@dataclass(frozen=True)
class Preset:
    """Un preset de génération football."""

    id: str
    label: str
    description: str
    positive: str
    negative: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "positive": self.positive,
            "negative": self.negative,
        }


def _build(
    preset_id: str,
    label: str,
    description: str,
    positive_action: str,
    negative_action: str = "",
) -> Preset:
    positive = ", ".join(p for p in (positive_action.strip(), BASE_POSITIVE) if p)
    negative = ", ".join(p for p in (negative_action.strip(), BASE_NEGATIVE) if p)
    return Preset(
        id=preset_id,
        label=label,
        description=description,
        positive=positive,
        negative=negative,
    )


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
_PRESETS: dict[str, Preset] = {
    p.id: p
    for p in (
        _build(
            "cinematic_football",
            "Cinematic Football",
            "Plan cinématographique général d'un joueur en action.",
            "cinematic football player in action on a professional pitch, "
            "dynamic tracking shot, shallow depth of field",
        ),
        _build(
            "goal_celebration",
            "Goal Celebration",
            "Célébration de but : joie, mouvement, ambiance stade.",
            "football player celebrating a goal, arms raised, crowd in the "
            "background, emotional, dynamic motion",
            "empty stadium, sad expression",
        ),
        _build(
            "dribble",
            "Dribble",
            "Dribble serré : contrôle de balle, changements de direction.",
            "football player dribbling the ball past opponents, close ball "
            "control, quick direction changes, low camera angle",
        ),
        _build(
            "sprint",
            "Sprint",
            "Course rapide : vitesse, foulée naturelle, mouvement latéral.",
            "football player sprinting at full speed, natural running stride, "
            "motion blur on the background, side tracking camera",
        ),
        _build(
            "shot_on_goal",
            "Shot on Goal",
            "Frappe au but : puissance, trajectoire de balle, suivi.",
            "football player striking the ball towards the goal, powerful "
            "shot, ball trajectory, dynamic follow-through",
        ),
        _build(
            "goalkeeper_save",
            "Goalkeeper Save",
            "Arrêt du gardien : plongeon, extension, réflexes.",
            "goalkeeper diving to make a save, gloves reaching the ball, "
            "athletic extension, slow-motion feel",
        ),
        _build(
            "player_introduction",
            "Player Introduction",
            "Présentation de joueur : pose, regard caméra, kit détaillé.",
            "football player standing on the pitch facing the camera, "
            "confident pose, detailed kit, stadium lights",
            "multiple players, crowd chaos",
        ),
        _build(
            "slow_motion_hero",
            "Slow Motion Hero",
            "Plan héroïque au ralenti : mise en valeur, dramatisation.",
            "heroic slow motion shot of a football player, dramatic lighting, "
            "detailed motion, cinematic depth, epic moment",
        ),
    )
}

DEFAULT_PRESET_ID = "cinematic_football"


def list_presets() -> list[Preset]:
    """Retourne tous les presets (ordre stable)."""
    return list(_PRESETS.values())


def get_preset(preset_id: str | None) -> Preset:
    """Retourne un preset par identifiant (fallback sur le preset par défaut)."""
    if preset_id and preset_id in _PRESETS:
        return _PRESETS[preset_id]
    return _PRESETS[DEFAULT_PRESET_ID]


def preset_ids() -> list[str]:
    return list(_PRESETS.keys())


def build_prompts(
    preset_id: str | None = None,
    positive_override: str | None = None,
    negative_override: str | None = None,
) -> tuple[str, str]:
    """Construit la paire de prompts finale.

    Un override non vide remplace la partie spécifique du preset tout en
    conservant les bases communes (réalisme / anti-artefacts).
    """
    preset = get_preset(preset_id)

    if positive_override and positive_override.strip():
        positive = ", ".join([positive_override.strip(), BASE_POSITIVE])
    else:
        positive = preset.positive

    if negative_override and negative_override.strip():
        negative = ", ".join([negative_override.strip(), BASE_NEGATIVE])
    else:
        negative = preset.negative

    return positive, negative

"""Générateur de workflow ComfyUI pour **Wan 2.2 TI2V 5B**.

Ce module construit le graphe JSON soumis à ``POST /prompt`` de ComfyUI. Il est
**indépendant** du client HTTP (:mod:`goalreel.studio.comfyui`) : cela permet de
tester et valider le workflow sans aucun GPU ni serveur ComfyUI.

Le graphe reproduit **exactement** le workflow validé de l'environnement cible :

    LoadImage → UNETLoader → ModelSamplingSD3 → CLIPLoader → VAELoader
      → CLIPTextEncode (pos/neg) → Wan22ImageToVideoLatent
      → KSampler → VAEDecode → CreateVideo → SaveVideo

Contraintes respectées :

* ``SaveVideo`` utilise le format **plat** ``{"format": "mp4", "codec": "h264"}``
  (pas d'ancien ``format`` imbriqué) ;
* aucun checkpoint inventé : les noms de modèles sont injectés depuis la
  configuration (``wan2.2_ti2v_5B_fp16.safetensors``, ``wan2.2_vae.safetensors``,
  ``umt5_xxl_fp8_e4m3fn_scaled.safetensors`` par défaut) ;
* tous les paramètres de génération (largeur, hauteur, longueur, fps, steps,
  cfg, seed, prompts) sont modifiables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .presets import build_prompts
from .studio_config import ComfyModels, GenerationDefaults

#: Identifiants de nœuds du graphe (stables, réutilisés dans tout le module).
NODE_LOAD_IMAGE = "1"
NODE_UNET_LOADER = "2"
NODE_MODEL_SAMPLING = "3"
NODE_CLIP_LOADER = "4"
NODE_VAE_LOADER = "5"
NODE_POSITIVE = "6"
NODE_NEGATIVE = "7"
NODE_LATENT = "8"
NODE_SAMPLER = "9"
NODE_VAE_DECODE = "10"
NODE_CREATE_VIDEO = "11"
NODE_SAVE_VIDEO = "12"


@dataclass
class Wan22Params:
    """Paramètres d'une génération Wan 2.2 TI2V 5B."""

    image_name: str
    positive_prompt: str
    negative_prompt: str
    width: int = 832
    height: int = 480
    length: int = 49
    fps: float = 16.0
    steps: int = 20
    cfg: float = 5.0
    seed: int = 123456789
    shift: float = 3.0
    sampler_name: str = "euler"
    scheduler: str = "simple"
    denoise: float = 1.0
    batch_size: int = 1
    filename_prefix: str = "GoalReel/Wan22"
    weight_dtype: str = "fp8_e4m3fn"
    models: ComfyModels = field(default_factory=ComfyModels)


def default_params(
    image_name: str,
    preset_id: str | None = None,
    positive_override: str | None = None,
    negative_override: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    fps: float | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    seed: int | None = None,
    models: ComfyModels | None = None,
    filename_prefix: str = "GoalReel/Wan22",
) -> Wan22Params:
    """Construit des :class:`Wan22Params` en partant des défauts (T4) + preset."""
    d = GenerationDefaults()
    positive, negative = build_prompts(preset_id, positive_override, negative_override)
    return Wan22Params(
        image_name=image_name,
        positive_prompt=positive,
        negative_prompt=negative,
        width=width if width is not None else d.width,
        height=height if height is not None else d.height,
        length=length if length is not None else d.length,
        fps=fps if fps is not None else d.fps,
        steps=steps if steps is not None else d.steps,
        cfg=cfg if cfg is not None else d.cfg,
        seed=seed if seed is not None else d.seed,
        shift=d.shift,
        sampler_name=d.sampler_name,
        scheduler=d.scheduler,
        models=models or ComfyModels(),
        filename_prefix=filename_prefix,
    )


def build_workflow(params: Wan22Params) -> dict[str, Any]:
    """Construit le graphe de workflow ComfyUI (API format)."""
    m = params.models
    return {
        NODE_LOAD_IMAGE: {
            "class_type": "LoadImage",
            "inputs": {"image": params.image_name, "upload": "image"},
        },
        NODE_UNET_LOADER: {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": m.diffusion,
                "weight_dtype": params.weight_dtype,
            },
        },
        NODE_MODEL_SAMPLING: {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": [NODE_UNET_LOADER, 0], "shift": params.shift},
        },
        NODE_CLIP_LOADER: {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": m.text_encoder, "type": "wan"},
        },
        NODE_VAE_LOADER: {
            "class_type": "VAELoader",
            "inputs": {"vae_name": m.vae},
        },
        NODE_POSITIVE: {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": params.positive_prompt,
                "clip": [NODE_CLIP_LOADER, 0],
            },
        },
        NODE_NEGATIVE: {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": params.negative_prompt,
                "clip": [NODE_CLIP_LOADER, 0],
            },
        },
        NODE_LATENT: {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": [NODE_VAE_LOADER, 0],
                "width": params.width,
                "height": params.height,
                "length": params.length,
                "batch_size": params.batch_size,
                "start_image": [NODE_LOAD_IMAGE, 0],
            },
        },
        NODE_SAMPLER: {
            "class_type": "KSampler",
            "inputs": {
                "model": [NODE_MODEL_SAMPLING, 0],
                "seed": params.seed,
                "steps": params.steps,
                "cfg": params.cfg,
                "sampler_name": params.sampler_name,
                "scheduler": params.scheduler,
                "positive": [NODE_POSITIVE, 0],
                "negative": [NODE_NEGATIVE, 0],
                "latent_image": [NODE_LATENT, 0],
                "denoise": params.denoise,
            },
        },
        NODE_VAE_DECODE: {
            "class_type": "VAEDecode",
            "inputs": {"samples": [NODE_SAMPLER, 0], "vae": [NODE_VAE_LOADER, 0]},
        },
        NODE_CREATE_VIDEO: {
            "class_type": "CreateVideo",
            "inputs": {"images": [NODE_VAE_DECODE, 0], "fps": params.fps},
        },
        NODE_SAVE_VIDEO: {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [NODE_CREATE_VIDEO, 0],
                "filename_prefix": params.filename_prefix,
                # Format PLAT requis par SaveVideo (pas de nested format).
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


# ---------------------------------------------------------------------------
# Validation structurelle (utilisée par les tests et au démarrage)
# ---------------------------------------------------------------------------
REQUIRED_NODES: dict[str, str] = {
    NODE_LOAD_IMAGE: "LoadImage",
    NODE_UNET_LOADER: "UNETLoader",
    NODE_MODEL_SAMPLING: "ModelSamplingSD3",
    NODE_CLIP_LOADER: "CLIPLoader",
    NODE_VAE_LOADER: "VAELoader",
    NODE_POSITIVE: "CLIPTextEncode",
    NODE_NEGATIVE: "CLIPTextEncode",
    NODE_LATENT: "Wan22ImageToVideoLatent",
    NODE_SAMPLER: "KSampler",
    NODE_VAE_DECODE: "VAEDecode",
    NODE_CREATE_VIDEO: "CreateVideo",
    NODE_SAVE_VIDEO: "SaveVideo",
}


def validate_workflow(workflow: dict[str, Any]) -> list[str]:
    """Valide la structure du workflow et retourne la liste des erreurs.

    Vérifie la présence et le type de chaque nœud requis, l'usage du format plat
    pour ``SaveVideo``, et l'absence de référence de nœud pendante.
    """
    errors: list[str] = []

    for node_id, class_type in REQUIRED_NODES.items():
        node = workflow.get(node_id)
        if node is None:
            errors.append(f"missing node {node_id} ({class_type})")
            continue
        if node.get("class_type") != class_type:
            errors.append(
                f"node {node_id} expected class_type {class_type!r}, "
                f"got {node.get('class_type')!r}"
            )

    # SaveVideo : format plat obligatoire
    save = workflow.get(NODE_SAVE_VIDEO, {})
    save_inputs = save.get("inputs", {})
    if save_inputs.get("format") != "mp4":
        errors.append("SaveVideo.format must be 'mp4'")
    if save_inputs.get("codec") != "h264":
        errors.append("SaveVideo.codec must be 'h264'")
    if isinstance(save_inputs.get("format"), dict):
        errors.append("SaveVideo.format must be a flat string, not nested")

    # Références de nœuds : [node_id, slot] -> node_id doit exister
    valid_ids = set(workflow.keys())
    for node_id, node in workflow.items():
        for key, value in node.get("inputs", {}).items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
            ):
                if value[0] not in valid_ids:
                    errors.append(
                        f"node {node_id}.{key} references unknown node {value[0]!r}"
                    )

    return errors

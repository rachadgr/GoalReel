"""Tests du générateur de workflow Wan 2.2 TI2V 5B."""
from __future__ import annotations

from goalreel.studio import wan22
from goalreel.studio.studio_config import ComfyModels


def _wf(image_name="example.png", **kwargs):
    params = wan22.default_params(image_name=image_name, **kwargs)
    return wan22.build_workflow(params)


def test_workflow_is_valid_by_default():
    wf = _wf()
    assert wan22.validate_workflow(wf) == []


def test_all_required_nodes_present_with_right_class_types():
    wf = _wf()
    for node_id, class_type in wan22.REQUIRED_NODES.items():
        assert node_id in wf
        assert wf[node_id]["class_type"] == class_type


def test_savevideo_uses_flat_mp4_h264_format():
    wf = _wf()
    inputs = wf[wan22.NODE_SAVE_VIDEO]["inputs"]
    assert inputs["format"] == "mp4"
    assert inputs["codec"] == "h264"
    assert not isinstance(inputs["format"], dict)


def test_model_filenames_injected_from_config():
    models = ComfyModels(
        diffusion="d.safetensors", vae="v.safetensors", text_encoder="t.safetensors"
    )
    wf = _wf(models=models)
    assert wf[wan22.NODE_UNET_LOADER]["inputs"]["unet_name"] == "d.safetensors"
    assert wf[wan22.NODE_VAE_LOADER]["inputs"]["vae_name"] == "v.safetensors"
    assert wf[wan22.NODE_CLIP_LOADER]["inputs"]["clip_name"] == "t.safetensors"


def test_parameters_are_configurable():
    wf = _wf(width=512, height=288, length=33, fps=12.0, steps=8, cfg=7.5, seed=42)
    latent = wf[wan22.NODE_LATENT]["inputs"]
    assert latent["width"] == 512
    assert latent["height"] == 288
    assert latent["length"] == 33
    sampler = wf[wan22.NODE_SAMPLER]["inputs"]
    assert sampler["steps"] == 8
    assert sampler["cfg"] == 7.5
    assert sampler["seed"] == 42
    assert wf[wan22.NODE_CREATE_VIDEO]["inputs"]["fps"] == 12.0


def test_image_name_is_used_as_start_image():
    wf = _wf(image_name="my_upload.png")
    assert wf[wan22.NODE_LOAD_IMAGE]["inputs"]["image"] == "my_upload.png"
    assert wf[wan22.NODE_LATENT]["inputs"]["start_image"] == [wan22.NODE_LOAD_IMAGE, 0]


def test_positive_and_negative_are_wired_to_sampler():
    wf = _wf()
    sampler = wf[wan22.NODE_SAMPLER]["inputs"]
    assert sampler["positive"] == [wan22.NODE_POSITIVE, 0]
    assert sampler["negative"] == [wan22.NODE_NEGATIVE, 0]


def test_validation_detects_missing_node():
    wf = _wf()
    del wf[wan22.NODE_SAVE_VIDEO]
    errors = wan22.validate_workflow(wf)
    assert any("missing node 12" in e for e in errors)


def test_validation_detects_bad_savevideo_format():
    wf = _wf()
    wf[wan22.NODE_SAVE_VIDEO]["inputs"]["format"] = {"format": "mp4"}  # ancien nested
    errors = wan22.validate_workflow(wf)
    assert any("flat" in e or "mp4" in e for e in errors)


def test_validation_detects_dangling_reference():
    wf = _wf()
    wf[wan22.NODE_SAMPLER]["inputs"]["model"] = ["999", 0]
    errors = wan22.validate_workflow(wf)
    assert any("unknown node '999'" in e for e in errors)


def test_workflow_is_json_serializable():
    import json

    wf = _wf()
    json.dumps(wf)  # ne doit pas lever

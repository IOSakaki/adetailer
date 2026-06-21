import importlib.util
from pathlib import Path


def load_common_module():
    common_path = (
        Path(__file__).parents[1] / "adetailer_custom_controlnet_ext" / "common.py"
    )
    spec = importlib.util.spec_from_file_location("controlnet_ext_common", common_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cn_model_regex_matches_anytest_names():
    cn_model_regex = load_common_module().cn_model_regex

    assert cn_model_regex.search("CN-anytest_v3-50000_fp16.safetensors")
    assert cn_model_regex.search("anima-lllite-any-test-like-v2.safetensors")
    assert cn_model_regex.search("some_any_test_model.safetensors")

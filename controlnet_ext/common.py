import re

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
}
# Keep existing filters and include Anytest family models.
_names = [*cn_model_module, "union", "anytest"]
cn_model_regex = re.compile("|".join(_names), flags=re.IGNORECASE)

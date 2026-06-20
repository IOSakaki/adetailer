import re

cn_model_module = {
    "noobai": "inpaint_noobai",
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
}
# Keep existing filters and include Anytest family models.
_patterns = [*map(re.escape, cn_model_module), "union", r"any[-_]?test"]
cn_model_regex = re.compile("|".join(_patterns), flags=re.IGNORECASE)

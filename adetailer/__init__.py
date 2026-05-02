from .__version__ import __version__
from .args import ALL_ARGS, ADetailerArgs
from .common import PredictOutput, get_models

ADETAILER = "ADetailer"

__all__ = [
    "ADETAILER",
    "ALL_ARGS",
    "ADetailerArgs",
    "PredictOutput",
    "__version__",
    "get_models",
    "mediapipe_predict",
    "sam3_predict",
    "SAM3DependencyError",
    "ultralytics_predict",
]


def mediapipe_predict(*args, **kwargs):
    from .mediapipe import mediapipe_predict as _impl

    return _impl(*args, **kwargs)


def ultralytics_predict(*args, **kwargs):
    from .ultralytics import ultralytics_predict as _impl

    return _impl(*args, **kwargs)


def sam3_predict(*args, **kwargs):
    from .sam3 import sam3_predict as _impl

    return _impl(*args, **kwargs)


class SAM3DependencyError(RuntimeError):
    pass


def __getattr__(name: str):
    if name == "SAM3DependencyError":
        from .sam3 import SAM3DependencyError as _err

        return _err
    raise AttributeError(name)

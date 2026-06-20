from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version  # python >= 3.8

from packaging.version import parse

import_name = {"py-cpuinfo": "cpuinfo", "protobuf": "google.protobuf"}


def _log(msg: str):
    print(f"[ADetailer Custom install] {msg}")  # noqa: T201


def is_installed(
    package: str,
    min_version: str | None = None,
    max_version: str | None = None,
):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


def _is_reforge_or_webui() -> bool:
    lowered = " ".join(sys.argv).lower()
    keys = ("reforge", "forge", "stable-diffusion-webui", "webui")
    return any(key in lowered for key in keys)


def _pillow_health_check(stage: str):
    try:
        from PIL import Image  # noqa: F401
    except Exception as e:
        _log(f"Pillow import check ({stage}) failed: {e}")
        _log("If you are using reForge/Forge Neo, repair in that venv with:")
        _log(f'  "{sys.executable}" -m pip install --force-reinstall "Pillow==10.4.0"')
        msg = "Pillow import failed. See guidance above."
        raise RuntimeError(msg) from e

    try:
        pillow_ver = version("Pillow")
    except PackageNotFoundError:
        _log(f"Pillow is not installed ({stage}).")
        return

    _log(f"Pillow version ({stage}): {pillow_ver}")
    if parse(pillow_ver) >= parse("11"):
        _log("WARNING: Pillow>=11 detected. reForge/Forge Neo stacks often require Pillow<11.")
        _log('Recommended range for webui compatibility: "Pillow>=10.2.0,<11"')


def run_pip(*args, no_deps: bool = False):
    cmd = [sys.executable, "-m", "pip", "install"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(args)
    _log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def install():
    deps = [
        ("ultralytics", "8.3.75", None),
        ("mediapipe", "0.10.13", None),
        ("rich", "13.0.0", None),
    ]

    _log(f"Python executable: {sys.executable}")
    _log(f"argv: {sys.argv}")

    is_webui_env = _is_reforge_or_webui()
    allow_full_deps = os.environ.get("ADETAILER_INSTALL_DEPS", "").strip() == "1"
    no_deps = is_webui_env and not allow_full_deps

    if no_deps:
        _log("Detected reForge/Forge Neo/webui-like environment; using conservative '--no-deps' install mode.")
        _log("Set ADETAILER_INSTALL_DEPS=1 to allow dependency resolution (may alter shared packages).")

    _pillow_health_check("before pip")

    pkgs = []
    for pkg, low, high in deps:
        if is_installed(pkg, low, high):
            _log(f"Already satisfied: {pkg}")
            continue

        if low and high:
            spec = f"{pkg}>={low},<={high}"
        elif low:
            spec = f"{pkg}>={low}"
        elif high:
            spec = f"{pkg}<={high}"
        else:
            spec = pkg

        _log(f"Will install: {spec}")
        if no_deps:
            _log(f"NOTE: {pkg} dependencies are not auto-installed in --no-deps mode.")
        pkgs.append(spec)

    if pkgs:
        run_pip(*pkgs, no_deps=no_deps)
    else:
        _log("All install requirements already satisfied. No action needed.")

    _pillow_health_check("after pip")


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
else:
    _log("Skipped by --skip-install")

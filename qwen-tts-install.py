#!/usr/bin/env python3

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from packaging.version import InvalidVersion, Version


PYTORCH_ROOT_INDEX = "https://download.pytorch.org/whl/"
GITHUB_RELEASES_API = "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases"
CACHE_DIR = Path.home() / ".cache" / "qwen_tts_installer"
CACHE_FILE = CACHE_DIR / "flash_matrix.json"
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours


@dataclass
class FlashAsset:
    release_tag: str
    asset_name: str
    download_url: str
    flash_version: str
    cuda_tag: str
    torch_version: str
    py_tag: str
    platform_tag: str


@dataclass
class InstallPlan:
    host_cuda_version: str
    host_cuda_numeric: int
    python_tag: str
    platform_tag: str
    selected_cuda_tag: str
    candidate_cuda_tags: List[str]
    torch_version: str
    torchvision_version: str
    torchaudio_version: str
    flash_release_tag: str
    flash_asset_name: str
    flash_download_url: str
    flash_version: str
    torch_command: str
    flash_command: str
    qwen_command: str
    torch_dry_run_ok: bool
    flash_url_ok: bool
    qwen_dry_run_ok: bool


def log_step(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_ok(message: str) -> None:
    print(f"[ OK ] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def run_command(
    cmd: List[str],
    check: bool = True,
    announce: Optional[str] = None,
) -> Tuple[int, str]:
    if announce:
        log_step(announce)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")

    if proc.returncode == 0:
        if announce:
            log_ok(announce)
    else:
        if announce:
            log_warn(f"{announce} failed")

    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{output}")

    return proc.returncode, output


def run_shell_command(
    cmd: str,
    check: bool = True,
    announce: Optional[str] = None,
) -> Tuple[int, str]:
    return run_command(shlex.split(cmd), check=check, announce=announce)


def get_host_cuda_version_from_nvidia_smi() -> Tuple[str, int]:
    _, output = run_command(
        ["nvidia-smi"],
        announce="Detecting host CUDA version with nvidia-smi",
    )
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", output)
    if not m:
        raise RuntimeError("Could not find CUDA version in nvidia-smi output.")
    major = int(m.group(1))
    minor = int(m.group(2))
    human = f"{major}.{minor}"
    numeric = major * 10 + minor
    log_ok(f"Detected host CUDA {human}")
    return human, numeric


def current_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def current_platform_tag() -> str:
    machine = platform.machine().lower()
    system = platform.system().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        return "linux_x86_64"

    raise RuntimeError(f"Unsupported platform for this installer: {system}_{machine}")


def get_available_pytorch_cuda_tags() -> List[str]:
    log_step("Reading available PyTorch CUDA wheel buckets")
    r = requests.get(PYTORCH_ROOT_INDEX, timeout=30)
    r.raise_for_status()
    tags = set(re.findall(r"\bcu(\d{2,3})\b", r.text))
    result = sorted((f"cu{t}" for t in tags), key=lambda x: int(x[2:]))
    log_ok(f"Found PyTorch CUDA buckets: {', '.join(result)}")
    return result


def choose_candidate_cuda_tags(host_cuda_numeric: int, available_tags: List[str]) -> List[str]:
    eligible = sorted(
        [tag for tag in available_tags if int(tag[2:]) <= host_cuda_numeric],
        key=lambda x: int(x[2:]),
        reverse=True,
    )
    if not eligible:
        raise RuntimeError("No compatible PyTorch CUDA buckets found for this host.")
    return eligible


def parse_flash_asset_name(name: str) -> Optional[FlashAsset]:
    pattern = (
        r"^flash_attn-"
        r"(?P<flash>[0-9][0-9A-Za-z.\-]*)"
        r"\+"
        r"(?P<cuda>cu\d{2,3})"
        r"torch(?P<torch>[0-9]+(?:\.[0-9]+)*)"
        r"-"
        r"(?P<py1>cp\d{3,4})"
        r"-"
        r"(?P<py2>cp\d{3,4})"
        r"-"
        r"(?P<platform>linux_x86_64)\.whl$"
    )
    m = re.match(pattern, name)
    if not m:
        return None

    if m.group("py1") != m.group("py2"):
        return None

    try:
        Version(m.group("flash"))
        Version(m.group("torch"))
    except InvalidVersion:
        return None

    return FlashAsset(
        release_tag="",
        asset_name=name,
        download_url="",
        flash_version=m.group("flash"),
        cuda_tag=m.group("cuda"),
        torch_version=m.group("torch"),
        py_tag=m.group("py1"),
        platform_tag=m.group("platform"),
    )


def load_cache() -> Optional[List[Dict]]:
    if not CACHE_FILE.exists():
        return None

    try:
        data = json.loads(CACHE_FILE.read_text())
        fetched_at = data.get("fetched_at", 0)
        ttl = data.get("ttl_seconds", CACHE_TTL_SECONDS)
        if time.time() - fetched_at > ttl:
            return None
        return data.get("items", [])
    except Exception:
        return None


def save_cache(items: List[Dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": time.time(),
        "ttl_seconds": CACHE_TTL_SECONDS,
        "items": items,
    }
    CACHE_FILE.write_text(json.dumps(payload, indent=2))


def fetch_flash_matrix(use_cache: bool = True, github_token: Optional[str] = None) -> List[FlashAsset]:
    if use_cache:
        cached = load_cache()
        if cached is not None:
            log_ok(f"Using cached FlashAttention matrix ({len(cached)} entries)")
            return [FlashAsset(**item) for item in cached]

    log_step("Fetching FlashAttention release assets from GitHub")

    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    r = requests.get(GITHUB_RELEASES_API, headers=headers, timeout=30)
    r.raise_for_status()
    releases = r.json()

    assets: List[FlashAsset] = []
    for rel in releases:
        release_tag = rel.get("tag_name", "")
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            parsed = parse_flash_asset_name(name)
            if not parsed:
                continue
            parsed.release_tag = release_tag
            parsed.download_url = asset["browser_download_url"]
            assets.append(parsed)

    save_cache([asdict(a) for a in assets])
    log_ok(f"Fetched and cached {len(assets)} FlashAttention wheel entries")
    return assets


def map_torch_family_versions(torch_version_str: str) -> Tuple[str, str, str]:
    tv = Version(torch_version_str)

    if tv.major != 2:
        raise RuntimeError(f"Unsupported torch major version: {torch_version_str}")

    torchvision_minor = tv.minor + 15
    torchvision_version = f"0.{torchvision_minor}.{tv.micro}"
    torchaudio_version = f"{tv.major}.{tv.minor}.{tv.micro}"
    torch_version = torchaudio_version

    return torch_version, torchvision_version, torchaudio_version


def choose_best_asset(
    matrix: List[FlashAsset],
    candidate_cuda_tags: List[str],
    python_tag: str,
    platform_tag: str,
) -> FlashAsset:
    filtered = [a for a in matrix if a.py_tag == python_tag and a.platform_tag == platform_tag]
    if not filtered:
        raise RuntimeError(f"No flash-attn assets found for Python {python_tag} on {platform_tag}.")

    log_step("Selecting best compatible FlashAttention asset")

    for cuda_tag in candidate_cuda_tags:
        bucket = [a for a in filtered if a.cuda_tag == cuda_tag]
        if not bucket:
            log_warn(f"No FlashAttention wheels found for fallback bucket {cuda_tag}")
            continue

        bucket.sort(
            key=lambda a: (Version(a.torch_version), Version(a.flash_version), a.release_tag),
            reverse=True,
        )
        best = bucket[0]
        log_ok(f"Selected {best.asset_name} from {best.release_tag}")
        return best

    raise RuntimeError("No flash-attn asset found for any compatible CUDA fallback bucket.")


def build_torch_command(cuda_tag: str, torch_v: str, torchvision_v: str, torchaudio_v: str) -> str:
    return (
        f"pip3 install "
        f"torch=={torch_v} "
        f"torchvision=={torchvision_v} "
        f"torchaudio=={torchaudio_v} "
        f"--index-url https://download.pytorch.org/whl/{cuda_tag}"
    )


def build_flash_command(download_url: str) -> str:
    return f"pip install {download_url}"


def validate_flash_url(url: str) -> bool:
    log_step("Validating FlashAttention download URL")
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code < 400:
            log_ok("FlashAttention URL is reachable")
            return True

        r = requests.get(url, stream=True, timeout=30)
        ok = r.status_code < 400
        if ok:
            log_ok("FlashAttention URL is reachable")
        else:
            log_warn(f"FlashAttention URL returned status {r.status_code}")
        return ok
    except Exception as exc:
        log_warn(f"FlashAttention URL validation failed: {exc}")
        return False


def pip_dry_run(cmd: str, label: str) -> bool:
    full_cmd = f"{cmd} --dry-run"
    rc, _ = run_shell_command(full_cmd, check=False, announce=f"Running dry-run for {label}")
    return rc == 0


def resolve_plan(use_cache: bool = True, github_token: Optional[str] = None) -> InstallPlan:
    host_cuda_version, host_cuda_numeric = get_host_cuda_version_from_nvidia_smi()

    python_tag = current_python_tag()
    platform_tag = current_platform_tag()
    log_ok(f"Detected Python tag {python_tag} on {platform_tag}")

    available_cuda_tags = get_available_pytorch_cuda_tags()
    candidate_cuda_tags = choose_candidate_cuda_tags(host_cuda_numeric, available_cuda_tags)
    log_ok(f"Candidate CUDA fallback buckets: {', '.join(candidate_cuda_tags)}")

    log_step("Loading FlashAttention compatibility matrix")
    matrix = fetch_flash_matrix(use_cache=use_cache, github_token=github_token)
    log_ok(f"Loaded {len(matrix)} FlashAttention wheel entries")

    best = choose_best_asset(matrix, candidate_cuda_tags, python_tag, platform_tag)

    torch_v, torchvision_v, torchaudio_v = map_torch_family_versions(best.torch_version)

    torch_cmd = build_torch_command(best.cuda_tag, torch_v, torchvision_v, torchaudio_v)
    flash_cmd = build_flash_command(best.download_url)
    qwen_cmd = "pip install -U qwen-tts"

    torch_dry_run_ok = pip_dry_run(torch_cmd, "PyTorch stack")
    if torch_dry_run_ok:
        log_ok("PyTorch dry-run succeeded")
    else:
        log_warn("PyTorch dry-run failed")

    flash_url_ok = validate_flash_url(best.download_url)

    qwen_dry_run_ok = pip_dry_run(qwen_cmd, "qwen-tts")
    if qwen_dry_run_ok:
        log_ok("qwen-tts dry-run succeeded")
    else:
        log_warn("qwen-tts dry-run failed")

    return InstallPlan(
        host_cuda_version=host_cuda_version,
        host_cuda_numeric=host_cuda_numeric,
        python_tag=python_tag,
        platform_tag=platform_tag,
        selected_cuda_tag=best.cuda_tag,
        candidate_cuda_tags=candidate_cuda_tags,
        torch_version=torch_v,
        torchvision_version=torchvision_v,
        torchaudio_version=torchaudio_v,
        flash_release_tag=best.release_tag,
        flash_asset_name=best.asset_name,
        flash_download_url=best.download_url,
        flash_version=best.flash_version,
        torch_command=torch_cmd,
        flash_command=flash_cmd,
        qwen_command=qwen_cmd,
        torch_dry_run_ok=torch_dry_run_ok,
        flash_url_ok=flash_url_ok,
        qwen_dry_run_ok=qwen_dry_run_ok,
    )


def print_preview(plan: InstallPlan) -> None:
    print()
    print("Resolved installation plan")
    print("--------------------------")
    print(f"Host CUDA version : {plan.host_cuda_version}")
    print(f"Python tag        : {plan.python_tag}")
    print(f"Platform          : {plan.platform_tag}")
    print(f"Fallback buckets  : {', '.join(plan.candidate_cuda_tags)}")
    print(f"Selected CUDA tag : {plan.selected_cuda_tag}")
    print(f"Torch             : {plan.torch_version}")
    print(f"Torchvision       : {plan.torchvision_version}")
    print(f"Torchaudio        : {plan.torchaudio_version}")
    print(f"FlashAttention    : {plan.flash_version}")
    print(f"Release tag       : {plan.flash_release_tag}")
    print(f"Asset             : {plan.flash_asset_name}")
    print()
    print("Validation")
    print("----------")
    print(f"PyTorch dry-run   : {plan.torch_dry_run_ok}")
    print(f"Flash URL         : {plan.flash_url_ok}")
    print(f"qwen-tts dry-run  : {plan.qwen_dry_run_ok}")
    print()
    print("Commands")
    print("--------")
    print(plan.torch_command)
    print(plan.flash_command)
    print(plan.qwen_command)
    print()


def execute_install(plan: InstallPlan) -> None:
    commands = [
        ("PyTorch stack", plan.torch_command),
        ("FlashAttention", plan.flash_command),
        ("qwen-tts", plan.qwen_command),
    ]

    for label, cmd in commands:
        log_step(f"Installing {label}")
        rc, out = run_shell_command(cmd, check=False)
        print(out, end="" if out.endswith("\n") else "\n")
        if rc != 0:
            raise RuntimeError(f"Installation failed for {label}: {cmd}")
        log_ok(f"Installed {label}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="qwen-tts-install")
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="Resolve the install plan")
    p_resolve.add_argument("--json", action="store_true", help="Emit JSON only")
    p_resolve.add_argument("--no-cache", action="store_true", help="Bypass cached GitHub matrix")

    p_preview = sub.add_parser("preview", help="Resolve and show a human-readable preview")
    p_preview.add_argument("--json", action="store_true", help="Emit JSON instead of preview text")
    p_preview.add_argument("--no-cache", action="store_true", help="Bypass cached GitHub matrix")

    p_install = sub.add_parser("install", help="Resolve, preview, and optionally install")
    p_install.add_argument("--json", action="store_true", help="Emit JSON before install")
    p_install.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p_install.add_argument("--no-cache", action="store_true", help="Bypass cached GitHub matrix")

    args = parser.parse_args()
    github_token = os.environ.get("GITHUB_TOKEN")

    try:
        plan = resolve_plan(use_cache=not args.no_cache, github_token=github_token)

        if getattr(args, "json", False):
            print(json.dumps(asdict(plan), indent=2))
            if args.command != "install":
                return

        if args.command in ("resolve", "preview"):
            if not getattr(args, "json", False):
                print_preview(plan)
            return

        if args.command == "install":
            if not getattr(args, "json", False):
                print_preview(plan)

            if not args.yes:
                answer = input("Proceed with installation? [y/N]: ").strip().lower()
                if answer not in ("y", "yes"):
                    print("Installation cancelled.")
                    return

            execute_install(plan)

    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

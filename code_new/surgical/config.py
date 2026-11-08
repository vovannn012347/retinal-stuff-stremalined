"""Config loading: YAML with `extends:` inheritance and `--set key.sub=value` overrides.

Relative paths in the config are resolved against the repository root
(the folder that contains `code_new/`), so the same config works no matter
which folder the script is started from.
"""
import copy
import os
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parents[1]      # .../code_new
REPO_ROOT = CODE_DIR.parent                         # .../retinal-stuff-streamlined


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_plain(self):
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        data = deep_merge(_load_yaml(parent_path), data)
    return data


def _parse_value(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(data: dict, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return data


def resolve_path(p):
    """Resolve a config path against the repository root (absolute paths stay as they are)."""
    if p is None or p == "":
        return None
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_config(path, overrides=None) -> Cfg:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        # allow "configs/x.yaml" relative to code_new
        candidate = CODE_DIR / path
        if candidate.exists():
            path = candidate
    data = _load_yaml(path.resolve())
    data = apply_overrides(data, overrides)
    return Cfg.wrap(data)

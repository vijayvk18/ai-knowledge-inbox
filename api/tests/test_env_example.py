"""The .env.example must stay true to the code.

A settings file that documents keys which no longer exist, or omits ones that
do, is worse than none: it is confidently wrong, and someone will set a value
and wonder why nothing changed. These tests fail the build when the two drift.
"""

import re
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings

EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

# KEY=value, whether commented out or live.
_SETTING_LINE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=(.*)$")


def _documented() -> dict[str, str]:
    documented = {}
    for line in EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        match = _SETTING_LINE.match(line.strip())
        if match:
            documented[match.group(1)] = match.group(2).split("#")[0].strip()
    return documented


def _real_settings() -> dict[str, object]:
    return {(field.alias or name).upper(): field.default for name, field in Settings.model_fields.items()}


def test_the_example_file_exists():
    assert EXAMPLE_PATH.is_file(), "api/.env.example is how anyone configures this app"


def test_every_documented_key_is_a_real_setting():
    unknown = sorted(set(_documented()) - set(_real_settings()))

    assert unknown == [], f"documented but not real settings (renamed or removed?): {unknown}"


def test_every_setting_is_documented():
    missing = sorted(set(_real_settings()) - set(_documented()))

    assert missing == [], f"real settings missing from .env.example: {missing}"


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
def test_credential_keys_are_present_and_empty(key):
    """They must be uncommented so a copied file is ready to fill in - and blank."""
    documented = _documented()

    assert key in documented
    assert documented[key] == "", f"{key} must ship empty, never with a value"


def test_no_real_credential_leaks_into_the_example():
    content = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "sk-ant-api" not in content
    assert not re.search(r"sk-[a-zA-Z0-9]{20,}", content), "that looks like a real API key"


def test_documented_defaults_match_the_code():
    """Numbers and strings shown as defaults must actually be the defaults."""
    real = _real_settings()
    mismatched = []

    for key, shown in _documented().items():
        default = real.get(key)
        if shown == "" or default is None:
            continue  # blank credential slots and None-defaults carry no claim
        if key == "DB_FILE":
            continue  # an absolute path computed at import; the file explains it
        # 60 and 60.0 are the same claim.
        try:
            if float(shown) == float(default):
                continue
        except (TypeError, ValueError):
            pass
        if str(shown).lower() != str(default).lower():
            mismatched.append(f"{key}: example says {shown!r}, code default is {default!r}")

    assert mismatched == [], "\n".join(mismatched)


def test_the_example_is_not_silently_loaded_as_config():
    """.env.example must stay an example - only .env is read."""
    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_the_example_parses_as_env_and_produces_valid_settings(tmp_path, monkeypatch):
    """Uncomment everything and the result must still be a usable configuration."""
    lines = []
    for line in EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        match = _SETTING_LINE.match(line.strip())
        if match and match.group(2).split("#")[0].strip():
            lines.append(f"{match.group(1)}={match.group(2).split('#')[0].strip()}")

    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(lines), encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.port > 0
    assert settings.chunk_overlap_chars < settings.chunk_target_chars
    assert settings.retrieval_top_k <= settings.retrieval_max_top_k
    assert settings.embedding_provider_name in {"openai", "local"}
    assert settings.answer_provider_name in {"anthropic", "echo"}

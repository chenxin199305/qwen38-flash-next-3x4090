from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "MODEL_ID",
    "MODEL_DIR",
    "FP8HYBRID_DIR",
    "MTP_FP8_DIR",
    "MODEL_PATH",
    "VENV",
    "GPU_IDS",
    "TP_SIZE",
    "BIND_HOST",
    "PORT",
    "MTP_K",
    "MAX_MODEL_LEN",
}


def parse_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def test_sample_has_required_keys():
    env = parse_env(ROOT / ".env.sample")
    assert REQUIRED <= set(env)
    assert env["BIND_HOST"] == "127.0.0.1"
    assert env["TP_SIZE"] == "3"


def test_profiles_are_parseable():
    for path in (ROOT / "config" / "profiles").glob("*.env"):
        env = parse_env(path)
        assert "MTP_K" in env, path
        assert "MAX_MODEL_LEN" in env, path
        assert "MAX_NUM_SEQS" in env, path
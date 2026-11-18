"""
Deploy the API + chat UI to a Hugging Face Space (free Docker CPU tier).

One-time:   pip install huggingface_hub   and   hf auth login   (paste a *write* token)
Deploy:     python scripts/deploy_hf.py --space <hf-username>/finresearch-agent
            add --push-secrets to copy GOOGLE_API_KEY / DATABASE_URL / GROQ_API_KEY from
            your local .env into the Space's encrypted secrets (values are never printed).
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/shruti1303-runwal/financial-research-agent"

SPACE_README = f"""---
title: FinResearch Agent
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers on annual reports (RAG + MCP agent)
---

# Agentic Financial Research Assistant

A LangChain agent that answers questions about company annual reports with page-level citations,
using RAG on PostgreSQL/pgvector plus live market data and calculator tools served over MCP.

Source code, architecture and evaluation: {GITHUB}
"""

# Non-secret settings for the public demo
VARIABLES = {
    "CHAT_MODEL": "gemini-flash-lite-latest",
    "FALLBACK_MODELS": "gemini-flash-latest,groq:meta-llama/llama-4-scout-17b-16e-instruct",
    "EMBEDDING_MODEL": "models/gemini-embedding-001",
    "EMBEDDING_DIM": "768",
    "CORS_ORIGINS": "https://finresearch-agent-shruti.netlify.app",
    "ALLOW_UPLOADS": "false",
    "RATE_LIMIT_PER_HOUR": "5",
}
SECRETS = ["GOOGLE_API_KEY", "DATABASE_URL", "GROQ_API_KEY"]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    skip = ("docs/", ".github/", "scripts/", "netlify.toml", "README.md", "app/static/config.js")
    return [f for f in out.splitlines() if f and not f.startswith(skip) and (ROOT / f).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--push-secrets", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    print("Logged in to Hugging Face as:", api.whoami()["name"])
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for f in tracked_files():
            dst = Path(tmp) / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dst)
        (Path(tmp) / "README.md").write_text(SPACE_README, encoding="utf-8")
        api.upload_folder(folder_path=tmp, repo_id=a.space, repo_type="space",
                          commit_message="Deploy from GitHub repo",
                          delete_patterns=["app/**", "eval/**", "data/**"])

    for k, v in VARIABLES.items():
        api.add_space_variable(a.space, k, v)
    if a.push_secrets:
        env = dotenv_values(ROOT / ".env")
        for k in SECRETS:
            if env.get(k):
                api.add_space_secret(a.space, k, env[k])
                print(f"  secret {k}: set")
            else:
                print(f"  secret {k}: missing in .env, skipped")

    user, name = a.space.split("/")
    print(f"\nSpace page : https://huggingface.co/spaces/{a.space}")
    print(f"API + UI   : https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")
    print("The first build takes ~5 minutes (watch the 'Logs' tab on the Space page).")


if __name__ == "__main__":
    main()

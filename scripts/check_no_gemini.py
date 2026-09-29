"""Politique de conformite : aucun runtime GenAI interdit dans le depot.

Le scan recherche les references runtime interdites dans le code et la
configuration. Les mentions du present script (qui doit bien etre appele par
la CI) sont neutralisees avant analyse afin de ne pas declencher de faux
positif.

Sortie :
  - "NO_GEMINI_SCAN_PASS"                                   -> code 0
  - "FORBIDDEN REFERENCES:" suivi des couples (fichier, terme) -> code 1
"""
from pathlib import Path

TERMS = ["gemini", "google-genai", "GEMINI_API_KEY", "veo"]

# Jetons du script de politique lui-meme : retires avant analyse.
SELF_TOKENS = ["check_no_gemini", "no_gemini"]

IGNORE_NAMES = {"README.md", "NO_GEMINI.md"}
IGNORE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".json",
                   ".toml", ".yaml", ".yml", ".env.example"}


def scan(root: Path):
    bad = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in IGNORE_NAMES or p.name == "check_no_gemini.py":
            continue
        suffix = p.suffix if p.name != ".env.example" else ".env.example"
        if suffix not in IGNORE_SUFFIXES:
            continue
        try:
            text = p.read_text(errors="ignore").lower()
        except OSError:
            continue
        for tok in SELF_TOKENS:
            text = text.replace(tok.lower(), "")
        for term in TERMS:
            if term.lower() in text:
                bad.append((str(p.relative_to(root)), term))
    return bad


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bad = scan(root)
    if bad:
        print("FORBIDDEN REFERENCES:")
        for item in bad:
            print(item)
        raise SystemExit(1)
    print("NO_GEMINI_SCAN_PASS")


if __name__ == "__main__":
    main()

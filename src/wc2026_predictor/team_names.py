from __future__ import annotations

import unicodedata

ALIASES = {
    "USA": "United States",
    "USMNT": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Czech Republic": "Czechia",
    "Türkiye": "Turkiye",
    "Turkey": "Turkiye",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def normalize_team(name: str) -> str:
    """Normalize team names across datasets without losing common display names."""
    if not isinstance(name, str):
        return name
    cleaned = " ".join(name.strip().split())
    cleaned = ALIASES.get(cleaned, cleaned)
    ascii_name = (
        unicodedata.normalize("NFKD", cleaned)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return ALIASES.get(ascii_name, ascii_name)

"""
overlay.py
==========
Dessine un petit panneau de télémétrie (métriques choisies) sur une frame
RGB, à partir de la ligne de step_records correspondante. Pur traitement
d'image (PIL) — aucune logique RL ici, c'est pour ça que ce module peut
être importé aussi bien côté API (video_export.py, pour la vidéo de démo
persistante) que côté GUI (app.py, pour l'aperçu temporaire) : dans les
deux cas, on ne fait qu'annoter une image déjà produite avec des valeurs
déjà calculées.
"""

import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from signals import SIGNAL_COLUMNS


def _ascii_safe(text: str) -> str:
    """
    Supprime les accents avant dessin : la police bitmap par défaut de PIL
    (ImageFont.load_default) ne couvre pas fiablement les caractères
    accentués selon les versions/OS, et affiche des tofu ("Récompense" ->
    "R�compense") plutôt que de lever une erreur. On garde les labels
    accentués partout ailleurs (dropdown Gradio, etc.) — seul le rendu
    PIL sur les frames vidéo est concerné.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _format_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def draw_metrics_overlay(frame: np.ndarray, step_record: dict, selected_labels: list[str]) -> np.ndarray:
    """
    Renvoie une COPIE de `frame` avec un panneau semi-transparent en haut à
    gauche affichant, pour chaque label de `selected_labels` (des clés de
    SIGNAL_COLUMNS), la valeur correspondante dans `step_record`.

    Labels absents de SIGNAL_COLUMNS ou de step_record sont ignorés
    silencieusement plutôt que de faire échouer tout l'encodage vidéo pour
    un nom de signal invalide.
    """
    if not selected_labels:
        return frame

    lines = []
    for label in selected_labels:
        col = SIGNAL_COLUMNS.get(label)
        if col is None or col not in step_record:
            continue
        lines.append(_ascii_safe(f"{label}: {_format_value(step_record[col])}"))

    if not lines:
        return frame

    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        # anciennes versions de Pillow : load_default() ne prend pas size=
        font = ImageFont.load_default()

    padding = 6
    line_height = 16
    panel_width = max(draw.textlength(line, font=font) for line in lines) + 2 * padding
    panel_height = len(lines) * line_height + 2 * padding

    draw.rectangle([0, 0, panel_width, panel_height], fill=(0, 0, 0, 140))
    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_height), line, fill=(255, 255, 255, 255), font=font)

    return np.array(img)

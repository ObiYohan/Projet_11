"""
signals.py
==========
Source UNIQUE de la liste des signaux disponibles (label affiché -> colonne
de step_records), partagée entre :
- app.py (menu déroulant du dashboard, menu de surimpression vidéo)
- video_export.py (surimpression sur la vidéo de démo persistante)

Évite que les deux endroits divergent silencieusement si on ajoute un
signal plus tard.

"Action moteur principal"/"latéral" n'existent que pour un modèle PPO
continu ; "Action (discrète)" seulement pour un DQN discret. Les colonnes
non pertinentes pour le modèle chargé restent à None dans step_records et
sont filtrées dynamiquement par les appelants (voir app.py::refresh_dashboard).
"""

SIGNAL_COLUMNS = {
    "Récompense": "reward",
    "Altitude (Y)": "obs_y",
    "Position X": "obs_x",
    "Vitesse X": "obs_vx",
    "Vitesse Y": "obs_vy",
    "Angle": "obs_angle",
    "Vitesse angulaire": "obs_angular_velocity",
    "Contact jambe gauche": "obs_leg1_contact",
    "Contact jambe droite": "obs_leg2_contact",
    "Action moteur principal": "action_main",
    "Action moteur latéral": "action_lateral",
    "Action (discrète)": "action_discrete",
}

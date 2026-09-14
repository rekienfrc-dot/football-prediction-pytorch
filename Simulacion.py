"""
Módulo de Simulación Instantánea de Partidos de Fútbol Internacional
Utiliza los modelos entrenados previamente (Red Neuronal PyTorch + Modelo Bivariado Dixon-Coles)
sin necesidad de volver a procesar el dataset ni entrenar las redes.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Importar arquitectura y utilidades desde Fut.py
from Fut import (
    FootballPredictorNN,
    PoissonGoalModel,
    calcular_matriz_poisson,
    asignar_peso_economico,
    cargar_modelo_completo,
)

# Ruta por defecto del archivo binario del modelo
MODELO_PATH = Path(__file__).resolve().parent / "modelo_futbol_entrenado.pt"

# Caché global para mantener el modelo cargado en memoria RAM
_MODELO_CACHE: Optional[dict] = None


def obtener_modelo() -> dict:
    """Carga y mantiene en memoria el modelo entrenado para ejecuciones instantáneas."""
    global _MODELO_CACHE
    if _MODELO_CACHE is None:
        if not MODELO_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo del modelo en: {MODELO_PATH}\n"
                "Por favor, ejecuta primero 'python Fut.py' para entrenar y generar el archivo del modelo."
            )
        print("Cargando modelo entrenado y metadatos en memoria...")
        _MODELO_CACHE = cargar_modelo_completo(MODELO_PATH)
        print("[OK] Modelo listo para simulaciones instantáneas.")
    return _MODELO_CACHE


def simular_partido(
    equipo1: str,
    equipo2: str,
    ciudad: Optional[str] = None,
    pais: Optional[str] = None,
    torneo: str = "FIFA World Cup qualification",
    neutral: bool = False,
    mostrar_grafico: bool = True,
    max_goles: int = 5,
) -> Dict[str, any]:
    """
    Simula un partido entre equipo1 (local) y equipo2 (visitante):
    - Calcula la probabilidad de victoria de cada uno y la probabilidad de empate.
    - Enumera del 1 al 5 los marcadores exactos más probables.
    - Muestra interactivamente el mapa de calor con matplotlib (plt.show()).

    Parámetros:
    - equipo1: Nombre de la selección local (ej. 'Argentina', 'Spain', 'Mexico').
    - equipo2: Nombre de la selección visitante (ej. 'Brazil', 'Germany', 'France').
    - ciudad: Ciudad del partido (opcional, si es None se toma la sede habitual de equipo1).
    - pais: País sede (opcional, si es None se toma el país habitual de equipo1).
    - torneo: Competición (ej. 'FIFA World Cup', 'Copa América', 'Friendly').
    - neutral: True si se juega en cancha neutral, False si equipo1 tiene localía.
    - mostrar_grafico: Si True, abre la ventana interactiva del mapa de calor con plt.show().
    - max_goles: Límite máximo de goles en la cuadrícula de probabilidades (por defecto 5).
    """
    bundle = obtener_modelo()

    modelo_nn: FootballPredictorNN = bundle["modelo_nn"]
    modelo_poisson: PoissonGoalModel = bundle["modelo_poisson"]
    scaler_nn = bundle["scaler_nn"]
    scaler_poisson = bundle["scaler_poisson"]
    team_to_id = bundle["team_to_id"]
    city_to_id = bundle["city_to_id"]
    country_to_id = bundle["country_to_id"]
    elos = bundle["elos_actuales"]
    historia = bundle["historial_actual"]
    sedes = bundle.get("equipos_sedes", {})

    # Inferir ciudad y país si no se especifican
    if ciudad is None or pais is None:
        sede_info = sedes.get(equipo1, {"city": "Capital", "country": equipo1})
        ciudad = ciudad or sede_info.get("city", "Capital")
        pais = pais or sede_info.get("country", equipo1)

    # Identificadores categóricos
    h_id = team_to_id.get(equipo1, 0)
    a_id = team_to_id.get(equipo2, 0)
    c_id = city_to_id.get(ciudad, 0)
    co_id = country_to_id.get(pais, 0)

    # ELO actual
    eh = elos.get(equipo1, 1500.0)
    ea = elos.get(equipo2, 1500.0)
    elo_diff = eh - ea

    # Historial de los últimos 10 partidos
    h_hist = historia.get(equipo1, [])[-10:]
    a_hist = historia.get(equipo2, [])[-10:]
    h_n = len(h_hist) if h_hist else 1
    a_n = len(a_hist) if a_hist else 1

    h_pts = sum(x[0] for x in h_hist) if h_hist else 0
    a_pts = sum(x[0] for x in a_hist) if a_hist else 0
    h_gf = sum(x[1] for x in h_hist) if h_hist else 0.0
    a_gf = sum(x[1] for x in a_hist) if a_hist else 0.0
    h_ga = sum(x[2] for x in h_hist) if h_hist else 0.0
    a_ga = sum(x[2] for x in a_hist) if a_hist else 0.0
    h_wr = (sum(x[3] for x in h_hist) / h_n) if h_hist else 0.0
    a_wr = (sum(x[3] for x in a_hist) / a_n) if a_hist else 0.0

    peso = asignar_peso_economico(torneo)
    neutral_val = 1.0 if neutral else 0.0

    # 1. Inferencia de la Red Neuronal Clasificadora
    cont_df = pd.DataFrame(
        [[eh, ea, elo_diff, peso, h_pts, a_pts, h_gf, a_gf, h_ga, a_ga, h_wr, a_wr, neutral_val]],
        columns=[
            "home_elo",
            "away_elo",
            "elo_diff",
            "peso_economico",
            "home_pts_10",
            "away_pts_10",
            "home_gf_10",
            "away_gf_10",
            "home_ga_10",
            "away_ga_10",
            "home_win_rate_10",
            "away_win_rate_10",
            "neutral_num",
        ],
    )
    cont_scaled = scaler_nn.transform(cont_df)

    cat_tensor = torch.tensor([[h_id, a_id, c_id, co_id]], dtype=torch.long)
    cont_tensor = torch.tensor(cont_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = modelo_nn(cat_tensor, cont_tensor)
        logit_val = logit.item()
        prob_victoria_nn = torch.sigmoid(logit).item()

    # 2. Inferencia del Modelo Poisson de Goles Esperados (xG)
    p_df = pd.DataFrame(
        [[logit_val, prob_victoria_nn, elo_diff, h_gf, a_gf, h_ga, a_ga, neutral_val, peso]],
        columns=[
            "nn_logit",
            "nn_prob",
            "elo_diff",
            "home_gf_10",
            "away_gf_10",
            "home_ga_10",
            "away_ga_10",
            "neutral_num",
            "peso_economico",
        ],
    )
    p_scaled = scaler_poisson.transform(p_df)
    p_tensor = torch.tensor(p_scaled, dtype=torch.float32)

    with torch.no_grad():
        lh, la = modelo_poisson(p_tensor)
        lambda_h = lh.item()
        mu_a = la.item()

    # 3. Matriz Bivariada Dixon-Coles (El modelo más preciso)
    matriz = calcular_matriz_poisson(
        lambda_h=lambda_h,
        mu_a=mu_a,
        prob_victoria_nn=prob_victoria_nn,
        max_goles=max_goles,
        metodo="dixon_coles",
    )

    # 4. Cálculo de Probabilidades 1X2
    prob_gana_equipo1 = float(np.tril(matriz, -1).sum() * 100)
    prob_empate = float(np.diag(matriz).sum() * 100)
    prob_gana_equipo2 = float(np.triu(matriz, 1).sum() * 100)

    # 5. Top 5 Resultados Más Probables
    todos_los_resultados = []
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            todos_los_resultados.append(((i, j), matriz[i, j]))
    todos_los_resultados.sort(key=lambda x: x[1], reverse=True)
    top_5 = todos_los_resultados[:5]

    # Presentación en Consola
    print("\n" + "=" * 70)
    print(f"SIMULACION DE PARTIDO: {equipo1.upper()} vs {equipo2.upper()}")
    print(f"Sede: {ciudad}, {pais} | Torneo: {torneo} | {'Cancha Neutral' if neutral else 'Localía: ' + equipo1}")
    print(f"ELO: {equipo1} ({eh:.1f}) | {equipo2} ({ea:.1f}) | Diferencia: {elo_diff:+.1f}")
    print(f"Goles esperados (xG): {equipo1} = {lambda_h:.2f} | {equipo2} = {mu_a:.2f}")
    print("=" * 70)

    print("\n--- PROBABILIDADES DEL ENCUENTRO (1X2) ---")
    print(f"  [+] Victoria {equipo1}: {prob_gana_equipo1:5.2f}%")
    print(f"  [=] Empate:              {prob_empate:5.2f}%")
    print(f"  [-] Victoria {equipo2}: {prob_gana_equipo2:5.2f}%")

    print("\n--- TOP 5 RESULTADOS MAS PROBABLES (Marcador Exacto) ---")
    for rank, ((g1, g2), p) in enumerate(top_5, 1):
        desenlace = f"Victoria {equipo1}" if g1 > g2 else ("Empate" if g1 == g2 else f"Victoria {equipo2}")
        print(f"  {rank}. {equipo1} {g1} - {g2} {equipo2}  -->  {p * 100:5.2f}%  ({desenlace})")
    print("=" * 70)

    # 6. Mostrar el Mapa de Calor con Matplotlib (sin guardar como PNG)
    if mostrar_grafico:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            matriz * 100,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Probabilidad (%)"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(
            f"Mapa de Calor de Probabilidades de Resultado (Dixon-Coles Calibrado)\n"
            f"{equipo1} (Local) vs {equipo2} (Visita)\n"
            f"(xG: {lambda_h:.2f} vs {mu_a:.2f} | P({equipo1}): {prob_gana_equipo1:.1f}% | X: {prob_empate:.1f}% | P({equipo2}): {prob_gana_equipo2:.1f}%)",
            fontsize=11,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlabel(f"Goles de {equipo2} (Visitante)", fontsize=11, labelpad=10)
        ax.set_ylabel(f"Goles de {equipo1} (Local)", fontsize=11, labelpad=10)
        plt.tight_layout()
        plt.show()

    return {
        "equipo1": equipo1,
        "equipo2": equipo2,
        "elo1": eh,
        "elo2": ea,
        "xg_equipo1": round(lambda_h, 2),
        "xg_equipo2": round(mu_a, 2),
        "prob_victoria_equipo1": round(prob_gana_equipo1, 2),
        "prob_empate": round(prob_empate, 2),
        "prob_victoria_equipo2": round(prob_gana_equipo2, 2),
        "top_5_marcadores": top_5,
        "matriz_probabilidades": matriz,
    }


if __name__ == "__main__":
    # Ejemplo de uso directo
    simular_partido("Argentina", "Brazil", torneo="FIFA World Cup qualification")

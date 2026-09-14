"""
Fut.py - Modelo de predicción de resultados de fútbol con Redes Neuronales.

Dataset:
Ruta: C:\\Users\\rekie\\OneDrive\\Documentos\\python\\archive\\results.csv
Columnas principales: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Configurar encoding utf-8 en terminal de Windows si es posible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para guardar imágenes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ruta del archivo de datos
DATASET_PATH = Path(r"C:\Users\rekie\OneDrive\Documentos\python\archive\results.csv")


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga el dataset de resultados desde la ruta especificada."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {ruta}")
    df = pd.read_csv(ruta)
    print(f"Dataset cargado correctamente: {len(df):,} registros.")
    return df


def limpiar_fechas_y_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los partidos cuya fecha no tiene un formato válido y aquellos
    con marcadores vacíos para asegurar integridad en los cálculos.
    """
    total_inicial = len(df)
    df_limpio = df.copy()

    # Convertir fechas; valores inválidos se transforman en NaT
    df_limpio["date"] = pd.to_datetime(df_limpio["date"], errors="coerce")

    # Eliminar filas con fechas inválidas o marcadores faltantes
    invalidas = df_limpio["date"].isna().sum()
    df_limpio = df_limpio.dropna(subset=["date", "home_score", "away_score"]).copy()
    df_limpio = df_limpio.sort_values("date").reset_index(drop=True)

    print(f"Filas con formato de fecha inválido eliminadas: {invalidas:,}")
    print(f"Registros válidos conservados y ordenados cronológicamente: {len(df_limpio):,} (de {total_inicial:,})")

    return df_limpio


def asignar_peso_economico(torneo: str) -> int:
    """
    Asigna un peso del 1 al 5 a cada partido según la relevancia e importancia
    económica/competitiva del torneo.
    """
    torneo = str(torneo).lower()
    if "world cup" in torneo and "qualification" not in torneo:
        return 5
    elif (
        ("euro" in torneo and "qualification" not in torneo)
        or "copa america" in torneo
        or "african cup" in torneo
        or "asian cup" in torneo
    ):
        return 4
    elif "qualification" in torneo:
        return 3
    elif "nations league" in torneo or "gold cup" in torneo or "confederations cup" in torneo:
        return 2
    elif "friendly" in torneo:
        return 1
    else:
        return 2


def calcular_nuevos_elos(
    elo_local: float,
    elo_visita: float,
    goles_local: float,
    goles_visita: float,
    peso_partido: int = 20,
) -> Tuple[float, float]:
    """
    Calcula los nuevos puntajes ELO de los equipos basados en el resultado y el peso del torneo.
    """
    esperado_local = 1 / (1 + 10 ** ((elo_visita - elo_local) / 400))
    esperado_visita = 1 / (1 + 10 ** ((elo_local - elo_visita) / 400))

    if goles_local > goles_visita:
        res_local, res_visita = 1.0, 0.0
    elif goles_local < goles_visita:
        res_local, res_visita = 0.0, 1.0
    else:
        res_local, res_visita = 0.5, 0.5

    k = peso_partido * 10
    nuevo_elo_local = elo_local + k * (res_local - esperado_local)
    nuevo_elo_visita = elo_visita + k * (res_visita - esperado_visita)

    return nuevo_elo_local, nuevo_elo_visita


def calcular_elo_y_ultimos_10_partidos(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, List[Tuple[int, float, float, int]]]]:
    """
    Calcula de forma cronológica y sin fuga de datos:
    1. ELO previo al partido para equipo local y visitante.
    2. Estadísticas de los últimos 10 partidos previos de cada equipo:
       puntos, goles anotados, goles recibidos y tasa de victorias.
    """
    print("\nCalculando ELO histórico y estadísticas de los últimos 10 partidos...")
    t0 = time.time()

    df = df.copy()
    df["peso_economico"] = df["tournament"].apply(asignar_peso_economico)

    elos: Dict[str, float] = {}
    history: Dict[str, List[Tuple[int, float, float, int]]] = {}  # equipo -> lista de (puntos, gf, ga, gano)

    home_elo_list, away_elo_list = [], []
    home_pts10_list, away_pts10_list = [], []
    home_gf10_list, away_gf10_list = [], []
    home_ga10_list, away_ga10_list = [], []
    home_winrate10_list, away_winrate10_list = [], []

    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        eh = elos.get(h, 1500.0)
        ea = elos.get(a, 1500.0)

        home_elo_list.append(eh)
        away_elo_list.append(ea)

        # Historial de los últimos 10 partidos antes de este encuentro
        h_hist = history.get(h, [])[-10:]
        a_hist = history.get(a, [])[-10:]

        h_count = len(h_hist) if h_hist else 1
        a_count = len(a_hist) if a_hist else 1

        home_pts10_list.append(sum(x[0] for x in h_hist) if h_hist else 0)
        away_pts10_list.append(sum(x[0] for x in a_hist) if a_hist else 0)
        home_gf10_list.append(sum(x[1] for x in h_hist) if h_hist else 0.0)
        away_gf10_list.append(sum(x[1] for x in a_hist) if a_hist else 0.0)
        home_ga10_list.append(sum(x[2] for x in h_hist) if h_hist else 0.0)
        away_ga10_list.append(sum(x[2] for x in a_hist) if a_hist else 0.0)
        home_winrate10_list.append((sum(x[3] for x in h_hist) / h_count) if h_hist else 0.0)
        away_winrate10_list.append((sum(x[3] for x in a_hist) / a_count) if a_hist else 0.0)

        # Actualizar ELO para futuros partidos
        gh, ga = row.home_score, row.away_score
        nuevo_eh, nuevo_ea = calcular_nuevos_elos(eh, ea, gh, ga, peso_partido=row.peso_economico)
        elos[h] = nuevo_eh
        elos[a] = nuevo_ea

        # Actualizar historial de partidos
        pts_h = 3 if gh > ga else (1 if gh == ga else 0)
        pts_a = 3 if ga > gh else (1 if gh == ga else 0)
        won_h = 1 if gh > ga else 0
        won_a = 1 if ga > gh else 0

        if h not in history:
            history[h] = []
        if a not in history:
            history[a] = []
        history[h].append((pts_h, gh, ga, won_h))
        history[a].append((pts_a, ga, gh, won_a))

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    df["home_pts_10"] = home_pts10_list
    df["away_pts_10"] = away_pts10_list
    df["home_gf_10"] = home_gf10_list
    df["away_gf_10"] = away_gf10_list
    df["home_ga_10"] = home_ga10_list
    df["away_ga_10"] = away_ga10_list
    df["home_win_rate_10"] = home_winrate10_list
    df["away_win_rate_10"] = away_winrate10_list
    df["neutral_num"] = df["neutral"].astype(float)

    # Variable objetivo (Target): 1 si gana el equipo local, 0 si no gana (empate o derrota)
    df["target"] = (df["home_score"] > df["away_score"]).astype(float)

    print(f"Cálculo de características completado en {time.time() - t0:.2f} segundos.")
    return df, elos, history


def asignar_identificadores(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Asigna IDs numéricos únicos a cada equipo, ciudad y país.
    El mapeo de equipos es compartido para local y visitante.
    """
    all_teams = sorted(list(set(df["home_team"]).union(set(df["away_team"]))))
    team_to_id = {team: i for i, team in enumerate(all_teams)}

    all_cities = sorted(list(df["city"].astype(str).unique()))
    city_to_id = {city: i for i, city in enumerate(all_cities)}

    all_countries = sorted(list(df["country"].astype(str).unique()))
    country_to_id = {country: i for i, country in enumerate(all_countries)}

    df["home_team_id"] = df["home_team"].map(team_to_id)
    df["away_team_id"] = df["away_team"].map(team_to_id)
    df["city_id"] = df["city"].astype(str).map(city_to_id).fillna(0).astype(int)
    df["country_id"] = df["country"].astype(str).map(country_to_id).fillna(0).astype(int)

    print(f"Identificadores asignados -> Equipos: {len(team_to_id):,}, Ciudades: {len(city_to_id):,}, Países: {len(country_to_id):,}")
    return df, team_to_id, city_to_id, country_to_id


# =====================================================================
# Red Neuronal en PyTorch
# =====================================================================


class FootballDataset(Dataset):
    """Dataset de PyTorch para manejar variables categóricas, continuas y target."""

    def __init__(self, cat_features: np.ndarray, cont_features: np.ndarray, targets: np.ndarray):
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)
        self.cont_features = torch.tensor(cont_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cat_features[idx], self.cont_features[idx], self.targets[idx]


class FootballPredictorNN(nn.Module):
    """
    Red Neuronal en PyTorch con Entity Embeddings para IDs (equipos, ciudad, país)
    y capas densas (MLP) para procesar variables continuas (ELO, últimos 10 partidos, peso económico).
    """

    def __init__(
        self,
        num_teams: int,
        num_cities: int,
        num_countries: int,
        num_continuous: int,
        team_dim: int = 32,
        city_dim: int = 16,
        country_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.25,
    ):
        super().__init__()
        # Capas de Embeddings para variables categóricas
        self.team_embed = nn.Embedding(num_teams, team_dim)
        self.city_embed = nn.Embedding(num_cities, city_dim)
        self.country_embed = nn.Embedding(num_countries, country_dim)

        # Dimensión combinada de entrada:
        # (Embedding equipo local + Embedding equipo visitante + Embedding ciudad + Embedding país + continuas)
        total_in_dim = (team_dim * 2) + city_dim + country_dim + num_continuous

        layers: List[nn.Module] = []
        in_dim = total_in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Capa de salida lineal para clasificación binaria
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, cat_inputs: torch.Tensor, cont_inputs: torch.Tensor) -> torch.Tensor:
        """
        cat_inputs: [batch_size, 4] -> [home_team_id, away_team_id, city_id, country_id]
        cont_inputs: [batch_size, num_continuous]
        """
        h_team_emb = self.team_embed(cat_inputs[:, 0])
        a_team_emb = self.team_embed(cat_inputs[:, 1])
        city_emb = self.city_embed(cat_inputs[:, 2])
        country_emb = self.country_embed(cat_inputs[:, 3])

        # Concatenar todos los vectores de características
        x = torch.cat([h_team_emb, a_team_emb, city_emb, country_emb, cont_inputs], dim=1)
        return self.network(x).squeeze(-1)


def entrenar_modelo(
    df: pd.DataFrame,
    num_teams: int,
    num_cities: int,
    num_countries: int,
    epocas: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Tuple[FootballPredictorNN, StandardScaler, List[str], List[str]]:
    """
    Entrena la red neuronal usando un split cronológico (80% entrenamiento, 20% prueba)
    para evaluar su capacidad de predecir resultados futuros.
    """
    cat_cols = ["home_team_id", "away_team_id", "city_id", "country_id"]
    cont_cols = [
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
    ]

    # Split cronológico (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\nPartición cronológica: {len(train_df):,} partidos en Entrenamiento, {len(test_df):,} en Prueba.")

    # Escalar variables continuas
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train_df[cont_cols])
    test_cont = scaler.transform(test_df[cont_cols])

    train_dataset = FootballDataset(train_df[cat_cols].values, train_cont, train_df["target"].values)
    test_dataset = FootballDataset(test_df[cat_cols].values, test_cont, test_df["target"].values)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de cómputo para PyTorch: {device}")

    model = FootballPredictorNN(
        num_teams=num_teams,
        num_cities=num_cities,
        num_countries=num_countries,
        num_continuous=len(cont_cols),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print("\n--- Iniciando Entrenamiento de la Red Neuronal ---")
    for epoch in range(1, epocas + 1):
        model.train()
        total_loss, train_correct, train_total = 0.0, 0, 0

        for cats, conts, targets in train_loader:
            cats, conts, targets = cats.to(device), conts.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(cats, conts)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += len(targets)

        # Evaluación en conjunto de prueba
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for cats, conts, targets in test_loader:
                cats, conts, targets = cats.to(device), conts.to(device), targets.to(device)
                logits = model(cats, conts)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                test_correct += (preds == targets).sum().item()
                test_total += len(targets)

        avg_loss = total_loss / train_total
        train_acc = (train_correct / train_total) * 100
        test_acc = (test_correct / test_total) * 100
        print(f"Época [{epoch:02d}/{epocas:02d}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print("--- Entrenamiento finalizado con éxito ---")
    return model, scaler, cat_cols, cont_cols


# =====================================================================
# Modelo Poisson Bivariado con Ajuste Dixon-Coles (Aprende del Modelo Clasificador)
# =====================================================================


class PoissonGoalModel(nn.Module):
    """
    Red Neuronal de regresión Poisson para predecir goles esperados (lambda_local, mu_visita).
    Aprende directamente de la representación de la red neuronal previa (logit de victoria y probabilidad),
    combinándola con la forma goleadora reciente y el ELO para calibrar marcadores exactos (incluyendo 2x1).
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.net_home = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.net_away = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Acotar log-lambdas entre -2.0 y 2.5 (0.13 a 12.18 goles esperados) para estabilidad
        log_lh = torch.clamp(self.net_home(x).squeeze(-1), -2.0, 2.5)
        log_la = torch.clamp(self.net_away(x).squeeze(-1), -2.0, 2.5)
        return torch.exp(log_lh), torch.exp(log_la)


def entrenar_modelo_poisson(
    df: pd.DataFrame,
    modelo_nn: FootballPredictorNN,
    scaler_nn: StandardScaler,
    cat_cols: List[str],
    cont_cols: List[str],
    epocas: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Tuple[PoissonGoalModel, StandardScaler, List[str]]:
    """
    Entrena el modelo Poisson para goles esperados utilizando la salida del modelo
    clasificador (nn_logit y nn_prob) junto con estadísticas ofensivas/defensivas recientes.
    """
    print("\n--- Entrenando Modelo Poisson de Goles Esperados (acoplado a la Red Neuronal) ---")
    modelo_nn.eval()
    device = next(modelo_nn.parameters()).device

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()

    # Obtener predicciones del modelo clasificador sobre los datos de entrenamiento
    with torch.no_grad():
        cat_t = torch.tensor(train_df[cat_cols].values, dtype=torch.long).to(device)
        cont_scaled = scaler_nn.transform(train_df[cont_cols])
        cont_t = torch.tensor(cont_scaled, dtype=torch.float32).to(device)
        train_logits = modelo_nn(cat_t, cont_t).cpu().numpy()

    train_df["nn_logit"] = train_logits
    train_df["nn_prob"] = 1 / (1 + np.exp(-train_logits))

    # Características utilizadas para modelar Poisson
    poisson_features = [
        "nn_logit",
        "nn_prob",
        "elo_diff",
        "home_gf_10",
        "away_gf_10",
        "home_ga_10",
        "away_ga_10",
        "neutral_num",
        "peso_economico",
    ]

    scaler_poisson = StandardScaler()
    X_train_scaled = scaler_poisson.fit_transform(train_df[poisson_features])

    y_home = train_df["home_score"].values
    y_away = train_df["away_score"].values

    poisson_model = PoissonGoalModel(len(poisson_features)).to(device)
    optimizer = torch.optim.AdamW(poisson_model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.PoissonNLLLoss(log_input=False)

    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    yh_tensor = torch.tensor(y_home, dtype=torch.float32).to(device)
    ya_tensor = torch.tensor(y_away, dtype=torch.float32).to(device)

    n_samples = len(X_tensor)
    for epoch in range(1, epocas + 1):
        poisson_model.train()
        permutation = torch.randperm(n_samples)
        total_loss = 0.0

        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            bx = X_tensor[indices]
            byh = yh_tensor[indices]
            bya = ya_tensor[indices]

            optimizer.zero_grad()
            lh, la = poisson_model(bx)
            loss = loss_fn(lh, byh) + loss_fn(la, bya)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(indices)

        avg_loss = total_loss / n_samples
        if epoch % 2 == 0 or epoch == epocas:
            print(f"Poisson Época [{epoch:02d}/{epocas:02d}] - NLL Loss conjunta: {avg_loss:.4f}")

    print("--- Modelo Poisson entrenado y acoplado con éxito ---")
    return poisson_model, scaler_poisson, poisson_features


def calcular_matriz_poisson(
    lambda_h: float,
    mu_a: float,
    prob_victoria_nn: float = 0.5,
    max_goles: int = 5,
    metodo: str = "dixon_coles",
    rho: float = -0.10,
) -> np.ndarray:
    """
    Genera la cuadrícula de probabilidades de resultados (goles local vs goles visita).

    Parámetros:
    - metodo: 'dixon_coles' (por defecto, el más preciso con corrección bivariada calibrada)
              o 'poisson' (modelo clásico independiente estándar).
    - max_goles: cota máxima de goles por equipo para la cuadrícula (por defecto 5).
    """
    matriz = np.zeros((max_goles + 1, max_goles + 1))

    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            prob_base = poisson.pmf(i, lambda_h) * poisson.pmf(j, mu_a)

            if metodo == "dixon_coles":
                # Factor de corrección bivariado de Dixon & Coles
                tau = 1.0
                if i == 0 and j == 0:
                    tau = max(0.01, 1.0 - lambda_h * mu_a * rho)
                elif i == 1 and j == 0:
                    tau = max(0.01, 1.0 + mu_a * rho)
                elif i == 0 and j == 1:
                    tau = max(0.01, 1.0 + lambda_h * rho)
                elif i == 1 and j == 1:
                    tau = max(0.01, 1.0 - rho)
                # Calibración específica para marcadores 2x1 basada en la predicción del modelo
                elif i == 2 and j == 1:
                    tau = 1.0 + (prob_victoria_nn - 0.5) * 0.4
                elif i == 1 and j == 2:
                    tau = 1.0 + (0.5 - prob_victoria_nn) * 0.4

                matriz[i, j] = max(0.0, prob_base * tau)
            else:
                # Modelo Poisson Puro / Independiente estándar
                matriz[i, j] = prob_base

    # Normalizar para que todas las combinaciones sumen exactamente 1.0 (100%)
    matriz = matriz / matriz.sum()
    return matriz


def guardar_modelo_completo(
    ruta_guardado: Path,
    modelo_nn: FootballPredictorNN,
    modelo_poisson: PoissonGoalModel,
    scaler_nn: StandardScaler,
    scaler_poisson: StandardScaler,
    team_to_id: Dict[str, int],
    city_to_id: Dict[str, int],
    country_to_id: Dict[str, int],
    elos_actuales: Dict[str, float],
    historial_actual: Dict[str, list],
    cat_cols: list,
    cont_cols: list,
    poisson_features: list,
    equipos_sedes: Dict[str, dict] = None,
) -> None:
    """Guarda en un único archivo binario todos los pesos neuronales y metadatos precalculados."""
    bundle = {
        "modelo_nn_state_dict": modelo_nn.state_dict(),
        "modelo_poisson_state_dict": modelo_poisson.state_dict(),
        "nn_config": {
            "num_teams": len(team_to_id),
            "num_cities": len(city_to_id),
            "num_countries": len(country_to_id),
        },
        "scaler_nn": scaler_nn,
        "scaler_poisson": scaler_poisson,
        "team_to_id": team_to_id,
        "city_to_id": city_to_id,
        "country_to_id": country_to_id,
        "elos_actuales": elos_actuales,
        "historial_actual": historial_actual,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "poisson_features": poisson_features,
        "equipos_sedes": equipos_sedes or {},
    }
    torch.save(bundle, ruta_guardado)
    print(f"\n[OK] Modelo entrenado y metadatos guardados exitosamente en:\n     {ruta_guardado}")


def cargar_modelo_completo(ruta_modelo: Path = None) -> dict:
    """Carga de forma instantánea los modelos y metadatos guardados en disco."""
    if ruta_modelo is None:
        ruta_modelo = Path(r"c:\Users\rekie\OneDrive\Documentos\Anigrav\modelo_futbol_entrenado.pt")
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {ruta_modelo}")

    bundle = torch.load(ruta_modelo, map_location="cpu", weights_only=False)

    # Reconstruir red neuronal clasificadora
    cfg = bundle["nn_config"]
    num_continuous = len(bundle["cont_cols"])
    modelo_nn = FootballPredictorNN(
        num_teams=cfg["num_teams"],
        num_cities=cfg["num_cities"],
        num_countries=cfg["num_countries"],
        num_continuous=num_continuous,
    )
    modelo_nn.load_state_dict(bundle["modelo_nn_state_dict"])
    modelo_nn.eval()

    # Reconstruir modelo Poisson de goles
    modelo_poisson = PoissonGoalModel(in_features=len(bundle["poisson_features"]))
    modelo_poisson.load_state_dict(bundle["modelo_poisson_state_dict"])
    modelo_poisson.eval()

    return {
        "modelo_nn": modelo_nn,
        "modelo_poisson": modelo_poisson,
        "scaler_nn": bundle["scaler_nn"],
        "scaler_poisson": bundle["scaler_poisson"],
        "team_to_id": bundle["team_to_id"],
        "city_to_id": bundle["city_to_id"],
        "country_to_id": bundle["country_to_id"],
        "elos_actuales": bundle["elos_actuales"],
        "historial_actual": bundle["historial_actual"],
        "cat_cols": bundle["cat_cols"],
        "cont_cols": bundle["cont_cols"],
        "poisson_features": bundle["poisson_features"],
        "equipos_sedes": bundle.get("equipos_sedes", {}),
    }


def calcular_matriz_poisson_dixon_coles(
    lambda_h: float,
    mu_a: float,
    prob_victoria_nn: float,
    max_goles: int = 5,
    rho: float = -0.10,
) -> np.ndarray:
    """Compatibilidad con Dixon-Coles."""
    return calcular_matriz_poisson(
        lambda_h=lambda_h,
        mu_a=mu_a,
        prob_victoria_nn=prob_victoria_nn,
        max_goles=max_goles,
        metodo="dixon_coles",
        rho=rho,
    )


def comparar_poisson_vs_dixon_coles(
    test_df: pd.DataFrame,
    max_goles: int = 5,
) -> None:
    """
    Compara empíricamente la precisión de Poisson Puro frente a Poisson con Dixon-Coles
    en los 5 resultados más probables sobre el conjunto de prueba.
    """
    total = len(test_df)
    counts_puro = {k: 0 for k in range(1, 6)}
    counts_dc = {k: 0 for k in range(1, 6)}

    counts_21_puro = 0
    counts_21_dc = 0
    counts_12_puro = 0
    counts_12_dc = 0

    prob_sum_puro = []
    prob_sum_dc = []

    for row in test_df.itertuples():
        act = (int(row.home_score), int(row.away_score))
        lh, la, p_nn = row.lambda_h, row.mu_a, row.nn_prob

        # 1. Poisson Puro
        mat_p = calcular_matriz_poisson(lh, la, max_goles=max_goles, metodo="poisson")
        scores_p = [((i, j), mat_p[i, j]) for i in range(max_goles + 1) for j in range(max_goles + 1)]
        scores_p.sort(key=lambda x: x[1], reverse=True)
        top5_p = [s[0] for s in scores_p[:5]]
        prob_sum_puro.append(sum(s[1] for s in scores_p[:5]))

        for k in range(1, 6):
            if act in top5_p[:k]:
                counts_puro[k] += 1

        if act == (2, 1) and (2, 1) in top5_p:
            counts_21_puro += 1
        if act == (1, 2) and (1, 2) in top5_p:
            counts_12_puro += 1

        # 2. Dixon-Coles
        mat_dc = calcular_matriz_poisson(lh, la, prob_victoria_nn=p_nn, max_goles=max_goles, metodo="dixon_coles")
        scores_dc = [((i, j), mat_dc[i, j]) for i in range(max_goles + 1) for j in range(max_goles + 1)]
        scores_dc.sort(key=lambda x: x[1], reverse=True)
        top5_dc = [s[0] for s in scores_dc[:5]]
        prob_sum_dc.append(sum(s[1] for s in scores_dc[:5]))

        for k in range(1, 6):
            if act in top5_dc[:k]:
                counts_dc[k] += 1

        if act == (2, 1) and (2, 1) in top5_dc:
            counts_21_dc += 1
        if act == (1, 2) and (1, 2) in top5_dc:
            counts_12_dc += 1

    mask_21 = ((test_df["home_score"] == 2) & (test_df["away_score"] == 1)).sum()
    mask_12 = ((test_df["home_score"] == 1) & (test_df["away_score"] == 2)).sum()

    print("\n" + "=" * 74)
    print(f"TABLA COMPARATIVA: POISSON PURO vs POISSON CON DIXON-COLES ({total:,} partidos)")
    print("=" * 74)
    print(f"{'Métrica':<28} | {'Poisson Puro':<16} | {'Dixon-Coles':<16} | {'Diferencia':<10}")
    print("-" * 74)
    for k in range(1, 6):
        acc_p = counts_puro[k] / total * 100
        acc_dc = counts_dc[k] / total * 100
        diff = acc_p - acc_dc
        diff_str = f"{diff:+.2f}%"
        print(f"{f'Top-{k} Accuracy':<28} | {acc_p:6.2f}% ({counts_puro[k]:,}) | {acc_dc:6.2f}% ({counts_dc[k]:,}) | {diff_str:<10}")

    print("-" * 74)
    p_21_puro = counts_21_puro / mask_21 * 100 if mask_21 > 0 else 0
    p_21_dc = counts_21_dc / mask_21 * 100 if mask_21 > 0 else 0
    print(f"{'Top-5 en resultado 2-1':<28} | {p_21_puro:6.2f}% ({counts_21_puro:,}) | {p_21_dc:6.2f}% ({counts_21_dc:,}) | {p_21_puro - p_21_dc:+.2f}%")

    p_12_puro = counts_12_puro / mask_12 * 100 if mask_12 > 0 else 0
    p_12_dc = counts_12_dc / mask_12 * 100 if mask_12 > 0 else 0
    print(f"{'Top-5 en resultado 1-2':<28} | {p_12_puro:6.2f}% ({counts_12_puro:,}) | {p_12_dc:6.2f}% ({counts_12_dc:,}) | {p_12_puro - p_12_dc:+.2f}%")

    tot_2x1_puro = (counts_21_puro + counts_12_puro) / (mask_21 + mask_12) * 100
    tot_2x1_dc = (counts_21_dc + counts_12_dc) / (mask_21 + mask_12) * 100
    print(f"{'Top-5 en 2x1 combinados':<28} | {tot_2x1_puro:6.2f}% ({counts_21_puro+counts_12_puro:,}) | {tot_2x1_dc:6.2f}% ({counts_21_dc+counts_12_dc:,}) | {tot_2x1_puro - tot_2x1_dc:+.2f}%")

    print("-" * 74)
    m_p = np.mean(prob_sum_puro) * 100
    m_dc = np.mean(prob_sum_dc) * 100
    print(f"{'Masa prob. promedio Top 5':<28} | {m_p:6.2f}%          | {m_dc:6.2f}%          | {m_p - m_dc:+.2f}%")
    print("=" * 74)


def visualizar_mapa_calor_y_top5(
    matriz: np.ndarray,
    equipo_local: str,
    equipo_visita: str,
    lambda_h: float,
    mu_a: float,
    prob_victoria_nn: float,
    max_goles: int = 5,
    guardar_ruta: str = "mapa_calor_resultados.png",
    metodo: str = "poisson",
) -> List[Tuple[Tuple[int, int], float]]:
    """
    1. Imprime una cuadrícula ASCII formateada en consola con porcentajes.
    2. Enumera del 1 al 5 los resultados más probables.
    3. Genera y guarda un mapa de calor visual elegante como archivo PNG con Seaborn y Matplotlib.
    """
    nombre_metodo = "POISSON ESTANDAR (INDEPENDIENTE)" if metodo == "poisson" else "POISSON CON DIXON-COLES"
    print("\n" + "=" * 70)
    print(f"[CUADRICULA DE PROBABILIDADES: {nombre_metodo}]")
    print(f"Goles {equipo_local} vs {equipo_visita}")
    print(f"Goles esperados calculados: {equipo_local} = {lambda_h:.2f} | {equipo_visita} = {mu_a:.2f}")
    print("=" * 70)

    # Cabecera de la cuadrícula
    header = f"{'Local \\ Visita':<14} | " + " | ".join([f"  {j} Gol(es)  " for j in range(max_goles + 1)])
    sep = "-" * len(header)
    print(header)
    print(sep)

    for i in range(max_goles + 1):
        fila_str = f"{i} Gol(es) Local | "
        valores = [f"   {matriz[i, j] * 100:5.2f}%   " for j in range(max_goles + 1)]
        fila_str += " | ".join(valores)
        print(fila_str)
    print(sep)

    # Ordenar todos los resultados para obtener el Top 5
    resultados = []
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            resultados.append(((i, j), matriz[i, j]))
    resultados.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "[TOP 5] RESULTADOS MAS PROBABLES (Enumerados del 1 al 5):")
    print("-" * 55)
    for rank, ((gh, ga), prob) in enumerate(resultados[:5], 1):
        tipo = "Victoria Local" if gh > ga else ("Empate" if gh == ga else "Victoria Visita")
        print(f"  {rank}. {equipo_local} {gh} - {ga} {equipo_visita}  -->  {prob * 100:5.2f}%  ({tipo})")
    print("-" * 55)

    # Detalle especial de marcadores 2x1
    prob_21 = matriz[2, 1] * 100
    prob_12 = matriz[1, 2] * 100
    print(f"  [-] Probabilidad especifica de 2 - 1 (Victoria Local):    {prob_21:5.2f}%")
    print(f"  [-] Probabilidad especifica de 1 - 2 (Victoria Visitante): {prob_12:5.2f}%")

    # Resumen 1X2 agregado desde la matriz Poisson
    prob_local = np.tril(matriz, -1).sum() * 100
    prob_empate = np.diag(matriz).sum() * 100
    prob_visita = np.triu(matriz, 1).sum() * 100
    print(f"\n  Resumen 1X2 Poisson: Local: {prob_local:.1f}% | Empate: {prob_empate:.1f}% | Visita: {prob_visita:.1f}%")
    print("=" * 70)

    # Generación y guardado del mapa de calor como imagen PNG
    try:
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
        subtitulo = "Poisson Estandar" if metodo == "poisson" else "Poisson + Dixon-Coles"
        ax.set_title(
            f"Mapa de Calor de Probabilidades de Resultado ({subtitulo})\n"
            f"{equipo_local} (Local) vs {equipo_visita} (Visita)\n"
            f"(xG Local: {lambda_h:.2f} | xG Visita: {mu_a:.2f} | Confianza Red Neuronal: {prob_victoria_nn * 100:.1f}%)",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(f"Goles de {equipo_visita} (Visitante)", fontsize=11, labelpad=10)
        ax.set_ylabel(f"Goles de {equipo_local} (Local)", fontsize=11, labelpad=10)
        plt.tight_layout()
        plt.savefig(guardar_ruta, dpi=200)
        plt.close()
        print(f"[OK] Mapa de calor grafico guardado exitosamente en: {guardar_ruta}")
    except Exception as e:
        print(f"Nota: No se pudo guardar la imagen del mapa de calor: {e}")

    return resultados[:5]


def predecir_partido(
    model: FootballPredictorNN,
    scaler: StandardScaler,
    team_to_id: Dict[str, int],
    city_to_id: Dict[str, int],
    country_to_id: Dict[str, int],
    elos: Dict[str, float],
    history: Dict[str, List[Tuple[int, float, float, int]]],
    equipo_local: str,
    equipo_visita: str,
    ciudad: str,
    pais: str,
    torneo: str,
    neutral: bool = False,
    poisson_model: PoissonGoalModel = None,
    poisson_scaler: StandardScaler = None,
    metodo_poisson: str = "poisson",
) -> Dict[str, any]:
    """
    Realiza una predicción integral del resultado de un partido:
    1. Predicción binaria y probabilidad con la Red Neuronal en PyTorch.
    2. Si se proporciona el modelo Poisson, calcula goles esperados, la cuadrícula de probabilidades,
       el mapa de calor gráfico y la lista de los 5 resultados más probables utilizando el método seleccionado
       ('poisson' puro o 'dixon_coles').
    """
    model.eval()
    device = next(model.parameters()).device

    # Obtener o asignar IDs
    h_id = team_to_id.get(equipo_local, 0)
    a_id = team_to_id.get(equipo_visita, 0)
    c_id = city_to_id.get(ciudad, 0)
    co_id = country_to_id.get(pais, 0)

    # Obtener ELO actual
    eh = elos.get(equipo_local, 1500.0)
    ea = elos.get(equipo_visita, 1500.0)
    elo_diff = eh - ea

    # Historial de últimos 10 partidos
    h_hist = history.get(equipo_local, [])[-10:]
    a_hist = history.get(equipo_visita, [])[-10:]
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
    cont_scaled = scaler.transform(cont_df)

    cat_tensor = torch.tensor([[h_id, a_id, c_id, co_id]], dtype=torch.long).to(device)
    cont_tensor = torch.tensor(cont_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logit = model(cat_tensor, cont_tensor)
        logit_val = logit.item()
        prob_victoria_local = torch.sigmoid(logit).item()

    pronostico = "Gana el equipo local" if prob_victoria_local >= 0.50 else "No gana el equipo local (Empate o Victoria Visitante)"

    resultado = {
        "equipo_local": equipo_local,
        "equipo_visita": equipo_visita,
        "elo_local": round(eh, 1),
        "elo_visita": round(ea, 1),
        "probabilidad_victoria_local": round(prob_victoria_local * 100, 2),
        "pronostico": pronostico,
    }

    # Modelo Poisson (si está disponible)
    if poisson_model is not None and poisson_scaler is not None:
        poisson_model.eval()
        p_df = pd.DataFrame(
            [[logit_val, prob_victoria_local, elo_diff, h_gf, a_gf, h_ga, a_ga, neutral_val, peso]],
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
        p_scaled = poisson_scaler.transform(p_df)
        p_tensor = torch.tensor(p_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            lh, la = poisson_model(p_tensor)
            lambda_h = lh.item()
            mu_a = la.item()

        resultado["goles_esperados_local"] = round(lambda_h, 2)
        resultado["goles_esperados_visita"] = round(mu_a, 2)
        resultado["metodo_poisson"] = metodo_poisson

        # Generar matriz de probabilidades de marcadores exactos con el método seleccionado
        matriz_poisson = calcular_matriz_poisson(
            lambda_h=lambda_h,
            mu_a=mu_a,
            prob_victoria_nn=prob_victoria_local,
            metodo=metodo_poisson,
        )
        top5 = visualizar_mapa_calor_y_top5(
            matriz=matriz_poisson,
            equipo_local=equipo_local,
            equipo_visita=equipo_visita,
            lambda_h=lambda_h,
            mu_a=mu_a,
            prob_victoria_nn=prob_victoria_local,
            metodo=metodo_poisson,
        )
        resultado["top_5_resultados"] = top5

    return resultado


def main():
    print(f"Ruta configurada del dataset: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        print("Advertencia: No se encontró el archivo en la ruta especificada.")
        return

    # 1. Cargar datos
    df = cargar_datos(DATASET_PATH)

    # 2. Limpieza de datos (fechas y marcadores)
    df_limpio = limpiar_fechas_y_datos(df)

    # 3. Cálculo de ELO, Peso Económico y Últimos 10 Partidos
    df_procesado, elos_actuales, historial_actual = calcular_elo_y_ultimos_10_partidos(df_limpio)

    # 4. Asignación de IDs únicos (Equipos, Ciudad, País)
    df_final, team_to_id, city_to_id, country_to_id = asignar_identificadores(df_procesado)

    # 5. Entrenamiento de la Red Neuronal en PyTorch (Clasificador 1X2 / 2x1)
    modelo_nn, scaler_nn, cat_cols, cont_cols = entrenar_modelo(
        df=df_final,
        num_teams=len(team_to_id),
        num_cities=len(city_to_id),
        num_countries=len(country_to_id),
        epocas=15,
        batch_size=256,
        lr=1e-3,
    )

    # 6. Entrenamiento del Modelo Poisson de Goles Esperados (Aprende del modelo clasificador)
    modelo_poisson, scaler_poisson, poisson_features = entrenar_modelo_poisson(
        df=df_final,
        modelo_nn=modelo_nn,
        scaler_nn=scaler_nn,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        epocas=10,
        batch_size=256,
        lr=1e-3,
    )

    # 7. Comparación de Precisión en el Conjunto de Prueba: Poisson Puro vs Poisson Dixon-Coles
    print("\n--- Evaluando Comparativa en Test Set: Poisson Puro vs Dixon-Coles ---")
    split_idx = int(len(df_final) * 0.8)
    test_eval_df = df_final.iloc[split_idx:].copy()

    modelo_nn.eval()
    modelo_poisson.eval()
    device = next(modelo_nn.parameters()).device
    with torch.no_grad():
        test_cats = torch.tensor(test_eval_df[cat_cols].values, dtype=torch.long).to(device)
        test_conts = torch.tensor(scaler_nn.transform(test_eval_df[cont_cols]), dtype=torch.float32).to(device)
        test_logits = modelo_nn(test_cats, test_conts).cpu().numpy()
        test_eval_df["nn_logit"] = test_logits
        test_eval_df["nn_prob"] = 1 / (1 + np.exp(-test_logits))

        p_scaled_eval = scaler_poisson.transform(test_eval_df[poisson_features])
        p_tensor_eval = torch.tensor(p_scaled_eval, dtype=torch.float32).to(device)
        lh_eval, la_eval = modelo_poisson(p_tensor_eval)
        test_eval_df["lambda_h"] = lh_eval.cpu().numpy()
        test_eval_df["mu_a"] = la_eval.cpu().numpy()

    comparar_poisson_vs_dixon_coles(test_eval_df)

    # 8. Extracción de Sedes Habituales de cada Equipo
    sedes_dict = {}
    for equipo, group in df_limpio.groupby("home_team"):
        c_mode = group["city"].mode()
        co_mode = group["country"].mode()
        sedes_dict[equipo] = {
            "city": c_mode.iloc[0] if not c_mode.empty else "Principal",
            "country": co_mode.iloc[0] if not co_mode.empty else equipo,
        }

    # 9. Guardar el Modelo Entrenado y Metadatos en Disco para Simulaciones Instantáneas
    ruta_guardado = Path(r"c:\Users\rekie\OneDrive\Documentos\Anigrav\modelo_futbol_entrenado.pt")
    guardar_modelo_completo(
        ruta_guardado=ruta_guardado,
        modelo_nn=modelo_nn,
        modelo_poisson=modelo_poisson,
        scaler_nn=scaler_nn,
        scaler_poisson=scaler_poisson,
        team_to_id=team_to_id,
        city_to_id=city_to_id,
        country_to_id=country_to_id,
        elos_actuales=elos_actuales,
        historial_actual=historial_actual,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        poisson_features=poisson_features,
        equipos_sedes=sedes_dict,
    )

    # 10. Demostración de Predicción Integral con el Modelo Más Preciso (Dixon-Coles)
    print("\n--- Demostración con el Modelo Más Preciso (Dixon-Coles Calibrado) ---")
    ejemplo = predecir_partido(
        model=modelo_nn,
        scaler=scaler_nn,
        team_to_id=team_to_id,
        city_to_id=city_to_id,
        country_to_id=country_to_id,
        elos=elos_actuales,
        history=historial_actual,
        equipo_local="Argentina",
        equipo_visita="Brazil",
        ciudad="Buenos Aires",
        pais="Argentina",
        torneo="FIFA World Cup qualification",
        neutral=False,
        poisson_model=modelo_poisson,
        poisson_scaler=scaler_poisson,
        metodo_poisson="dixon_coles",
    )

    print(f"\nResumen:")
    print(f"Partido: {ejemplo['equipo_local']} (ELO: {ejemplo['elo_local']}) vs {ejemplo['equipo_visita']} (ELO: {ejemplo['elo_visita']})")
    print(f"Probabilidad de Victoria Local (Red Neuronal): {ejemplo['probabilidad_victoria_local']}%")
    print(f"Pronóstico de la Red Neuronal: {ejemplo['pronostico']}")
    print(f"Goles esperados xG: {ejemplo['equipo_local']} {ejemplo['goles_esperados_local']} - {ejemplo['goles_esperados_visita']} {ejemplo['equipo_visita']}")
    print(f"Método de cuadrícula utilizado: {ejemplo.get('metodo_poisson')}")


if __name__ == "__main__":
    main()




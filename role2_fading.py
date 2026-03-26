"""
Rôle 2 — Modélisation du canal radio (Fading)
Projet : Modélisation cross-layer des facteurs d'instabilité IoT
Modèles : Log-distance + Rayleigh fading
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ─── Paramètres ───────────────────────────────────────
FREQ_HZ       = 2.4e9       # Fréquence (2.4 GHz — Wi-Fi / Zigbee)
TX_POWER_DBM  = 20.0        # Puissance d'émission (dBm)
NOISE_DBM     = -100.0      # Bruit thermique (dBm)
PATH_LOSS_EXP = 3.0         # Exposant de perte (2=espace libre, 3-4=urbain)
D0            = 1.0         # Distance de référence (m)
SIGMA_SHADOW  = 4.0         # Écart-type shadowing (dB)
SIM_DURATION  = 120.0       # Durée simulation (s)
DT            = 0.5         # Pas de temps (s)
DISTANCES     = [10, 20, 40, 60, 80, 100]  # Distances à tester (m)
SEED          = 42

rng = np.random.default_rng(SEED)
time_steps = np.arange(0, SIM_DURATION, DT)

# ─── Modèle Log-Distance ──────────────────────────────
def path_loss_db(d, d0=D0, n=PATH_LOSS_EXP, freq=FREQ_HZ):
    """Perte de trajet log-distance (dB)."""
    lambda_ = 3e8 / freq
    PL0 = 20 * np.log10(4 * np.pi * d0 / lambda_)
    return PL0 + 10 * n * np.log10(d / d0)

# ─── Fading Rayleigh ──────────────────────────────────
def rayleigh_fading_db(n_samples, sigma=1.0):
    """Génère n_samples de fading Rayleigh en dB."""
    i = rng.normal(0, sigma, n_samples)
    q = rng.normal(0, sigma, n_samples)
    envelope = np.sqrt(i**2 + q**2)
    return 20 * np.log10(envelope / np.mean(envelope) + 1e-10)

# ─── Shadowing log-normal ─────────────────────────────
def shadowing_db(n_samples, sigma=SIGMA_SHADOW):
    return rng.normal(0, sigma, n_samples)

# ─── Simulation RSSI/SNR par distance ─────────────────
def simulate_channel():
    results = {}
    for d in DISTANCES:
        pl = path_loss_db(d)
        fading = rayleigh_fading_db(len(time_steps))
        shadow = shadowing_db(len(time_steps))
        rssi = TX_POWER_DBM - pl + fading + shadow
        snr  = rssi - NOISE_DBM
        results[d] = {'rssi': rssi, 'snr': snr}
    return results

results = simulate_channel()

# ─── Export CSV ───────────────────────────────────────
rows = []
for d, data in results.items():
    for i, t in enumerate(time_steps):
        rows.append({'time_s': round(t,2), 'distance_m': d,
                     'rssi_dbm': round(data['rssi'][i],3),
                     'snr_db':   round(data['snr'][i],3)})
df = pd.DataFrame(rows)
df.to_csv('/home/claude/IoT-CrossLayer/results/raw_data/channel_data.csv', index=False)

# ─── Graphique 1 : RSSI(t) pour plusieurs distances ──
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(DISTANCES)))

for d, color in zip(DISTANCES, colors):
    axes[0].plot(time_steps, results[d]['rssi'], color=color, alpha=0.8,
                 linewidth=1.0, label=f'd={d}m')
    axes[1].plot(time_steps, results[d]['snr'],  color=color, alpha=0.8,
                 linewidth=1.0, label=f'd={d}m')

axes[0].axhline(-70, color='red', linestyle='--', linewidth=1.2, label='Seuil RSSI (-70 dBm)')
axes[0].set_ylabel('RSSI (dBm)', fontsize=11)
axes[0].set_title('RSSI(t) — Modèle log-distance + Fading Rayleigh', fontsize=13)
axes[0].legend(fontsize=8, ncol=4); axes[0].grid(True, alpha=0.3)

axes[1].axhline(10, color='orange', linestyle='--', linewidth=1.2, label='SNR min (10 dB)')
axes[1].set_ylabel('SNR (dB)', fontsize=11)
axes[1].set_xlabel('Temps (s)', fontsize=11)
axes[1].set_title('SNR(t)', fontsize=13)
axes[1].legend(fontsize=8, ncol=4); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/rssi_snr_time.png', dpi=150)
plt.close()

# ─── Graphique 2 : RSSI moyen vs distance ────────────
mean_rssi = [np.mean(results[d]['rssi']) for d in DISTANCES]
std_rssi  = [np.std(results[d]['rssi'])  for d in DISTANCES]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(DISTANCES, mean_rssi, yerr=std_rssi, fmt='-o', color='steelblue',
            capsize=5, linewidth=2, label='RSSI moyen ± std')
ax.axhline(-70, color='red', linestyle='--', label='Seuil réception (-70 dBm)')
ax.set_xlabel('Distance (m)', fontsize=11)
ax.set_ylabel('RSSI moyen (dBm)', fontsize=11)
ax.set_title('RSSI moyen en fonction de la distance\n(Log-distance + Rayleigh)', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/rssi_vs_distance.png', dpi=150)
plt.close()
print("✅ Rôle 2 (Fading) — OK")

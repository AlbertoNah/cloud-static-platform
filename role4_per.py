"""
Rôle 4 — Modélisation du PER (Packet Error Rate)
Projet : Modélisation cross-layer des facteurs d'instabilité IoT
Dépend du Rôle 2 (fading) — utilise channel_data.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ─── Modèle PER basé sur SNR ──────────────────────────
# Formule BER BPSK en canal AWGN : BER = Q(sqrt(2*SNR_lin))
# PER = 1 - (1 - BER)^N  avec N = taille du paquet en bits

PACKET_SIZE_BITS = 256    # Taille d'un paquet IoT typique (bits)
DISTANCES = [10, 20, 40, 60, 80, 100]
SEED = 42
rng = np.random.default_rng(SEED)

def q_function(x):
    """Approximation de la fonction Q."""
    from scipy.special import erfc
    return 0.5 * erfc(x / np.sqrt(2))

def snr_to_per(snr_db, n_bits=PACKET_SIZE_BITS):
    """Convertit SNR (dB) en PER via modèle BPSK."""
    try:
        from scipy.special import erfc
    except ImportError:
        # Approximation sans scipy
        def erfc(x): return 2/(1+np.exp(2.5*x))
    snr_lin = 10 ** (snr_db / 10)
    ber = 0.5 * erfc(np.sqrt(snr_lin))
    per = 1 - (1 - ber) ** n_bits
    return np.clip(per, 0, 1)

# ─── Charger les données du canal ────────────────────
try:
    df_channel = pd.read_csv('/home/claude/IoT-CrossLayer/results/raw_data/channel_data.csv')
    USE_CSV = True
except FileNotFoundError:
    USE_CSV = False

# ─── Calcul PER ───────────────────────────────────────
results_per = {}
if USE_CSV:
    for d in DISTANCES:
        subset = df_channel[df_channel['distance_m'] == d]
        snr_vals = subset['snr_db'].values
        per_vals = snr_to_per(snr_vals)
        results_per[d] = {'time': subset['time_s'].values, 'per': per_vals, 'snr': snr_vals}
else:
    # Génération directe si CSV absent
    time_steps = np.arange(0, 120, 0.5)
    for d in DISTANCES:
        snr_base = 30 - 10*3*np.log10(max(d,1))
        snr_vals = snr_base + rng.normal(0, 4, len(time_steps))
        results_per[d] = {'time': time_steps, 'per': snr_to_per(snr_vals), 'snr': snr_vals}

# ─── Export CSV ───────────────────────────────────────
rows = []
for d, data in results_per.items():
    for t, per, snr in zip(data['time'], data['per'], data['snr']):
        rows.append({'time_s': round(t,2), 'distance_m': d,
                     'snr_db': round(snr,3), 'per': round(per,6)})
pd.DataFrame(rows).to_csv('/home/claude/IoT-CrossLayer/results/raw_data/per_data.csv', index=False)

# ─── Graphique 1 : PER(t) par distance ───────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(DISTANCES)))
for d, color in zip(DISTANCES, colors):
    ax.plot(results_per[d]['time'], results_per[d]['per'],
            color=color, alpha=0.8, linewidth=1.0, label=f'd={d}m')
ax.axhline(0.1, color='red', linestyle='--', linewidth=1.5, label='Seuil PER critique (10%)')
ax.set_xlabel('Temps (s)', fontsize=11)
ax.set_ylabel('PER', fontsize=11)
ax.set_title('Packet Error Rate en fonction du temps\n(pour différentes distances)', fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/per_over_time.png', dpi=150); plt.close()

# ─── Graphique 2 : PER moyen vs distance ─────────────
mean_per = [np.mean(results_per[d]['per']) for d in DISTANCES]
std_per  = [np.std(results_per[d]['per'])  for d in DISTANCES]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(DISTANCES, mean_per, yerr=std_per, fmt='-o', color='crimson',
            capsize=5, linewidth=2, label='PER moyen ± std')
ax.axhline(0.1, color='orange', linestyle='--', label='Seuil critique (10%)')
ax.set_xlabel('Distance (m)', fontsize=11)
ax.set_ylabel('PER moyen', fontsize=11)
ax.set_title('PER moyen en fonction de la distance', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/per_vs_distance.png', dpi=150); plt.close()
print("✅ Rôle 4 (PER) — OK")

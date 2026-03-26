"""
Semaine 4 — Intégration cross-layer complète
Projet : Modélisation cross-layer des facteurs d'instabilité IoT
Combine : Fading (S2) + Mobilité (S2) + RPL (S3) + Sécurité (S3)
Scénarios réalistes : urbain + attaques combinées
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erfc
import os

SEED = 42
rng  = np.random.default_rng(SEED)
TIME = np.arange(0, 180, 0.5)   # 3 minutes — simulation complète
NUM_NODES = 7
DISTANCES = [10, 20, 35, 50, 65, 80, 95]

os.makedirs('/home/claude/IoT-CrossLayer/results/figures', exist_ok=True)
os.makedirs('/home/claude/IoT-CrossLayer/results/raw_data', exist_ok=True)

# ══════════════════════════════════════════════════════════
# HELPERS — modèles des semaines précédentes
# ══════════════════════════════════════════════════════════
def path_loss(d, n=3.0, freq=2.4e9):
    lam = 3e8/freq
    pl0 = 20*np.log10(4*np.pi*1/lam)
    return pl0 + 10*n*np.log10(max(d,0.1))

def rssi_series(d, n_pts):
    pl = path_loss(d)
    fading = 20*np.log10(np.abs(rng.normal(0,1,n_pts)+1j*rng.normal(0,1,n_pts))+1e-10)
    shadow = rng.normal(0, 4, n_pts)
    return 20 - pl + fading + shadow

def per_from_snr(snr_db, n=256):
    snr_lin = 10**(snr_db/10)
    ber = 0.5*erfc(np.sqrt(np.maximum(snr_lin,0)))
    return np.clip(1-(1-ber)**n, 0, 1)

def mobility_distance(t_arr, base_d=50, amplitude=35, period=90):
    """Distance variable simulant la mobilité RWP de façon simplifiée."""
    noise = rng.normal(0, 5, len(t_arr))
    return np.clip(base_d + amplitude*np.sin(2*np.pi*t_arr/period) + noise, 1, 100)

# ══════════════════════════════════════════════════════════
# SCÉNARIO A — Urbain sans attaque (référence)
# ══════════════════════════════════════════════════════════
print("🏙️  Scénario A : Urbain sans attaque...")
scenario_A = {}
for d in DISTANCES:
    rssi  = rssi_series(d, len(TIME))
    snr   = rssi - (-100)
    per   = per_from_snr(snr)
    delay = 0.01*(1 + per*3) + rng.normal(0, 0.002, len(TIME))
    stab  = np.array([np.mean(per[max(0,i-20):i+1] < 0.1) for i in range(len(TIME))])
    energy= 1000 - np.cumsum(0.5 + per*0.3 + rng.uniform(0, 0.1, len(TIME)))
    energy= np.maximum(energy, 0)
    scenario_A[d] = {'rssi':rssi,'snr':snr,'per':per,'delay':delay,'stability':stab,'energy':energy}

# ══════════════════════════════════════════════════════════
# SCÉNARIO B — Urbain avec mobilité
# ══════════════════════════════════════════════════════════
print("🚶 Scénario B : Urbain + mobilité...")
scenario_B = {}
for i, base_d in enumerate(DISTANCES):
    dist_t = mobility_distance(TIME, base_d=base_d, amplitude=20, period=60+i*10)
    rssi   = np.array([rssi_series(d,1)[0] for d in dist_t])
    snr    = rssi - (-100)
    per    = per_from_snr(snr)
    delay  = 0.01*(1 + per*3) + rng.normal(0, 0.003, len(TIME))
    stab   = np.array([np.mean(per[max(0,j-20):j+1] < 0.1) for j in range(len(TIME))])
    energy = 1000 - np.cumsum(0.5 + per*0.3 + 0.2 + rng.uniform(0,0.15,len(TIME)))
    energy = np.maximum(energy, 0)
    scenario_B[base_d] = {'dist':dist_t,'rssi':rssi,'per':per,'delay':delay,'stability':stab,'energy':energy}

# ══════════════════════════════════════════════════════════
# SCÉNARIO C — Attaque DDoS + mobilité (pire cas)
# ══════════════════════════════════════════════════════════
print("⚠️  Scénario C : DDoS + mobilité...")
DDOS_S, DDOS_E = 60, 120
scenario_C = {}
for i, base_d in enumerate(DISTANCES):
    dist_t = mobility_distance(TIME, base_d=base_d, amplitude=25, period=70+i*8)
    rssi   = np.array([rssi_series(d,1)[0] for d in dist_t])
    snr    = rssi - (-100)
    per    = per_from_snr(snr)

    # Pendant DDoS : PER augmente, délai explose
    ddos_mask = (TIME >= DDOS_S) & (TIME <= DDOS_E)
    per_c  = per.copy()
    per_c[ddos_mask] = np.clip(per_c[ddos_mask] + rng.uniform(0.1,0.4,ddos_mask.sum()), 0, 1)

    delay  = 0.01*(1 + per_c*5) + rng.normal(0, 0.005, len(TIME))
    delay[ddos_mask] *= 3

    stab   = np.array([np.mean(per_c[max(0,j-20):j+1] < 0.1) for j in range(len(TIME))])
    energy = 1000 - np.cumsum(0.5 + per_c*0.3 + 0.2 + ddos_mask*2 + rng.uniform(0,0.2,len(TIME)))
    energy = np.maximum(energy, 0)
    scenario_C[base_d] = {'dist':dist_t,'per':per_c,'delay':delay,'stability':stab,'energy':energy,'rssi':rssi}

# ══════════════════════════════════════════════════════════
# GRAPHIQUES CROSS-LAYER
# ══════════════════════════════════════════════════════════
colors = plt.cm.tab10(np.linspace(0,1,NUM_NODES))

# ── G1 : Comparaison PER — 3 scénarios ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Scénario A\n(Urbain — référence)', 'Scénario B\n(Urbain + mobilité)', 'Scénario C\n(DDoS + mobilité)']
for ax, sc, title in zip(axes, [scenario_A, scenario_B, scenario_C], titles):
    for d, c in zip(DISTANCES, colors):
        data = sc[d]
        ax.plot(TIME, data['per'], color=c, alpha=0.7, linewidth=1, label=f'{d}m')
    if 'C' in title:
        ax.axvspan(DDOS_S, DDOS_E, alpha=0.12, color='red', label='DDoS')
    ax.axhline(0.1, color='red', linestyle='--', linewidth=1)
    ax.set_title(title, fontsize=11); ax.set_xlabel('Temps (s)'); ax.set_ylabel('PER')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
fig.suptitle('PER cross-layer — Comparaison des 3 scénarios', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s4_per_comparison.png', dpi=150)
plt.close()

# ── G2 : Délai moyen — comparaison scénarios ─────────────
fig, ax = plt.subplots(figsize=(12, 5))
means_A = [np.mean(scenario_A[d]['delay']) for d in DISTANCES]
means_B = [np.mean(scenario_B[d]['delay']) for d in DISTANCES]
means_C = [np.mean(scenario_C[d]['delay']) for d in DISTANCES]
x = np.arange(NUM_NODES); w = 0.28
ax.bar(x-w, means_A, w, label='Scénario A (référence)', color='steelblue', alpha=0.85)
ax.bar(x,   means_B, w, label='Scénario B (+ mobilité)', color='seagreen', alpha=0.85)
ax.bar(x+w, means_C, w, label='Scénario C (DDoS+mobilité)', color='crimson', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f'{d}m' for d in DISTANCES])
ax.set_xlabel('Distance à la base'); ax.set_ylabel('Délai moyen (s)')
ax.set_title('Délai moyen par nœud — Comparaison cross-layer des 3 scénarios', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s4_delay_comparison.png', dpi=150)
plt.close()

# ── G3 : Énergie résiduelle — 3 scénarios ────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, sc, title in zip(axes, [scenario_A, scenario_B, scenario_C], titles):
    for d, c in zip(DISTANCES, colors):
        ax.plot(TIME, sc[d]['energy'], color=c, linewidth=1.2, label=f'{d}m')
    if 'C' in title:
        ax.axvspan(DDOS_S, DDOS_E, alpha=0.1, color='red')
    ax.axhline(100, color='red', linestyle=':', linewidth=1.5, label='Seuil critique')
    ax.set_title(title, fontsize=11); ax.set_xlabel('Temps (s)'); ax.set_ylabel('Énergie (mJ)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle('Énergie résiduelle — Impact cross-layer', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s4_energy_comparison.png', dpi=150)
plt.close()

# ── G4 : Stabilité du lien — radar / heatmap ─────────────
fig, ax = plt.subplots(figsize=(12, 5))
stab_A = [np.mean(scenario_A[d]['stability']) for d in DISTANCES]
stab_B = [np.mean(scenario_B[d]['stability']) for d in DISTANCES]
stab_C = [np.mean(scenario_C[d]['stability']) for d in DISTANCES]
ax.plot(DISTANCES, stab_A, '-o', color='steelblue', linewidth=2, label='Scénario A')
ax.plot(DISTANCES, stab_B, '-s', color='seagreen',  linewidth=2, label='Scénario B')
ax.plot(DISTANCES, stab_C, '-^', color='crimson',   linewidth=2, label='Scénario C')
ax.axhline(0.5, color='orange', linestyle='--', label='Seuil stabilité min')
ax.fill_between(DISTANCES, stab_A, stab_C, alpha=0.1, color='red', label='Dégradation DDoS')
ax.set_xlabel('Distance à la base (m)'); ax.set_ylabel('Stabilité moyenne')
ax.set_title('Stabilité du lien — Analyse de sensibilité cross-layer', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s4_stability_sensitivity.png', dpi=150)
plt.close()

# ── G5 : Heatmap métriques cross-layer ───────────────────
metrics = {
    'PER moyen (A)':     [np.mean(scenario_A[d]['per']) for d in DISTANCES],
    'PER moyen (C)':     [np.mean(scenario_C[d]['per']) for d in DISTANCES],
    'Délai moyen (A)':   [np.mean(scenario_A[d]['delay'])*100 for d in DISTANCES],
    'Délai moyen (C)':   [np.mean(scenario_C[d]['delay'])*100 for d in DISTANCES],
    'Stabilité (A)':     [np.mean(scenario_A[d]['stability']) for d in DISTANCES],
    'Stabilité (C)':     [np.mean(scenario_C[d]['stability']) for d in DISTANCES],
}
df_heat = pd.DataFrame(metrics, index=[f'{d}m' for d in DISTANCES])
# Normaliser pour la heatmap
df_norm = (df_heat - df_heat.min()) / (df_heat.max() - df_heat.min())

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(df_norm.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(DISTANCES))); ax.set_xticklabels([f'{d}m' for d in DISTANCES])
ax.set_yticks(range(len(metrics))); ax.set_yticklabels(list(metrics.keys()))
plt.colorbar(im, ax=ax, label='Valeur normalisée (0=bon, 1=critique)')
ax.set_title('Heatmap cross-layer — Vue d\'ensemble des métriques (normalisées)', fontsize=13)
for i in range(len(metrics)):
    for j in range(len(DISTANCES)):
        ax.text(j, i, f'{df_norm.iloc[j,i]:.2f}', ha='center', va='center', fontsize=9,
                color='black' if 0.3<df_norm.iloc[j,i]<0.7 else 'white')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s4_crosslayer_heatmap.png', dpi=150)
plt.close()

# ── Export CSV ────────────────────────────────────────────
rows = []
for d in DISTANCES:
    for t_idx, t in enumerate(TIME):
        rows.append({
            'time_s': round(t,2), 'distance_m': d,
            'per_A':     round(float(scenario_A[d]['per'][t_idx]),4),
            'per_C':     round(float(scenario_C[d]['per'][t_idx]),4),
            'delay_A':   round(float(scenario_A[d]['delay'][t_idx]),5),
            'delay_C':   round(float(scenario_C[d]['delay'][t_idx]),5),
            'energy_A':  round(float(scenario_A[d]['energy'][t_idx]),2),
            'energy_C':  round(float(scenario_C[d]['energy'][t_idx]),2),
            'stab_A':    round(float(scenario_A[d]['stability'][t_idx]),3),
            'stab_C':    round(float(scenario_C[d]['stability'][t_idx]),3),
        })
pd.DataFrame(rows).to_csv('/home/claude/IoT-CrossLayer/results/raw_data/s4_crosslayer_data.csv', index=False)

print("✅ Semaine 4 (Intégration cross-layer) — OK")
print("   Graphiques : s4_per_comparison, s4_delay_comparison, s4_energy_comparison,")
print("                s4_stability_sensitivity, s4_crosslayer_heatmap")

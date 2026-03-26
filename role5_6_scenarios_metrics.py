"""
Rôle 5 — Scénarios simples
Rôle 6 — Analyse des métriques (taux de perte, délai, stabilité)
Projet : Modélisation cross-layer des facteurs d'instabilité IoT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erfc

SEED = 42
rng = np.random.default_rng(SEED)
DT = 0.5
TIME = np.arange(0, 120, DT)
PACKET_SIZE = 256

# ─── Helpers ──────────────────────────────────────────
def path_loss(d, n=3.0, freq=2.4e9):
    lam = 3e8/freq
    pl0 = 20*np.log10(4*np.pi*1/lam)
    return pl0 + 10*n*np.log10(max(d,0.1))

def rssi(d, tx=20, shadow_std=4):
    pl = path_loss(d)
    fading = 20*np.log10(np.abs(rng.normal(0,1,len(TIME))+1j*rng.normal(0,1,len(TIME)))+1e-10)
    shadow = rng.normal(0, shadow_std, len(TIME))
    return tx - pl + fading + shadow

def per_from_snr(snr_db, n=PACKET_SIZE):
    snr_lin = 10**(snr_db/10)
    ber = 0.5*erfc(np.sqrt(np.maximum(snr_lin,0)))
    return np.clip(1-(1-ber)**n, 0, 1)

def delay_model(per, base_delay=0.01, max_retx=3):
    """Délai = délai base + retransmissions si perte."""
    retx = np.minimum(np.floor(per * max_retx * rng.uniform(0.5,1.5,len(per))), max_retx)
    return base_delay * (1 + retx) + rng.normal(0, 0.002, len(per))

def link_stability(rssi_vals, threshold=-70):
    """Stabilité = % du temps au-dessus du seuil dans une fenêtre glissante."""
    window = 20
    stability = []
    for i in range(len(rssi_vals)):
        start = max(0, i-window)
        window_vals = rssi_vals[start:i+1]
        stability.append(np.mean(np.array(window_vals) > threshold))
    return np.array(stability)

# ══════════════════════════════════════════════════════
# SCÉNARIO 1 — 2 nœuds fixes à différentes distances
# ══════════════════════════════════════════════════════
print("📡 Scénario 1 : 2 nœuds fixes")
distances_s1 = [20, 80]
s1_data = {}
for d in distances_s1:
    r = rssi(d)
    snr = r - (-100)
    p = per_from_snr(snr)
    delay = delay_model(p)
    stab = link_stability(r)
    s1_data[d] = {'rssi':r,'snr':snr,'per':p,'delay':delay,'stability':stab}

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
colors = ['steelblue','crimson']
labels = [f'Nœud fixe d={d}m' for d in distances_s1]

for (d,c,lbl) in zip(distances_s1, colors, labels):
    axes[0,0].plot(TIME, s1_data[d]['rssi'],   color=c, alpha=0.8, linewidth=1, label=lbl)
    axes[0,1].plot(TIME, s1_data[d]['per'],    color=c, alpha=0.8, linewidth=1, label=lbl)
    axes[1,0].plot(TIME, s1_data[d]['delay'],  color=c, alpha=0.8, linewidth=1, label=lbl)
    axes[1,1].plot(TIME, s1_data[d]['stability'], color=c, alpha=0.8, linewidth=1, label=lbl)

axes[0,0].axhline(-70,color='k',linestyle='--',linewidth=1,label='Seuil -70dBm')
axes[0,1].axhline(0.1,color='k',linestyle='--',linewidth=1,label='Seuil 10%')
titles = ['RSSI (dBm)','PER','Délai (s)','Stabilité du lien']
for ax,t in zip(axes.flat, titles):
    ax.set_title(t,fontsize=11); ax.set_xlabel('Temps (s)'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

fig.suptitle('Scénario 1 — 2 nœuds fixes : impact de la distance', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/scenario1_fixed_nodes.png', dpi=150); plt.close()

# ══════════════════════════════════════════════════════
# SCÉNARIO 2 — 1 nœud fixe + 1 nœud mobile
# ══════════════════════════════════════════════════════
print("📡 Scénario 2 : 1 fixe + 1 mobile")
# Nœud fixe à 30m
fixed_rssi = rssi(30)
# Nœud mobile : distance variable sinusoïdale (simulation RWP simplifiée)
mobile_dist = 30 + 40*np.abs(np.sin(2*np.pi*TIME/60)) + rng.normal(0,5,len(TIME))
mobile_dist = np.clip(mobile_dist, 1, 100)
mobile_rssi_vals = np.array([rssi(d)[0] for d in mobile_dist])
# Pour la mobilité on recalcule proprement
mobile_rssi_vals = np.array([20 - path_loss(d) + rng.normal(0,4) for d in mobile_dist])

s2_data = {
    'fixed':  {'rssi': fixed_rssi, 'dist': np.full(len(TIME),30)},
    'mobile': {'rssi': mobile_rssi_vals, 'dist': mobile_dist}
}
for k in s2_data:
    snr = s2_data[k]['rssi'] - (-100)
    s2_data[k]['per']      = per_from_snr(snr)
    s2_data[k]['stability']= link_stability(s2_data[k]['rssi'])
    s2_data[k]['delay']    = delay_model(s2_data[k]['per'])

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
clrs = {'fixed':'steelblue','mobile':'orangered'}
lbls = {'fixed':'Nœud fixe (30m)','mobile':'Nœud mobile'}

for k in ['fixed','mobile']:
    axes[0,0].plot(TIME, s2_data[k]['dist'],      color=clrs[k], label=lbls[k])
    axes[0,1].plot(TIME, s2_data[k]['rssi'],      color=clrs[k], label=lbls[k])
    axes[1,0].plot(TIME, s2_data[k]['per'],       color=clrs[k], label=lbls[k])
    axes[1,1].plot(TIME, s2_data[k]['stability'], color=clrs[k], label=lbls[k])

axes[0,1].axhline(-70,color='k',linestyle='--',linewidth=1)
axes[1,0].axhline(0.1,color='k',linestyle='--',linewidth=1)
titles = ['Distance à la base (m)','RSSI (dBm)','PER','Stabilité du lien']
for ax,t in zip(axes.flat,titles):
    ax.set_title(t,fontsize=11); ax.set_xlabel('Temps (s)'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

fig.suptitle('Scénario 2 — Nœud fixe vs nœud mobile', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/scenario2_fixed_vs_mobile.png', dpi=150); plt.close()

# ══════════════════════════════════════════════════════
# SCÉNARIO 3 — Petit réseau (7 nœuds)
# ══════════════════════════════════════════════════════
print("📡 Scénario 3 : petit réseau 7 nœuds")
NODE_DISTANCES = [10, 20, 35, 50, 65, 80, 95]
NODE_LABELS    = [f'N{i}' for i in range(len(NODE_DISTANCES))]

s3_data = {}
for d,lbl in zip(NODE_DISTANCES, NODE_LABELS):
    r = rssi(d)
    snr = r - (-100)
    p = per_from_snr(snr)
    s3_data[lbl] = {
        'dist':d, 'rssi_mean':np.mean(r), 'rssi_std':np.std(r),
        'per_mean':np.mean(p), 'per_std':np.std(p),
        'stability_mean': np.mean(link_stability(r)),
        'delay_mean': np.mean(delay_model(p))
    }

df_s3 = pd.DataFrame(s3_data).T
df_s3['node'] = df_s3.index

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].bar(NODE_LABELS, df_s3['rssi_mean'], color='steelblue', alpha=0.8,
            yerr=df_s3['rssi_std'], capsize=4)
axes[0].axhline(-70,color='red',linestyle='--'); axes[0].set_title('RSSI moyen par nœud'); axes[0].set_ylabel('RSSI (dBm)')

axes[1].bar(NODE_LABELS, df_s3['per_mean'], color='crimson', alpha=0.8,
            yerr=df_s3['per_std'], capsize=4)
axes[1].axhline(0.1,color='orange',linestyle='--'); axes[1].set_title('PER moyen par nœud'); axes[1].set_ylabel('PER')

axes[2].bar(NODE_LABELS, df_s3['stability_mean'], color='seagreen', alpha=0.8)
axes[2].set_title('Stabilité du lien par nœud'); axes[2].set_ylabel('Stabilité (0-1)')

for ax in axes: ax.grid(True,alpha=0.3,axis='y'); ax.set_xlabel('Nœud')
fig.suptitle('Scénario 3 — Petit réseau IoT (7 nœuds)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/scenario3_network.png', dpi=150); plt.close()

# ─── Export CSV global ────────────────────────────────
df_s3.to_csv('/home/claude/IoT-CrossLayer/results/raw_data/scenario3_metrics.csv', index=False)
print("✅ Rôles 5 & 6 (Scénarios + Métriques) — OK")

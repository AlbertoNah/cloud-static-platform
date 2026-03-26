"""
Semaine 3 — Couche réseau & sécurité
Projet : Modélisation cross-layer des facteurs d'instabilité IoT
Modèles : RPL, 6LoWPAN, Congestion, DDoS, Energy Exhaustion
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict

SEED = 42
rng = np.random.default_rng(SEED)
TIME = np.arange(0, 120, 0.5)
NUM_NODES = 7

# ─── Paramètres réseau ────────────────────────────────────
BUFFER_SIZE     = 20        # Taille file d'attente (paquets)
LINK_CAPACITY   = 250e3     # Débit max (250 kbps — Zigbee)
PACKET_SIZE_B   = 127       # Taille paquet 802.15.4 (bytes)
BASE_DELAY      = 0.01      # Délai propagation (s)
MAX_RETX        = 3         # Retransmissions max RPL

# ─── Paramètres sécurité ──────────────────────────────────
DDOS_START      = 40.0      # Début attaque DDoS (s)
DDOS_END        = 80.0      # Fin attaque DDoS (s)
DDOS_RATE       = 10        # Paquets malveillants par seconde
ENERGY_INIT     = 1000.0    # Énergie initiale (mJ)
ENERGY_TX       = 0.5       # Énergie par transmission (mJ)
ENERGY_RX       = 0.3       # Énergie par réception (mJ)
ENERGY_ATTACK   = 2.0       # Énergie drainée par attaque (mJ)

# ══════════════════════════════════════════════════════════
# MODÈLE RPL — Routing Protocol for Low-Power and Lossy Networks
# ══════════════════════════════════════════════════════════
@dataclass
class RPLNode:
    id: int
    rank: int           # Rang dans le DODAG (1 = root)
    distance_m: float   # Distance à la base
    energy_mj: float = ENERGY_INIT
    buffer: List = field(default_factory=list)

    # Historiques
    hist_delay:     List[float] = field(default_factory=list)
    hist_per:       List[float] = field(default_factory=list)
    hist_energy:    List[float] = field(default_factory=list)
    hist_buffer:    List[int]   = field(default_factory=list)
    hist_stability: List[float] = field(default_factory=list)
    hist_throughput:List[float] = field(default_factory=list)

def build_rpl_dodag(distances):
    """Construit un DODAG RPL : rang basé sur la distance."""
    nodes = []
    for i, d in enumerate(distances):
        rank = max(1, int(d / 20))   # Rang = distance / 20m
        nodes.append(RPLNode(id=i, rank=rank, distance_m=d))
    return nodes

def per_from_distance(d):
    """PER simplifié basé sur la distance (modèle S2)."""
    snr_db = 20 - (40 + 30*np.log10(max(d,1))) - (-100)
    snr_lin = 10**(snr_db/10)
    from scipy.special import erfc
    ber = 0.5*erfc(np.sqrt(max(snr_lin,0)))
    return float(np.clip(1-(1-ber)**256, 0, 1))

def simulate_rpl_network(nodes: List[RPLNode], attack_active=False):
    """
    Simule le réseau RPL sur TIME.
    Calcule : délai, PER, énergie, occupation buffer, stabilité, débit.
    """
    distances = [n.distance_m for n in nodes]

    for t_idx, t in enumerate(TIME):
        ddos = attack_active and DDOS_START <= t <= DDOS_END

        for node in nodes:
            per = per_from_distance(node.distance_m)
            per += rng.normal(0, 0.02)           # Bruit
            per = float(np.clip(per, 0, 1))

            # Congestion : buffer se remplit si PER élevé ou DDoS
            incoming = rng.integers(1, 4)
            if ddos:
                incoming += int(rng.integers(DDOS_RATE//2, DDOS_RATE))
            outgoing = max(0, incoming - int(incoming * per))
            buf_level = min(len(node.buffer) + incoming - outgoing, BUFFER_SIZE)
            buf_level = max(0, buf_level)
            node.buffer = [1] * buf_level

            # Délai : base + retransmissions + congestion
            retx = per * MAX_RETX * rng.uniform(0.5, 1.5)
            congestion_delay = (buf_level / BUFFER_SIZE) * 0.05
            delay = BASE_DELAY * (1 + retx) + congestion_delay
            if ddos: delay *= 2.5

            # Énergie
            e_cost = ENERGY_TX * incoming + ENERGY_RX * outgoing
            if ddos: e_cost += ENERGY_ATTACK * rng.uniform(0.5, 1.5)
            node.energy_mj = max(0, node.energy_mj - e_cost * 0.01)

            # Stabilité du lien (fenêtre 20 pas)
            stab = 1.0 - per - (buf_level/BUFFER_SIZE)*0.3
            stab = float(np.clip(stab, 0, 1))

            # Débit effectif (bps)
            throughput = (outgoing * PACKET_SIZE_B * 8) / BASE_DELAY
            throughput = float(np.clip(throughput, 0, LINK_CAPACITY))

            node.hist_delay.append(delay)
            node.hist_per.append(per)
            node.hist_energy.append(node.energy_mj)
            node.hist_buffer.append(buf_level)
            node.hist_stability.append(stab)
            node.hist_throughput.append(throughput)

# ══════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════
DISTANCES = [10, 20, 35, 50, 65, 80, 95]

# Sans attaque
nodes_normal = build_rpl_dodag(DISTANCES)
simulate_rpl_network(nodes_normal, attack_active=False)

# Avec attaque DDoS
nodes_attack = build_rpl_dodag(DISTANCES)
simulate_rpl_network(nodes_attack, attack_active=True)

# ══════════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════════
import os
os.makedirs('/home/claude/IoT-CrossLayer/results/figures', exist_ok=True)
os.makedirs('/home/claude/IoT-CrossLayer/results/raw_data', exist_ok=True)

colors = plt.cm.tab10(np.linspace(0, 1, NUM_NODES))

# ── Graphique 1 : Délai moyen par nœud — normal vs attaque ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for node, c in zip(nodes_normal, colors):
    axes[0].plot(TIME, node.hist_delay, color=c, alpha=0.7, linewidth=1, label=f'N{node.id} ({node.distance_m}m)')
axes[0].set_title('Délai RPL — Sans attaque', fontsize=12)
axes[0].set_xlabel('Temps (s)'); axes[0].set_ylabel('Délai (s)')
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

for node, c in zip(nodes_attack, colors):
    axes[1].plot(TIME, node.hist_delay, color=c, alpha=0.7, linewidth=1, label=f'N{node.id} ({node.distance_m}m)')
axes[1].axvspan(DDOS_START, DDOS_END, alpha=0.15, color='red', label='Attaque DDoS')
axes[1].set_title('Délai RPL — Avec attaque DDoS', fontsize=12)
axes[1].set_xlabel('Temps (s)'); axes[1].set_ylabel('Délai (s)')
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
fig.suptitle('Impact DDoS sur le délai réseau RPL', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s3_delay_ddos.png', dpi=150)
plt.close()

# ── Graphique 2 : Énergie résiduelle (energy exhaustion) ──
fig, ax = plt.subplots(figsize=(12, 5))
for node_n, node_a, c in zip(nodes_normal, nodes_attack, colors):
    ax.plot(TIME, node_n.hist_energy, color=c, linestyle='-',  alpha=0.6, linewidth=1.2)
    ax.plot(TIME, node_a.hist_energy, color=c, linestyle='--', alpha=0.9, linewidth=1.5)
ax.axvspan(DDOS_START, DDOS_END, alpha=0.12, color='red')
ax.axhline(100, color='red', linestyle=':', linewidth=1.5, label='Seuil critique (100 mJ)')
# Légende manuelle
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='gray', linestyle='-',  label='Sans attaque'),
    Line2D([0],[0], color='gray', linestyle='--', label='Avec attaque DDoS'),
    Line2D([0],[0], color='red',  linestyle=':',  label='Seuil critique'),
]
ax.legend(handles=legend_elements, fontsize=10)
ax.set_xlabel('Temps (s)'); ax.set_ylabel('Énergie résiduelle (mJ)')
ax.set_title('Energy Exhaustion — Impact de l\'attaque DDoS sur la batterie des nœuds', fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s3_energy_exhaustion.png', dpi=150)
plt.close()

# ── Graphique 3 : Occupation du buffer (congestion) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for node, c in zip(nodes_normal, colors):
    axes[0].plot(TIME, node.hist_buffer, color=c, alpha=0.7, linewidth=1, label=f'N{node.id}')
axes[0].axhline(BUFFER_SIZE*0.8, color='orange', linestyle='--', label='Seuil congestion (80%)')
axes[0].set_title('Congestion — Sans attaque', fontsize=12)
axes[0].set_ylabel('Paquets dans le buffer'); axes[0].set_xlabel('Temps (s)')
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

for node, c in zip(nodes_attack, colors):
    axes[1].plot(TIME, node.hist_buffer, color=c, alpha=0.7, linewidth=1, label=f'N{node.id}')
axes[1].axhline(BUFFER_SIZE*0.8, color='orange', linestyle='--', label='Seuil congestion')
axes[1].axvspan(DDOS_START, DDOS_END, alpha=0.15, color='red', label='DDoS')
axes[1].set_title('Congestion — Avec attaque DDoS', fontsize=12)
axes[1].set_ylabel('Paquets dans le buffer'); axes[1].set_xlabel('Temps (s)')
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
fig.suptitle('Congestion réseau (occupation buffer) — Impact cross-layer', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s3_congestion.png', dpi=150)
plt.close()

# ── Graphique 4 : Stabilité du lien par nœud ──
fig, ax = plt.subplots(figsize=(12, 5))
for node_n, node_a, c in zip(nodes_normal, nodes_attack, colors):
    ax.plot(TIME, node_n.hist_stability, color=c, linestyle='-',  alpha=0.5, linewidth=1)
    ax.plot(TIME, node_a.hist_stability, color=c, linestyle='--', alpha=0.8, linewidth=1.5)
ax.axvspan(DDOS_START, DDOS_END, alpha=0.1, color='red')
ax.set_xlabel('Temps (s)'); ax.set_ylabel('Stabilité (0=instable, 1=stable)')
ax.set_title('Stabilité des liens — Normal (trait) vs DDoS (pointillé)', fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig('/home/claude/IoT-CrossLayer/results/figures/s3_link_stability.png', dpi=150)
plt.close()

# ── Export CSV ────────────────────────────────────────────
rows = []
for node_n, node_a in zip(nodes_normal, nodes_attack):
    for t_idx, t in enumerate(TIME):
        rows.append({
            'time_s': round(t,2), 'node_id': node_n.id,
            'distance_m': node_n.distance_m, 'rank': node_n.rank,
            'delay_normal': round(node_n.hist_delay[t_idx],5),
            'delay_attack': round(node_a.hist_delay[t_idx],5),
            'per_normal': round(node_n.hist_per[t_idx],4),
            'energy_normal': round(node_n.hist_energy[t_idx],2),
            'energy_attack': round(node_a.hist_energy[t_idx],2),
            'buffer_normal': node_n.hist_buffer[t_idx],
            'buffer_attack': node_a.hist_buffer[t_idx],
            'stability_normal': round(node_n.hist_stability[t_idx],3),
            'stability_attack': round(node_a.hist_stability[t_idx],3),
        })
pd.DataFrame(rows).to_csv('/home/claude/IoT-CrossLayer/results/raw_data/s3_network_data.csv', index=False)

print("✅ Semaine 3 (Réseau + Sécurité) — OK")
print(f"   Graphiques : s3_delay_ddos, s3_energy_exhaustion, s3_congestion, s3_link_stability")

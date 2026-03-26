"""
Rôle 3 — Modélisation de la mobilité (Random Waypoint)
Projet : Modélisation cross-layer des facteurs d'instabilité IoT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from typing import List

ZONE_X, ZONE_Y = 100.0, 100.0
NUM_NODES      = 5
V_MIN, V_MAX   = 1.0, 10.0
PAUSE_MIN, PAUSE_MAX = 0.0, 5.0
SIM_DURATION   = 120.0
DT             = 0.5
SEED           = 42

@dataclass
class Node:
    id: int; x: float; y: float
    vx: float=0.; vy: float=0.
    target_x: float=0.; target_y: float=0.
    pause_remaining: float=0.; speed: float=0.
    history_x: List[float]=field(default_factory=list)
    history_y: List[float]=field(default_factory=list)
    history_t: List[float]=field(default_factory=list)
    history_speed: List[float]=field(default_factory=list)
    history_dist: List[float]=field(default_factory=list)

def pick_waypoint(node, rng):
    node.target_x = rng.uniform(0, ZONE_X)
    node.target_y = rng.uniform(0, ZONE_Y)
    node.speed    = rng.uniform(V_MIN, V_MAX)
    dx = node.target_x - node.x; dy = node.target_y - node.y
    dist = np.sqrt(dx**2+dy**2)
    if dist > 0:
        node.vx = node.speed*dx/dist; node.vy = node.speed*dy/dist

def simulate():
    rng = np.random.default_rng(SEED)
    time_steps = np.arange(0, SIM_DURATION, DT)
    bx, by = ZONE_X/2, ZONE_Y/2
    nodes = [Node(i, rng.uniform(0,ZONE_X), rng.uniform(0,ZONE_Y)) for i in range(NUM_NODES)]
    for n in nodes: pick_waypoint(n, rng)

    for t in time_steps:
        for node in nodes:
            node.history_x.append(node.x); node.history_y.append(node.y)
            node.history_t.append(t)
            node.history_speed.append(node.speed if node.pause_remaining<=0 else 0.)
            node.history_dist.append(np.sqrt((node.x-bx)**2+(node.y-by)**2))
            if node.pause_remaining > 0:
                node.pause_remaining -= DT; node.vx=0.; node.vy=0.
            else:
                dx=node.target_x-node.x; dy=node.target_y-node.y
                dist=np.sqrt(dx**2+dy**2)
                if dist <= node.speed*DT:
                    node.x=node.target_x; node.y=node.target_y
                    node.pause_remaining=rng.uniform(PAUSE_MIN,PAUSE_MAX)
                    pick_waypoint(node, rng)
                else:
                    node.x+=node.vx*DT; node.y+=node.vy*DT
    return nodes, time_steps

def save_figures(nodes):
    colors = plt.cm.tab10(np.linspace(0,1,NUM_NODES))
    # Trajectoires
    fig, ax = plt.subplots(figsize=(8,8))
    for node,c in zip(nodes,colors):
        ax.plot(node.history_x, node.history_y, '-', color=c, alpha=0.6, linewidth=1.2, label=f'Nœud {node.id}')
        ax.scatter(node.history_x[0],node.history_y[0],color=c,marker='o',s=80,zorder=5)
        ax.scatter(node.history_x[-1],node.history_y[-1],color=c,marker='X',s=100,zorder=5)
    ax.scatter(ZONE_X/2,ZONE_Y/2,color='black',marker='*',s=300,zorder=10,label='Station de base')
    ax.set_xlim(0,ZONE_X); ax.set_ylim(0,ZONE_Y)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Trajectoires — Random Waypoint Mobility'); ax.legend(); ax.grid(True,alpha=0.3)
    fig.tight_layout()
    plt.savefig('/home/claude/IoT-CrossLayer/results/figures/trajectoires_rwp.png', dpi=150); plt.close()

    # Distance à la base
    fig, ax = plt.subplots(figsize=(12,5))
    for node,c in zip(nodes,colors):
        ax.plot(node.history_t, node.history_dist, color=c, alpha=0.8, linewidth=1.2, label=f'Nœud {node.id}')
    ax.axhline(70,color='red',linestyle='--',linewidth=1.5,label='Seuil instabilité (70m)')
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Distance à la base (m)')
    ax.set_title('Impact mobilité : distance nœud ↔ station de base')
    ax.legend(); ax.grid(True,alpha=0.3)
    fig.tight_layout()
    plt.savefig('/home/claude/IoT-CrossLayer/results/figures/distance_to_base.png', dpi=150); plt.close()

def save_csv(nodes, time_steps):
    rows=[]
    for node in nodes:
        for t,x,y,s,d in zip(node.history_t,node.history_x,node.history_y,node.history_speed,node.history_dist):
            rows.append({'node_id':node.id,'time_s':round(t,2),'pos_x':round(x,3),
                         'pos_y':round(y,3),'speed_ms':round(s,3),'dist_to_base_m':round(d,3)})
    pd.DataFrame(rows).to_csv('/home/claude/IoT-CrossLayer/results/raw_data/mobility_data.csv', index=False)

nodes, time_steps = simulate()
save_figures(nodes)
save_csv(nodes, time_steps)
print("✅ Rôle 3 (Mobilité) — OK")

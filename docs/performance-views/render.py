#!/usr/bin/env python3
"""Render compact crimson README assets from archived measurements; no GPU work."""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-selfforcing-crimson', 'font.size': 12})


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def hero():
    data = json.loads((ROOT/'history.json').read_text())['rows'][-1]
    values = [data['reference_ms'], data['candidate_ms']]
    assert all(math.isfinite(v) and v > 0 for v in values)
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-SelfForcing, optimized with an AI agent', color=RED,
             weight='bold', fontsize=19)
    fig.text(.06, .78, 'Self-Forcing DMD · 2 H200-class GPUs · 1 video per GPU · BF16 mixed precision',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.24, .24, .39, .45])
    ax.barh([1, 0], values, color=['#D7BDBF', RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Reference', 'Faster-SelfForcing'])
    for y, value in zip([1, 0], values):
        ax.text(value+300, y, f'{value:,.1f} ms', va='center',
                color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 14800)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 2500, 5000, 7500, 10000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    frame(ax, 'x')
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Time per training iteration (ms) ↓', color=MUTED, labelpad=12, fontsize=11)
    fig.text(.85, .54, f"{values[0]/values[1]:.2f}×", color=RED, fontsize=35, weight='bold', ha='center')
    fig.text(.85, .43, 'iteration speedup', color=MUTED, fontsize=12, ha='center')
    save(fig, ROOT/'01-benchmark')


def history():
    rows = json.loads((ROOT/'history.json').read_text())['rows']
    height = 7.7
    top = lambda inches: 1 - inches/height       # figure fraction from inches below the top edge
    fig = plt.figure(figsize=(9, height), facecolor='white')
    fig.text(.09, top(.36), 'Faster-SelfForcing', fontsize=11, weight='bold', color=RED)
    fig.text(.09, top(.79), 'Agent Optimization History', fontsize=21.5, weight='bold', color=INK)
    fig.text(.94, top(.65), f"{rows[-1]['speedup']:.2f}×", fontsize=29, ha='right', weight='bold', color=RED)
    fig.text(.94, top(.88), 'iteration speedup', fontsize=10.5, ha='right', color=MUTED)
    fig.text(.09, top(1.31), 'Cumulative speedup · 2 H200-class GPUs · 1 video per GPU', fontsize=11, color=MUTED)
    ax = fig.add_axes([.09, top(4.19), .85, 2.61/height])
    x = [row['round'] for row in rows]
    values = [row['speedup'] for row in rows]
    ax.plot(x, values, color=RED, linewidth=2.3, zorder=2)
    for xi, value, row in zip(x, values, rows):
        screen = row['check'] != 'full'
        ax.plot([xi], [value], marker='o', markersize=6.5, zorder=3, color=RED,
                markerfacecolor='white' if screen else RED, markeredgewidth=1.8)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.2f}×', (x[i], value), xytext=(0, 11 if i % 2 == 0 else -23),
                    textcoords='offset points', ha='center', fontsize=12,
                    color=RED if i == len(rows)-1 else INK, weight='bold')
    ax.set_xticks(x, [f"R{step:02d}" for step in x])
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2f}×'))
    ax.set_ylim(1.0, 2.3)
    ax.set_xlim(x[0]-.6, x[-1]+.6)
    frame(ax)
    for i, row in enumerate(rows):
        bx, by = .09+(i % 2)*.44, top(5.18+(i // 2)*.74)
        fig.add_artist(FancyBboxPatch((bx, by-.378/height), .41, .585/height,
                       boxstyle='round,pad=0.004,rounding_size=0.007',
                       transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(bx+.014, by-.08/height, f"R{row['round']:02d}", fontsize=12, weight='bold', color=RED)
        label = textwrap.fill(row['label'], width=24, break_long_words=False, break_on_hyphens=False)
        fig.text(bx+.079, by-.08/height, label, fontsize=11.5, color=INK,
                 weight='bold', va='center', linespacing=1.35)
    fig.text(.09, top(7.45), 'Open marker: timing screen + quick check. Filled: full check (all four gates). '
             'Cumulative configurations,\nnot isolated ablations; R06 is within run-to-run noise of R05.',
             fontsize=9.5, color=MUTED, linespacing=1.4)
    save(fig, ROOT/'02-history')


def modulemap():
    fig = plt.figure(figsize=(9, 6.6), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .92, 'What changed in Faster-SelfForcing', fontsize=23, color=INK, weight='bold')
    cards = [
        ('Attention', 'Vendored FlashAttention-3 for Hopper\nOn by default; SF_ATTN=fa2 restores FA2',
         'wan/modules/attention.py\nOfficial FA3 package (Hopper)', '1 file + dependency'),
        ('Rollout & FSDP', 'Generator gathered once per rollout\nCached cross-attn K/V · CUDA graphs',
         'pipeline/self_forcing_training.py\nutils/fsdp_hold.py · utils/cuda_graph.py\nwan/modules/causal_model.py',
         '4 files'),
        ('Bit-exact fused kernels', 'RMSNorm · LayerNorm + modulate\nQK-norm + RoPE · gated residual',
         'wan/modules/fused_ew.py · fused_ln.py\nfused_qk.py · fused_rope.py\ncompiled_ops.py · model.py',
         '6 files'),
        ('Memory & overhead', 'Selective checkpointing · BF16 frozen models\nResident text encoder · async EMA',
         'wan/modules/sac.py\ntrainer/distillation.py\nutils/distributed.py', '3 files'),
    ]
    for i, (title, description, paths, count) in enumerate(cards):
        x, y = .055+(i % 2)*.455, .49-(i // 2)*.39
        ax.add_patch(FancyBboxPatch((x, y), .425, .34,
                     boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x+.425], [y+.34, y+.34], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x+.014, y+.30, f'{i+1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x+.41, y+.30, count, fontsize=10, color=MUTED, ha='right')
        ax.text(x+.014, y+.254, title, fontsize=13, color=INK, weight='bold')
        ax.text(x+.014, y+.198, description, fontsize=10.3, color=INK, va='top', linespacing=1.6)
        ax.text(x+.014, y+.035, paths, fontsize=9.5, color=MUTED, linespacing=1.5)
    save(fig, ROOT.parent/'change-views/module-map')


if __name__ == '__main__':
    hero()
    history()
    modulemap()

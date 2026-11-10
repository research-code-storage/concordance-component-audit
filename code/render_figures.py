"""Redraw all paper figures from saved results; no fitting or simulation is run.

Uses Times New Roman when installed, with a serif fallback.
Run: python code/render_figures.py
"""
from pathlib import Path
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parent / 'generated/.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

DATA = HERE.parent / 'generated/plot_data'
OUT = HERE.parent / 'generated/figures'
OUT.mkdir(parents=True,exist_ok=True)
for name in ('times.ttf', 'timesi.ttf', 'timesbd.ttf', 'timesbi.ttf'):
    path = Path('C:/Windows/Fonts') / name
    if path.exists():
        fm.fontManager.addfont(str(path))
families={f.name for f in fm.fontManager.ttflist}
family=next((n for n in ['Times New Roman','Liberation Serif','DejaVu Serif'] if n in families))
font=fm.findfont(family)
plt.rcParams.update({
    'font.family': family, 'font.size': 10,
    'mathtext.fontset': 'custom', 'mathtext.rm': family,
    'mathtext.it': family+':italic', 'mathtext.bf': family+':bold',
    'mathtext.fallback': None, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'axes.edgecolor': '#333333',
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.fontsize': 8.5, 'legend.frameon': False,
    'savefig.facecolor': 'white',
})

def rows(name):
    with (DATA / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))

def base(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e3e3e3', linewidth=.45)

report = {'font': fm.FontProperties(fname=font).get_name(),
          'scientific_experiments_rerun': False, 'input_hashes': {}, 'figures': {}}
for name, info in load('../plot_input_manifest.json').items():
    digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    assert digest == info['sha256']
    report['input_hashes'][name] = digest

def save(fig, name, info):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check that all visible figure text lies within the exported canvas.
    w, h = fig.bbox.width, fig.bbox.height
    text_boxes = []
    for obj in fig.findobj(matplotlib.text.Text):
        if obj.get_visible() and obj.get_text().strip():
            box = obj.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= w+1 and box.y1 <= h+1, (name, obj.get_text(), box.bounds, (w, h))
                text_boxes.append(obj.get_text())
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.png'), dpi=180)
    info['canvas_inches'] = list(fig.get_size_inches())
    info['text_within_canvas'] = True
    info['texts'] = text_boxes
    report['figures'][name] = info
    plt.close(fig)

# Main Figure 1.
cells = load('fitted_component_audit.json')['cells']
assert len(cells) == 12 and all(r['R'] == 200 for r in cells)
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.15), sharex=True)
fig.subplots_adjust(left=.105, right=.985, bottom=.12, top=.945, wspace=.39, hspace=.32)
ax = axes.ravel()
series = []
for inf, colour, marker in [(True, '#155f80', 'o'), (False, '#ba5324', 's')]:
    for initial, style in [(0., '--'), (1.3132616877555847, '-')]:
        group = sorted([r for r in cells if r['informative'] == inf and abs(r['initial']-initial)<1e-9], key=lambda r:r['shift'])
        assert [r['shift'] for r in group] == [0, 6, 16]
        label = ('Informative' if inf else 'Noise') + (', start 0' if initial == 0 else ', start 1.313')
        for a, key in zip(ax, ['auc', 'auc_difference', 'A', 'B']):
            x = [r['shift'] for r in group]
            y = [r[key+'_mean'] for r in group]
            error = [2*r[key+'_mcse'] for r in group]
            a.errorbar(x, y, yerr=error, color=colour, linestyle=style, marker=marker,
                       markersize=3.6, linewidth=1.3, elinewidth=.8, capsize=2.5, capthick=.8, label=label)
            series.append({'metric': key, 'label': label, 'x': x, 'y': y, 'error': error})
for a, label, letter in zip(ax, ['Mean AUC', r'Mean $\Delta$', r'Mean $A_\omega$', r'Mean $B_\omega$'], 'ABCD'):
    base(a)
    a.set_ylabel(label, labelpad=5)
    a.set_xticks([0, 6, 16])
    a.set_xlim(-.75, 16.75)
    a.text(0, 1.045, letter, transform=a.transAxes, weight='bold', size=12, va='bottom')
ax[0].set_ylim(.48, .97)
ax[0].set_yticks([.5, .6, .7, .8, .9])
ax[1].set_ylim(-.008, .24)
ax[1].set_yticks([0, .05, .10, .15, .20])
ax[2].set_ylim(-.01, .33)
ax[2].set_yticks([0, .1, .2, .3])
ax[3].set_ylim(-.02, .62)
ax[3].set_yticks([0, .2, .4, .6])
ax[0].axhline(.5, color='#888888', linewidth=.65, linestyle=':')
ax[1].axhline(0, color='#888888', linewidth=.65, linestyle=':')
legend = ax[3].legend(loc='upper right', borderaxespad=.25, labelspacing=.35, handlelength=2.1, handletextpad=.55)
fig.supxlabel('Embedding displacement', y=.025, fontsize=10.5)
save(fig, 'Figure_1', {'printed_figure': 'Figure 1', 'series': series, 'legend': 'upper right of panel D'})

# Main Figure 2.
results = load('results.json')
auc_rows = {f"{r['training_compounds']}|{r['domain']}|{r['arm']}": r for r in rows('all_results.csv')}
budget_rows = rows('figure2BC_budget_points.csv')
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.85))
fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.76, wspace=.43)
readouts = [('softplus_original', 'Full predictor', '#1f4e79'),
            ('fixed_sum', 'Fixed sum', '#2e8b57'),
            ('kernel_original', 'Kernel only', '#c0504d')]
bar_records = []
for k, (key, label, colour) in enumerate(readouts):
    means, lo, hi = [], [], []
    for domain in ('PHH', 'external'):
        identity = f'60|{domain}|matched_centred__{key}'
        mean = results[identity]['auc_mean']
        source = auc_rows[identity]
        ci = json.loads(source['compound_bootstrap_ci'])
        assert abs(mean - float(source['auc_mean'])) < 1e-12
        means.append(mean); lo.append(mean-ci[0]); hi.append(ci[1]-mean)
        bar_records.append({'domain': domain, 'readout': key, 'mean': mean, 'ci': ci})
    xx = np.arange(2)+(k-1)*.24
    ax[0].bar(xx, means, width=.24, label=label, color=colour, edgecolor='white', linewidth=.4)
    ax[0].errorbar(xx, means, yerr=[lo, hi], fmt='none', ecolor='#222222', elinewidth=.75, capsize=2)
ax[0].set_xticks([0, 1], ['PHH\n11 compounds', 'DILImap\n49 compounds'])
ax[0].tick_params(axis='x', labelsize=8.7)
ax[0].set_ylabel('Mean AUC', labelpad=4)
ax[0].set_ylim(0, 1.04)
ax[0].set_yticks([0, .25, .5, .75, 1])
ax[0].axhline(.5, color='#888888', linewidth=.65, linestyle=':')
ax[0].set_title('A', loc='left', weight='bold', pad=6)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.085, .995), frameon=False,
           borderaxespad=0, labelspacing=.3, handlelength=1.6, handletextpad=.55)
certificate_records = []
for a, (domain, title) in zip(ax[1:], [('PHH', 'B  PHH'), ('external', 'C  DILImap')]):
    record = results[f'60|{domain}|matched_centred__softplus_original']
    ts = sorted(float(t) for t in record['budget_curve'])
    assert ts == [0, .25, .5, 1, 2, 4, 8]
    per = record['budget_curve_per_draw']
    for t in ts:
        vals = per[f'{t:g}']
        verified = sorted([r for r in budget_rows if r['domain'] == domain and abs(float(r['t'])-t)<1e-12], key=lambda r:int(r['draw']))
        assert len(verified) == 10
        np.testing.assert_allclose(vals, [float(r['tie_aware_bound']) for r in verified], atol=1e-12, rtol=0)
        a.plot([t]*10, vals, 'o', color='#91b1d3', markersize=3, alpha=.75, markeredgewidth=0)
    maxima = [record['budget_curve'][f'{t:g}'] for t in ts]
    np.testing.assert_allclose(maxima, [max(per[f'{t:g}']) for t in ts], atol=0, rtol=0)
    a.plot(ts, maxima, 'o', color='#1f4e79', markersize=4, markeredgewidth=0)
    actual = max(abs(v) for v in record['actual_change_per_draw'])
    a.plot(1, actual, 's', color='#d67732', markersize=5, markeredgewidth=0)
    a.set_xticks([0, 1, 2, 4, 8])
    a.set_xlim(-.25, 8.3)
    a.set_xlabel(r'Multiplier $t$', labelpad=4)
    a.set_ylabel(r'$B_\omega(t)$', labelpad=3)
    a.set_title(title, loc='left', fontweight='bold', pad=6)
    certificate_records.append({'domain': domain, 't': ts, 'per_draw': per, 'maximum': maxima, 'absolute_net_change_at_one': actual})
ax[1].set_ylim(-.008, .315)
ax[1].set_yticks([0, .1, .2, .3])
ax[2].set_ylim(-.001, .040)
ax[2].set_yticks([0, .01, .02, .03, .04])
cert_handles = [Line2D([], [], marker='o', linestyle='none', color='#91b1d3', markersize=4),
                Line2D([], [], marker='o', linestyle='none', color='#1f4e79', markersize=4),
                Line2D([], [], marker='s', linestyle='none', color='#d67732', markersize=5)]
fig.legend(cert_handles, ['Per-draw bound', 'Maximum across draws', r'Maximum $|\Delta|$, $t=1$'],
           loc='upper right', bbox_to_anchor=(.985, .995), frameon=False, borderaxespad=0,
           labelspacing=.3, handlelength=1.6, handletextpad=.55)
for a in ax:
    base(a)
save(fig, 'Figure_2', {'printed_figure': 'Figure 2', 'bars': bar_records,
                      'certificate_points': certificate_records, 'legends': 'above axes; shared for panels B and C'})

# Supporting Figure S1.
curves = rows('figureS1A_analytic_curves.csv')
mc = rows('stable_dilution_60_cells.csv')
counter = rows('figureS1B_counterexamples.csv')
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0))
fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.90, wspace=.29)
colours = ['#1f4e79', '#2e8b57', '#c07000', '#7b3f9d']
curve_records = []
for alpha, colour in zip([.7, 1., 1.5, 2.], colours):
    for mu, style in [(.8, '-'), (-.8, '--')]:
        group = sorted([r for r in curves if float(r['alpha']) == alpha and float(r['mu']) == mu], key=lambda r:float(r['tau']))
        points = sorted([r for r in mc if float(r['alpha']) == alpha and float(r['mu']) == mu], key=lambda r:float(r['tau']))
        assert len(group) == 121 and len(points) == 5
        x = [float(r['tau']) for r in group]; y = [float(r['auc']) for r in group]
        px = [float(r['tau']) for r in points]; py = [float(r['simulated_auc']) for r in points]
        ax[0].plot(x, y, linestyle=style, color=colour, linewidth=1.25, label=fr'$\alpha={alpha:g}$' if mu>0 else None)
        ax[0].plot(px, py, 'o', color=colour, markersize=3, markeredgecolor='#222222', markeredgewidth=.3)
        curve_records.append({'alpha': alpha, 'mu': mu, 'x': x, 'y': y, 'mc_x': px, 'mc_y': py})
ax[0].set_xlabel(r'Noise scale $\tau$')
ax[0].set_ylabel('AUC')
ax[0].set_xticks([0, 1, 2, 3, 4])
ax[0].set_ylim(.265, .75)
ax[0].set_yticks([.3, .4, .5, .6, .7])
ax[0].set_title('A  Symmetric-stable noise', loc='left', weight='bold', pad=9)
ax[0].legend(loc='upper right', ncol=2, columnspacing=.9, handlelength=1.6, handletextpad=.4,
             borderaxespad=.2, labelspacing=.3)
for row, colour, label in zip(counter, ['#c0504d', '#1f4e79'], ['Away from 0.5', 'Across 0.5']):
    ax[1].plot([0, 1], [float(row['baseline_auc']), float(row['noisy_auc'])], 'o-',
               color=colour, linewidth=1.4, markersize=4.5, label=label)
ax[1].set_xticks([0, 1], ['Baseline', 'After noise'])
ax[1].set_xlim(-.22, 1.22)
ax[1].set_ylabel('AUC')
ax[1].set_ylim(.44, .73)
ax[1].set_yticks([.45, .50, .55, .60, .65, .70])
ax[1].set_title('B  Two-point noise', loc='left', weight='bold', pad=9)
ax[1].legend(loc='upper left', borderaxespad=.25, handlelength=1.8, labelspacing=.4)
for a in ax:
    a.axhline(.5, color='#888888', linewidth=.65, linestyle=':')
    base(a)
save(fig, 'Figure_S1', {'printed_figure': 'Figure S1', 'analytic_and_MC_curves': curve_records,
                      'counterexamples': counter, 'legends': 'upper corners with no data obscured'})
(HERE.parent / 'generated/plot_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(f'Generated Figures 1, 2 and S1 using {family}.')

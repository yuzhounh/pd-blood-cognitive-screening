"""Export the editable study workflow and empirical ROCs from frozen predictions."""
from pathlib import Path
from display_labels import MODEL_LABELS, PREDICTOR_SET_LABELS
import json
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from common_preprocessing import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'manuscript/figures'
OUT = ROOT / 'manuscript/results/bayes100'
PRIVATE = ROOT / 'data/derived/bayes100'
NAMES = ['Age_LR', 'Basic_LR', 'LR', 'SVM', 'RF', 'XGBoost', 'CatBoost', 'NN']
LABELS = [MODEL_LABELS[name] for name in NAMES]
COLORS = ['#596775', '#285f8f', '#c7652e', '#2a9d8f', '#80649e', '#9c6644', '#b34f79', '#809c32']
STYLES = ['--', '-.', '-', '-', ':', '--', '-.', ':']


def workflow_figure():
    """The SVG is the maintained source; do not replace it with generated artwork."""
    executable = shutil.which('inkscape')
    if executable is None:
        raise RuntimeError('Inkscape is required to export the maintained workflow SVG.')
    source = FIG / 'figure1_study_design.svg'
    subprocess.run([executable, str(source), '--export-area-page', '--export-text-to-path',
                    '--export-filename=' + str(FIG / 'figure1_study_design.pdf')],
                   check=True, capture_output=True)
    subprocess.run([executable, str(source), '--export-area-page', '--export-dpi=300',
                    '--export-filename=' + str(FIG / 'figure1_study_design.png')],
                   check=True, capture_output=True)


def roc_figure(reuse_coordinates=False):
    font_manager.findfont(font_manager.FontProperties(family='Arial'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10.5, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.unicode_minus': True,
                         'axes.spines.top': False, 'axes.spines.right': False})
    performance = pd.read_csv(OUT / 'all_model_performance.csv')
    retained = pd.read_csv(OUT / 'roc_curves.csv') if reuse_coordinates else None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.3))
    coordinates = []
    observed = []
    for ax, (cohort, ykey, pkey, idkey, title, expected_n) in zip(axes, [
        ('PPMI_internal', 'yi', 'pi', 'internal_id', 'A. Internal PPMI (n=416)', 416),
        ('FAHHAUST_external', 'ye', 'p', 'external_id', 'B. External LY-PD (n=197)', 197),
    ]):
        reference_ids = reference_y = None
        legend_rows = []
        for index, (name, label, color, style) in enumerate(zip(NAMES, LABELS, COLORS, STYLES)):
            if reuse_coordinates:
                points = retained.loc[retained.cohort.eq(cohort) & retained.model.eq(name)]
                fpr, tpr = points.fpr.to_numpy(), points.tpr.to_numpy()
            else:
                with np.load(PRIVATE / (name + '.npz'), allow_pickle=False) as data:
                    y, probability, identifiers = data[ykey], data[pkey], data[idkey]
                assert len(y) == len(probability) == len(identifiers) == expected_n
                assert np.isfinite(probability).all() and np.isin(y, [0, 1]).all()
                if reference_ids is None:
                    reference_ids, reference_y = identifiers, y
                else:
                    assert np.array_equal(identifiers, reference_ids) and np.array_equal(y, reference_y)
                fpr, tpr, _ = roc_curve(y, probability)
            record = performance.loc[performance.cohort.eq(cohort) & performance.model.eq(name)].iloc[0]
            area = auc(fpr, tpr)
            assert np.isclose(area, record.AUROC, rtol=0, atol=1e-12), (cohort, name, area)
            ax.plot(fpr, tpr, color=color, linestyle=style,
                    linewidth=2.05 if index < 3 else 1.15,
                    alpha=1.0 if index < 3 else 0.92,
                    label=label, zorder=4 if index < 3 else 2)
            legend_rows.append((label, area, color, style, index < 3))
            coordinates.extend(dict(cohort=cohort, model=name, fpr=float(x), tpr=float(z))
                               for x, z in zip(fpr, tpr))
            observed.append(dict(cohort=cohort, model=name, n=expected_n, AUROC=float(area),
                                 agrees_with_retained_result=True))
        ax.plot([0, 1], [0, 1], color='#aeb6bc', linestyle='--', linewidth=.8, zorder=0)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='1 − specificity', ylabel='Sensitivity')
        ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(title, loc='left', fontsize=11.8, fontweight='bold', pad=11)
        ax.set_box_aspect(4/5)
        ax.grid(color='#e6eaed', linewidth=.6, zorder=0)
        # Separate text columns align names and numerical AUROCs without padded spaces.
        for col in range(2):
            base = .55 * col
            ax.text(base + .055, -.22, 'Strategy', transform=ax.transAxes,
                    fontsize=9.6, weight='bold', ha='left', va='center')
            ax.text(base + .44, -.22, 'AUROC', transform=ax.transAxes,
                    fontsize=9.6, weight='bold', ha='right', va='center')
            for row, (label, area, color, style, emphasized) in enumerate(legend_rows[col*4:(col+1)*4]):
                yy = -.305 - .083 * row
                ax.plot([base-.025, base+.035], [yy, yy], transform=ax.transAxes,
                        clip_on=False, color=color, linestyle=style,
                        linewidth=2.05 if emphasized else 1.15)
                ax.text(base+.055, yy, label, transform=ax.transAxes,
                        fontsize=10.0, ha='left', va='center')
                ax.text(base+.44, yy, f'{area:.3f}', transform=ax.transAxes,
                        fontsize=10.0, ha='right', va='center')
    fig.subplots_adjust(left=.075, right=.985, top=.90, bottom=.34, wspace=.25)
    fig.canvas.draw()
    assert all(np.isclose(ax.get_window_extent().height/ax.get_window_extent().width,4/5)
               for ax in axes)
    for extension in ['pdf', 'png']:
        fig.savefig(FIG / ('figure3_roc.' + extension), dpi=600,
                    bbox_inches='tight', pad_inches=.09)
    plt.close(fig)
    if not reuse_coordinates:
        pd.DataFrame(coordinates).to_csv(OUT / 'roc_curves.csv', index=False)
    return observed


def main(workflow_only=False, roc_only=False):
    if workflow_only or roc_only:
        # A layout revision must not redraw or recompute the retained ROC results.
        path = OUT / 'presentation_figure_manifest.json'
        retained = json.loads(path.read_text())
        workflow_source = 'manuscript/figures/figure1_study_design.svg'
        for rel, expected in retained['input_sha256'].items():
            if roc_only or rel != workflow_source:
                assert sha(ROOT / rel) == expected, rel
        for rel, expected in retained['output_sha256'].items():
            assert sha(ROOT / rel) == expected, rel
        stem = 'figure3_roc' if roc_only else 'figure1_study_design'
        if roc_only:
            retained['AUROC_checks'] = roc_figure(reuse_coordinates=True)
        else:
            workflow_figure()
            retained['input_sha256'][workflow_source] = sha(ROOT / workflow_source)
        changed = {f'manuscript/figures/{stem}.{ext}':
                   sha(FIG / f'{stem}.{ext}') for ext in ['pdf', 'png']}
        retained['output_sha256'].update(changed)
        retained['code_sha256'] = sha(__file__)
        retained['latest_revision'] = ('ROC legend labels/alignment only; retained coordinates reused'
                                      if roc_only else 'Workflow layout only; retained ROC outputs and checks reused')
        write_json(path, retained)
        legacy_path = ROOT / 'manuscript/results/manuscript_statistics/figure_manifest.json'
        legacy = json.loads(legacy_path.read_text())
        legacy['output_sha256'].update(changed)
        write_json(legacy_path, legacy)
        print(json.dumps(dict(status='passed', main_figures_generated=[3 if roc_only else 1],
                              retained_ROC_evidence_reused=True, model_refits=0)))
        return
    sources = [FIG / 'figure1_study_design.svg', OUT / 'all_model_performance.csv']
    sources += [PRIVATE / (name + '.npz') for name in NAMES]
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    workflow_figure()
    checks = roc_figure()
    assert all(sha(ROOT / rel) == expected for rel, expected in inputs.items())
    outputs = {p.relative_to(ROOT).as_posix(): sha(p) for stem in
               ['figure1_study_design', 'figure3_roc'] for extension in ['pdf', 'png']
               for p in [FIG / (stem + '.' + extension)]}
    write_json(OUT / 'presentation_figure_manifest.json', dict(
        status='generated_and_checked', input_sha256=inputs, output_sha256=outputs,
        code_sha256=sha(__file__), font_family='Arial', empirical_ROC_curves=16,
        model_refits=0, new_bootstrap_samples=0, threshold_updates=0,
        AUROC_checks=checks, SVG_is_workflow_source=True))
    # Preserve the original figure-2 evidence while recording the replaced workflow.
    path = ROOT / 'manuscript/results/manuscript_statistics/figure_manifest.json'
    legacy = json.loads(path.read_text())
    old = {rel: legacy['output_sha256'][rel] for rel in outputs if rel in legacy['output_sha256']}
    legacy.setdefault('previous_workflow_output_sha256', old)
    legacy['output_sha256'].update(outputs)
    legacy['presentation_revision_manifest'] = 'manuscript/results/bayes100/presentation_figure_manifest.json'
    write_json(path, legacy)
    print(json.dumps(dict(status='passed', main_figures_generated=[1, 3], ROC_curves=16,
                          model_refits=0, all_AUROCs_match=True)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--workflow-only', action='store_true',
                        help='Export only the maintained SVG and retain existing ROC evidence.')
    group.add_argument('--roc-only', action='store_true',
                       help='Redraw only ROC presentation using retained coordinates.')
    args = parser.parse_args()
    main(workflow_only=args.workflow_only, roc_only=args.roc_only)

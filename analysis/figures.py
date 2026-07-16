from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

from analysis.load_data import df, metrics_df, sequence_quality_score, sequence_lookup

metrics_df['sequence_quality'] = df.apply(
    lambda row: sequence_quality_score(
        row['tool_sequence_parsed'], sequence_lookup.get((row['dataset'], row['question_no']), [])
    ),
    axis=1
)

LEVEL_ORDER = ['minimal', 'current', 'enhanced']
LEVEL_LABELS = {'minimal': 'Minimal', 'current': 'Current', 'enhanced': 'Structured'}
LEVEL_COLORS = {'minimal': '#B8C9D9', 'current': '#5C1A2E', 'enhanced': '#F5E6C8'}
BACKGROUND = '#FDFAF4'


def mean_and_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(len(values))
    ci95 = 1.96 * sem
    return mean, ci95


def plot_bar_with_ci(ax, data, value_col, title, ylabel):
    means = []
    cis = []
    colors = []
    labels = []
    for level in LEVEL_ORDER:
        values = data.loc[data['description_level'] == level, value_col]
        mean, ci95 = mean_and_ci95(values)
        means.append(mean)
        cis.append(ci95)
        colors.append(LEVEL_COLORS[level])
        labels.append(LEVEL_LABELS[level])

    x = np.arange(len(LEVEL_ORDER))
    ax.bar(
        x, means, yerr=cis, capsize=6, color=colors,
        edgecolor='#5C1A2E', linewidth=1, error_kw={'ecolor': '#2C2C2A', 'elinewidth': 1.2}
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, color='#2C2C2A', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, color='#2C2C2A')
    ax.set_ylim(0, 1)
    ax.set_facecolor(BACKGROUND)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#2C2C2A')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BACKGROUND)

plot_bar_with_ci(ax1, metrics_df, 'f1', 'F1 Score by Description Level', 'F1 Score')
plot_bar_with_ci(ax2, metrics_df, 'sequence_quality', 'Sequence Quality by Description Level', 'Sequence Quality')

plt.tight_layout()
plt.savefig('figures/figure1_main_results.png', dpi=300, facecolor=BACKGROUND)
plt.close()

print('Figure 1 saved to figures/figure1_main_results.png')

top_tools = df['first_tool'].value_counts().head(7).index.tolist()

counts = (
    df[df['first_tool'].isin(top_tools)]
    .groupby(['first_tool', 'description_level'])
    .size()
    .unstack(fill_value=0)
    .reindex(index=top_tools, columns=LEVEL_ORDER, fill_value=0)
)

fig2, ax = plt.subplots(figsize=(11, 6))
fig2.patch.set_facecolor(BACKGROUND)

n_levels = len(LEVEL_ORDER)
bar_width = 0.8 / n_levels
x = np.arange(len(top_tools))

for i, level in enumerate(LEVEL_ORDER):
    offset = (i - (n_levels - 1) / 2) * bar_width
    ax.bar(
        x + offset, counts[level], width=bar_width, color=LEVEL_COLORS[level],
        edgecolor='#5C1A2E', linewidth=1, label=LEVEL_LABELS[level]
    )

ax.set_xticks(x)
ax.set_xticklabels(top_tools, rotation=30, ha='right')
ax.set_title('First Tool Called by Description Level', color='#2C2C2A', fontsize=13, fontweight='bold')
ax.set_ylabel('Count', color='#2C2C2A')
ax.set_facecolor(BACKGROUND)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors='#2C2C2A')
ax.legend(frameon=False, labelcolor='#2C2C2A')

plt.tight_layout()
plt.savefig('figures/figure2_first_tool.png', dpi=300, facecolor=BACKGROUND)
plt.close()

print('Figure 2 saved to figures/figure2_first_tool.png')

MATCHED_COLOR = '#8FB89A'
MISSED_COLOR = '#C97B76'
INK = '#2C2C2A'
TEXT_ON_DARK = '#F5E6C8'

reference_tools = [
    'get_missing_values', 'get_dtypes', 'get_descriptive_stats', 'get_preprocessing_recommendations'
]
agent_tools = ['get_missing_values', 'get_dtypes']
matched_tools = set(agent_tools) & set(reference_tools)

precision = len(matched_tools) / len(agent_tools)
recall = len(matched_tools) / len(reference_tools)
f1_example = 2 * precision * recall / (precision + recall)


def draw_tool_row(ax, tools, y, box_w, box_h, gap, colors):
    n = len(tools)
    total_w = n * box_w + (n - 1) * gap
    start_x = 5 - total_w / 2
    for i, tool in enumerate(tools):
        x = start_x + i * (box_w + gap)
        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5, edgecolor='#5C1A2E', facecolor=colors[i]
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2, y + box_h / 2, tool, ha='center', va='center',
            fontsize=8.5, color=INK, wrap=True
        )


fig3, ax = plt.subplots(figsize=(11, 8))
fig3.patch.set_facecolor(BACKGROUND)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor(BACKGROUND)

ax.set_title(
    'How Correctness is Scored: A Worked Example', color=INK, fontsize=15, fontweight='bold', pad=10
)

ax.text(5, 9.1, 'Reference Set (Q2 - Iris)', ha='center', va='center', fontsize=11.5, color=INK, fontweight='bold')
ref_colors = [MATCHED_COLOR if t in matched_tools else MISSED_COLOR for t in reference_tools]
draw_tool_row(ax, reference_tools, y=7.6, box_w=2.1, box_h=1.0, gap=0.25, colors=ref_colors)

ax.text(5, 6.4, 'Agent Called', ha='center', va='center', fontsize=11.5, color=INK, fontweight='bold')
agent_colors = [MATCHED_COLOR for _ in agent_tools]
draw_tool_row(ax, agent_tools, y=4.9, box_w=2.1, box_h=1.0, gap=0.25, colors=agent_colors)

legend_y = 3.9
ax.add_patch(FancyBboxPatch(
    (2.6, legend_y), 0.4, 0.35, boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=1, edgecolor='#5C1A2E', facecolor=MATCHED_COLOR
))
ax.text(3.15, legend_y + 0.17, 'Matched', ha='left', va='center', fontsize=9.5, color=INK)
ax.add_patch(FancyBboxPatch(
    (5.0, legend_y), 0.4, 0.35, boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=1, edgecolor='#5C1A2E', facecolor=MISSED_COLOR
))
ax.text(5.55, legend_y + 0.17, 'Missed', ha='left', va='center', fontsize=9.5, color=INK)

calc_box = FancyBboxPatch(
    (1.5, 0.6), 7.0, 2.5, boxstyle="round,pad=0.05,rounding_size=0.15",
    linewidth=1.5, edgecolor='#5C1A2E', facecolor='#5C1A2E'
)
ax.add_patch(calc_box)
ax.text(
    5, 2.55,
    f'Precision = {len(matched_tools)}/{len(agent_tools)} = {precision:.3f}',
    ha='center', va='center', fontsize=11.5, color=TEXT_ON_DARK
)
ax.text(
    5, 1.85,
    f'Recall = {len(matched_tools)}/{len(reference_tools)} = {recall:.3f}',
    ha='center', va='center', fontsize=11.5, color=TEXT_ON_DARK
)
ax.text(
    5, 1.15,
    f'F1 = 2 x (Precision x Recall) / (Precision + Recall) = {f1_example:.3f}',
    ha='center', va='center', fontsize=11.5, color=TEXT_ON_DARK, fontweight='bold'
)

plt.tight_layout()
plt.savefig('figures/figure3_scoring_example.png', dpi=300, facecolor=BACKGROUND)
plt.close()

print('Figure 3 saved to figures/figure3_scoring_example.png')

RED_TITLE = '#B5433E'
GREEN_TITLE = '#4E7A5A'
FAIL_COLOR = '#C97B76'
STUCK_COLOR = '#8B2E2E'
OK_COLOR = '#8FB89A'


def draw_flow_box(ax, cx, y_top, box_w, box_h, label, facecolor, textcolor=INK, bold=False):
    box = FancyBboxPatch(
        (cx - box_w / 2, y_top - box_h), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor='#5C1A2E', facecolor=facecolor
    )
    ax.add_patch(box)
    ax.text(
        cx, y_top - box_h / 2, label, ha='center', va='center',
        fontsize=9.5, color=textcolor, fontweight='bold' if bold else 'normal'
    )
    return y_top - box_h


def draw_vertical_flow(ax, cx, items, colors, box_w=3.4, box_h=0.78, gap=0.3, top_y=8.6, bold_last=False):
    y = top_y
    centers = []
    for i, (label, color) in enumerate(zip(items, colors)):
        y_bottom = draw_flow_box(
            ax, cx, y, box_w, box_h, label, color,
            bold=(bold_last and i == len(items) - 1)
        )
        centers.append((y, y_bottom))
        if i < len(items) - 1:
            ax.annotate(
                '', xy=(cx, y_bottom - gap + 0.03), xytext=(cx, y_bottom - 0.03),
                arrowprops=dict(arrowstyle='-|>', color='#5C1A2E', lw=1.6)
            )
        y = y_bottom - gap
    return centers


def has_repetition(tools):
    return len(tools) != len(set(tools))


q3_minimal = df[(df['description_level'] == 'minimal') & (df['question_no'] == 3)]
minimal_repetition_cases = q3_minimal[q3_minimal['analysis_tools'].apply(has_repetition)]

minimal_example = None
structured_example = None
for _, candidate_row in minimal_repetition_cases.iterrows():
    structured_candidates = df[
        (df['description_level'] == 'enhanced') &
        (df['question_no'] == 3) &
        (df['dataset'] == candidate_row['dataset']) &
        (~df['status'].astype(str).str.startswith('FAILED'))
    ]
    if len(structured_candidates) > 0:
        minimal_example = candidate_row
        structured_example = structured_candidates.iloc[0]
        break

print(f"Minimal repetition example: {minimal_example['run_id']}")
print(f"Tool sequence: {minimal_example['analysis_tools']}")

print(f"Structured non-fail example: {structured_example['run_id']}")
print(f"Tool sequence: {structured_example['analysis_tools']}")

minimal_tools = list(minimal_example['analysis_tools'])
structured_tools = list(structured_example['analysis_tools'])

repeated_tool = next(t for t in minimal_tools if minimal_tools.count(t) > 1)
first_rep_idx = minimal_tools.index(repeated_tool)
last_rep_idx = len(minimal_tools) - 1 - minimal_tools[::-1].index(repeated_tool)

left_items = minimal_tools + ['STUCK']
left_colors = [
    STUCK_COLOR if i == last_rep_idx else (FAIL_COLOR if i == first_rep_idx else LEVEL_COLORS['minimal'])
    for i in range(len(minimal_tools))
] + [STUCK_COLOR]

right_items = structured_tools + ['Done ✓']
right_colors = [OK_COLOR for _ in structured_tools] + [OK_COLOR]

fig4, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 8.5))
fig4.patch.set_facecolor(BACKGROUND)

fig4.suptitle(
    'Tool Repetition Loop vs Forward Sequencing (Q3 — Correlation with Target)',
    color=INK, fontsize=14, fontweight='bold'
)

for ax in (ax_left, ax_right):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BACKGROUND)

ax_left.text(5, 9.6, 'Minimal Description', ha='center', va='center', fontsize=13, color=RED_TITLE, fontweight='bold')
ax_left.text(
    5, 9.1, f'{len(minimal_repetition_cases)} of {len(q3_minimal)} repetition cases',
    ha='center', va='center', fontsize=10, color=INK
)

left_cx = 6.3
left_box_h = 0.72
left_gap = 0.22
centers_left = draw_vertical_flow(
    ax_left, left_cx, left_items, left_colors, box_w=3.4, box_h=left_box_h, gap=left_gap, top_y=8.5
)
rep_top = centers_left[first_rep_idx][0]
rep_bottom = centers_left[last_rep_idx][1]
loop_x = left_cx - 3.4 / 2 - 0.35
ax_left.annotate(
    '', xy=(left_cx - 3.4 / 2, rep_top - left_box_h / 2 + 0.05), xytext=(left_cx - 3.4 / 2, rep_bottom + 0.05),
    arrowprops=dict(arrowstyle='-|>', color=RED_TITLE, lw=2, connectionstyle="arc3,rad=0.7")
)
ax_left.text(
    loop_x - 1.0, (rep_top + rep_bottom) / 2, f'{repeated_tool}\ncalled again', ha='center', va='center',
    fontsize=8.5, color=RED_TITLE, style='italic'
)
ax_left.text(
    5, 0.9, f"Real example: {minimal_example['run_id']}", ha='center', va='center',
    fontsize=9, color=INK, style='italic'
)

ax_right.text(5, 9.6, 'Structured Description', ha='center', va='center', fontsize=13, color=GREEN_TITLE, fontweight='bold')
ax_right.text(5, 9.1, '0 repetition cases', ha='center', va='center', fontsize=10, color=INK)

draw_vertical_flow(
    ax_right, 5, right_items, right_colors, box_w=3.6, box_h=0.85, gap=0.35, top_y=8.5, bold_last=True
)
ax_right.text(
    5, 0.9, f"Real example: {structured_example['run_id']}", ha='center', va='center',
    fontsize=9, color=INK, style='italic'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/figure4_repetition_loop.png', dpi=300, facecolor=BACKGROUND)
plt.close()

print('Figure 4 saved to figures/figure4_repetition_loop.png')

q5_shortcut_candidates = df[
    (df['question_no'] == 5) &
    (df['description_level'].isin(['minimal', 'current'])) &
    (df['analysis_tools'].apply(lambda t: t == ['get_preprocessing_recommendations']))
]
shortcut_example = q5_shortcut_candidates.iloc[0]

q5_full_candidates = df[
    (df['question_no'] == 5) &
    (df['description_level'] == 'enhanced') &
    (df['dataset'] == shortcut_example['dataset']) &
    (df['analysis_tools'].apply(len) > 1)
]
full_example = q5_full_candidates.iloc[0]

shortcut_sq = sequence_quality_score(
    shortcut_example['tool_sequence_parsed'],
    sequence_lookup.get((shortcut_example['dataset'], shortcut_example['question_no']), [])
)
full_sq = sequence_quality_score(
    full_example['tool_sequence_parsed'],
    sequence_lookup.get((full_example['dataset'], full_example['question_no']), [])
)

print(f"Q5 shortcut example: {shortcut_example['run_id']} — {shortcut_example['analysis_tools']} — sequence_quality={shortcut_sq:.3f}")
print(f"Q5 full exploration example: {full_example['run_id']} — {full_example['analysis_tools']} — sequence_quality={full_sq:.3f}")

shortcut_tools = list(shortcut_example['analysis_tools'])
full_tools = list(full_example['analysis_tools'])

fig5, (ax_left5, ax_right5) = plt.subplots(1, 2, figsize=(13, 6.5))
fig5.patch.set_facecolor(BACKGROUND)

fig5.suptitle(
    'The Q5 Shortcut: Skipping Straight to Recommendations vs Full Exploration',
    color=INK, fontsize=14, fontweight='bold'
)

for ax in (ax_left5, ax_right5):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BACKGROUND)

ax_left5.text(5, 9.5, 'Shortcut (Minimal/Current)', ha='center', va='center', fontsize=13, color=RED_TITLE, fontweight='bold')
centers_left5 = draw_vertical_flow(
    ax_left5, 5, shortcut_tools, [FAIL_COLOR for _ in shortcut_tools],
    box_w=4.6, box_h=0.85, gap=0.3, top_y=8.4
)
left5_bottom = centers_left5[-1][1]
ax_left5.text(
    5, left5_bottom - 0.5, f'Sequence Quality: {shortcut_sq:.3f}', ha='center', va='center',
    fontsize=12, color=RED_TITLE, fontweight='bold'
)
ax_left5.text(
    5, 0.9, f"Real example: {shortcut_example['run_id']}", ha='center', va='center',
    fontsize=9, color=INK, style='italic'
)

ax_right5.text(5, 9.5, 'Full Exploration (Structured)', ha='center', va='center', fontsize=13, color=GREEN_TITLE, fontweight='bold')
centers_right5 = draw_vertical_flow(
    ax_right5, 5, full_tools, [OK_COLOR for _ in full_tools],
    box_w=4.6, box_h=0.85, gap=0.3, top_y=8.4, bold_last=True
)
right5_bottom = centers_right5[-1][1]
ax_right5.text(
    5, right5_bottom - 0.5, f'Sequence Quality: {full_sq:.3f}', ha='center', va='center',
    fontsize=12, color=GREEN_TITLE, fontweight='bold'
)
ax_right5.text(
    5, 0.9, f"Real example: {full_example['run_id']}", ha='center', va='center',
    fontsize=9, color=INK, style='italic'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/figure5_q5_shortcut.png', dpi=300, facecolor=BACKGROUND)
plt.close()

print('Figure 5 saved to figures/figure5_q5_shortcut.png')

all_tools_flat = [tool for seq in df['analysis_tools'] for tool in seq]
top_tools = [tool for tool, _ in Counter(all_tools_flat).most_common(8)]


def build_transition_matrix(sequences, tools):
    idx = {tool: i for i, tool in enumerate(tools)}
    n = len(tools)
    counts = np.zeros((n, n))
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
    return probs


maroon_cmap = LinearSegmentedColormap.from_list('white_to_maroon', ['#FFFFFF', '#5C1A2E'])

fig6, axes6 = plt.subplots(1, 3, figsize=(21, 7.5), gridspec_kw={'wspace': 0.55})
fig6.patch.set_facecolor(BACKGROUND)
fig6.suptitle('Tool Transition Probabilities by Description Level', color=INK, fontsize=15, fontweight='bold')

im6 = None
for panel_idx, (ax6, level) in enumerate(zip(axes6, LEVEL_ORDER)):
    sequences = df.loc[df['description_level'] == level, 'analysis_tools']
    probs = build_transition_matrix(sequences, top_tools)
    im6 = ax6.imshow(probs, cmap=maroon_cmap, vmin=0, vmax=1)
    ax6.set_xticks(range(len(top_tools)))
    ax6.set_yticks(range(len(top_tools)))
    ax6.set_xticklabels(top_tools, rotation=45, ha='right', fontsize=7.5, color=INK)
    if panel_idx == 0:
        ax6.set_yticklabels(top_tools, fontsize=7.5, color=INK)
    else:
        ax6.set_yticklabels([])
    ax6.set_title(LEVEL_LABELS[level], color=INK, fontsize=12, fontweight='bold')
    ax6.set_xlabel('To tool', color=INK, fontsize=9)
    for i in range(len(top_tools)):
        for j in range(len(top_tools)):
            val = probs[i, j]
            text_color = 'white' if val > 0.5 else INK
            ax6.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=text_color)

axes6[0].set_ylabel('From tool', color=INK, fontsize=9)

cbar = fig6.colorbar(im6, ax=axes6, shrink=0.8, pad=0.02)
cbar.set_label('Transition probability', color=INK)
cbar.ax.tick_params(colors=INK)

plt.savefig('figures/figure6_markov_transitions.png', dpi=300, facecolor=BACKGROUND, bbox_inches='tight')
plt.close()

print('Figure 6 saved to figures/figure6_markov_transitions.png')

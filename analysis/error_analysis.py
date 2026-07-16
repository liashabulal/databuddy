import pandas as pd

from analysis.load_data import df

EXPECTED_COUNTS = {1: 7.7, 2: 4.0, 3: 3.0, 4: 2.0, 5: 4.0}

CATEGORY_ORDER = [
    'Q3 categorical failure',
    'Q5 shortcut',
    'Tool repetition',
    'Multi-column outlier check',
    'Single tool',
    'Over-exploration',
    'Under-exploration',
    'Normal',
]


def classify_behavior(row):
    if row['question_no'] == 3 and str(row['status']).startswith('FAILED'):
        return 'Q3 categorical failure'

    if row['question_no'] == 5 and row['analysis_tools'] == ['get_preprocessing_recommendations']:
        return 'Q5 shortcut'

    tools = row['analysis_tools']
    repeated_tools = {t for t in set(tools) if tools.count(t) > 1}
    if repeated_tools:
        legitimate_q4_repetition = row['question_no'] == 4 and repeated_tools == {'detect_outliers_iqr'}
        if legitimate_q4_repetition:
            return 'Multi-column outlier check'
        return 'Tool repetition'

    if row['n_tools_called'] == 1:
        return 'Single tool'

    expected = EXPECTED_COUNTS[row['question_no']]
    if row['n_tools_called'] > expected * 1.5:
        return 'Over-exploration'
    if row['n_tools_called'] < expected * 0.5:
        return 'Under-exploration'

    return 'Normal'


df['behavior_category'] = df.apply(classify_behavior, axis=1)

overall_counts = df['behavior_category'].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
overall_pct = (overall_counts / len(df) * 100).round(2)

taxonomy_df = pd.DataFrame({
    'category': CATEGORY_ORDER,
    'count': overall_counts.values,
    'percent': overall_pct.values,
})

print('=== Overall Behavioral Taxonomy ===')
print(taxonomy_df.to_string(index=False))

level_counts = pd.crosstab(df['behavior_category'], df['description_level']).reindex(CATEGORY_ORDER, fill_value=0)
level_pct = (level_counts / level_counts.sum(axis=0) * 100).round(2)

print('\n=== Counts by Description Level ===')
print(level_counts.to_string())

print('\n=== Percentages by Description Level ===')
print(level_pct.to_string())

LEVEL_DISPLAY_ORDER = ['minimal', 'current', 'enhanced']
LEVEL_DISPLAY_NAMES = {'minimal': 'Minimal', 'current': 'Current', 'enhanced': 'Structured'}

paper_table = pd.DataFrame({'Category': CATEGORY_ORDER})
paper_table['N'] = overall_counts.values
paper_table['%'] = overall_pct.map(lambda v: f'{v:.1f}%').values

for level in LEVEL_DISPLAY_ORDER:
    name = LEVEL_DISPLAY_NAMES[level]
    paper_table[f'{name} N'] = level_counts[level].values
    paper_table[f'{name} %'] = level_pct[level].map(lambda v: f'{v:.1f}%').values

print('\n=== Full Taxonomy Table (paper-ready) ===')
print(paper_table.to_markdown(index=False))

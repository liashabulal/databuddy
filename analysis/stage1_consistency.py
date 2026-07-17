import pandas as pd

from analysis.load_data import df, reference_lookup, sequence_lookup


def compute_group_consistency(group):
    sequences = group['analysis_tools'].apply(tuple)
    sequence_consistency = sequences.value_counts().iloc[0] / len(group)

    tool_sets = group['analysis_tools'].apply(frozenset)
    set_consistency = tool_sets.value_counts().iloc[0] / len(group)

    return pd.Series({
        'n_runs': len(group),
        'sequence_consistency': sequence_consistency,
        'set_consistency': set_consistency,
    })


consistency_df = (
    df.groupby(['dataset', 'description_level', 'question_no'])
    .apply(compute_group_consistency, include_groups=False)
    .reset_index()
)

consistency_df['reference_tools'] = consistency_df.apply(
    lambda row: reference_lookup.get((row['dataset'], row['question_no']), []), axis=1
)
consistency_df['reference_sequence'] = consistency_df.apply(
    lambda row: sequence_lookup.get((row['dataset'], row['question_no']), []), axis=1
)

print(f'Consistency computed for {len(consistency_df)} (dataset, description_level, question_no) groups')
print()

summary = consistency_df.groupby('description_level')[['sequence_consistency', 'set_consistency']].mean().round(3)
summary['n_groups'] = consistency_df.groupby('description_level').size()

print('=== Consistency Summary by Description Level ===')
print(summary.to_string())

consistency_df.to_csv('data/stage1_consistency.csv', index=False)
print('\nSaved to data/stage1_consistency.csv')

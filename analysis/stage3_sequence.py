from analysis.load_data import df, sequence_lookup, sequence_quality_score

df['sequence_quality'] = df.apply(
    lambda row: sequence_quality_score(
        row['tool_sequence_parsed'], sequence_lookup.get((row['dataset'], row['question_no']), [])
    ),
    axis=1
)

sq_by_level = df.groupby('description_level')['sequence_quality'].agg(['mean', 'std', 'count']).round(3)

print('=== Sequence Quality Summary by Description Level ===')
print(sq_by_level.to_string())

sq_pivot = df.pivot_table(
    index='question_no', columns='description_level', values='sequence_quality', aggfunc='mean'
).round(3)

print('\n=== Mean Sequence Quality by Question Number x Description Level ===')
print(sq_pivot.to_string())

df.to_csv('data/stage3_sequence.csv', index=False)
print('\nSaved to data/stage3_sequence.csv')

sequence_df = df

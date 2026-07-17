from analysis.load_data import df, metrics_df, reference_lookup

metrics_df['expected_tool_count'] = df.apply(
    lambda row: len(reference_lookup.get((row['dataset'], row['question_no']), [])), axis=1
)

f1_by_level = metrics_df.groupby('description_level')['f1'].agg(['mean', 'std', 'count']).round(3)
f1_by_question = metrics_df.groupby('question_no')['f1'].agg(['mean', 'std', 'count']).round(3)

print('=== F1 Summary by Description Level ===')
print(f1_by_level.to_string())

print('\n=== F1 Summary by Question Number ===')
print(f1_by_question.to_string())

f1_pivot = metrics_df.pivot_table(
    index='question_no', columns='description_level', values='f1', aggfunc='mean'
).round(3)

print('\n=== Mean F1 by Question Number x Description Level ===')
print(f1_pivot.to_string())

metrics_df.to_csv('data/stage2_correctness.csv', index=False)
print('\nSaved to data/stage2_correctness.csv')

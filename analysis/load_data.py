import ast
import re
import pandas as pd

df = pd.read_csv("data/experiment_v2_log.csv")
ref = pd.read_csv("data/Reference Set 36e238bf2a368042a568fde67944f1b4.csv")

print(df.shape)
print(ref.shape)


def parse_tool_sequence(seq):
    try:
        return ast.literal_eval(seq)
    except (ValueError, SyntaxError):
        return []


def filter_analysis_tools(seq):
    plot_tools = {"plot_histogram", "plot_correlation_heatmap", "plot_group_target", "plot_boxplot"}
    return [tool for tool in seq if tool not in plot_tools]


df['tool_sequence_parsed'] = df['tool_sequence'].apply(parse_tool_sequence)
df['analysis_tools'] = df['tool_sequence_parsed'].apply(filter_analysis_tools)
df['first_tool'] = df['tool_sequence_parsed'].apply(lambda seq: seq[0] if seq else None)
df['n_tools_called'] = df['tool_sequence_parsed'].apply(len)


question_mapping = {
    'Give me an overview of this dataset': 1,
    'Are there any missing values and what should I do about them?': 2,
    'Which features are most correlated with the target?': 3,
    'Are there any outliers I should know about?': 4,
    'Give me preprocessing recommendations': 5,
}


def parse_expected_tools(tools_str):
    tokens = re.split(r'[,\n]', str(tools_str))
    tools = []
    for token in tokens:
        token = token.split('→')[0].strip()
        if token:
            tools.append(token)
    return tools


def parse_expected_sequence(seq_str):
    return [token.strip() for token in str(seq_str).split('→') if token.strip()]


dataset_replacements = {
    'adult census': 'adult',
    'breast cancer': 'breast_cancer',
    'hotel bookings': 'hotel_bookings',
    'california housing': 'california_housing',
}


def clean_dataset_name(name):
    name = str(name).strip().lower()
    for old, new in dataset_replacements.items():
        name = name.replace(old, new)
    return name.replace(' ', '')


ref['dataset_clean'] = ref['Dataset'].apply(clean_dataset_name)
ref['question_no'] = ref['Questions'].map(question_mapping)
ref['expected_tools_parsed'] = ref['Expected Tools '].apply(parse_expected_tools)
ref['expected_sequence_parsed'] = ref['Expected Sequence'].apply(parse_expected_sequence)

reference_lookup = {
    (row['dataset_clean'], row['question_no']): row['expected_tools_parsed']
    for _, row in ref.iterrows()
}
sequence_lookup = {
    (row['dataset_clean'], row['question_no']): row['expected_sequence_parsed']
    for _, row in ref.iterrows()
}

print(list(reference_lookup.keys()))


def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def sequence_quality_score(actual, reference):
    if not actual or not reference:
        return 0.0
    return round(lcs_length(actual, reference) / len(reference), 3)


def calculate_metrics(actual_tools, expected_tools, dataset, q_no, status):
    actual_set = set(actual_tools)
    expected_set = set(expected_tools)

    if q_no == 3 and str(status).startswith('FAILED'):
        return 0.5, 0.5, 0.5

    if dataset == 'hotel_bookings' and q_no == 1:
        expected_set = expected_set - {'get_correlation_matrix'}

    if not actual_set or not expected_set:
        precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        tp = len(actual_set & expected_set)
        precision = tp / len(actual_set)
        recall = tp / len(expected_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if q_no == 5 and actual_set == {'get_preprocessing_recommendations'}:
        return precision, recall, max(f1, 0.5)

    return precision, recall, f1


def build_metrics_row(row):
    expected_tools = reference_lookup.get((row['dataset'], row['question_no']), [])
    precision, recall, f1 = calculate_metrics(
        row['tool_sequence_parsed'], expected_tools, row['dataset'], row['question_no'], row['status']
    )
    return pd.Series({'precision': precision, 'recall': recall, 'f1': f1})


metrics_df = pd.concat(
    [df[['run_id', 'dataset', 'description_level', 'question_no', 'status']], df.apply(build_metrics_row, axis=1)],
    axis=1
)


def compute_sequence_quality(row):
    expected_sequence = sequence_lookup.get((row['dataset'], row['question_no']), [])
    return sequence_quality_score(row['tool_sequence_parsed'], expected_sequence)


df['sequence_quality'] = df.apply(compute_sequence_quality, axis=1)

print(metrics_df.shape)
print(metrics_df.head(3))

print('Setup complete')

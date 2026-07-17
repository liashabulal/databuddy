import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal

from analysis.load_data import df, metrics_df
from analysis.stage3_sequence import sequence_df


def fit_and_report(data, outcome, label):
    model = smf.mixedlm(
        f'{outcome} ~ C(description_level)',
        data=data,
        groups=data['dataset'],
        re_formula='1',
        vc_formula={'question_no': '0 + C(question_no)'}
    )
    result = model.fit()

    print(f'=== Linear Mixed Effects Model: {label} ===')
    print(result.summary())

    fe_names = result.fe_params.index
    fixed_effects = pd.DataFrame({
        'coef': result.fe_params,
        'p_value': result.pvalues.loc[fe_names],
    }).join(result.conf_int().loc[fe_names].rename(columns={0: 'ci_lower', 1: 'ci_upper'}))

    print(f'\n--- {label}: Fixed Effects (coef, p-value, 95% CI) ---')
    print(fixed_effects.round(4).to_string())

    print(f'\n--- {label}: Random Effects Variance Components ---')
    print(f"Group (dataset) variance: {result.cov_re.iloc[0, 0]:.5f}")
    print(f"question_no variance:     {result.vcomp[0]:.5f}")
    print(f"Residual variance:        {result.scale:.5f}")
    print()

    return result


f1_result = fit_and_report(metrics_df, 'f1', 'F1 Score')
seq_result = fit_and_report(sequence_df, 'sequence_quality', 'Sequence Quality')

consistency_df = pd.read_csv('data/stage1_consistency.csv')

print('=== Kruskal-Wallis: Sequence Consistency by Description Level ===')
seq_groups = [g['sequence_consistency'].values for _, g in consistency_df.groupby('description_level')]
seq_stat, seq_p = kruskal(*seq_groups)
print(f'H-statistic = {seq_stat:.4f}, p-value = {seq_p:.4f}')

print('\n=== Kruskal-Wallis: Set Consistency by Description Level ===')
set_groups = [g['set_consistency'].values for _, g in consistency_df.groupby('description_level')]
set_stat, set_p = kruskal(*set_groups)
print(f'H-statistic = {set_stat:.4f}, p-value = {set_p:.4f}')

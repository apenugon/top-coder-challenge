import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_data():
    """Load and prepare the public cases data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for case in data:
        input_data = case['input']
        rows.append({
            'trip_duration_days': input_data['trip_duration_days'],
            'miles_traveled': input_data['miles_traveled'],
            'total_receipts_amount': input_data['total_receipts_amount'],
            'reimbursement': case['expected_output']
        })
    
    return pd.DataFrame(rows)

def create_interview_features(df):
    """Create features based on interview insights"""
    df = df.copy()
    
    # Basic derived features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    
    # === Interview-based features ===
    
    # 1. The "5-day sweet spot" - Lisa mentioned 5-day trips get bonuses
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)
    
    # 2. The "4-6 day sweet spot range" - Jennifer mentioned 4-6 days optimal
    df['is_sweet_spot_length'] = ((df['trip_duration_days'] >= 4) & 
                                  (df['trip_duration_days'] <= 6)).astype(int)
    
    # 3. Kevin's "efficiency sweet spot" - 180-220 miles per day
    df['in_efficiency_sweet_spot'] = ((df['miles_per_day'] >= 180) & 
                                      (df['miles_per_day'] <= 220) & 
                                      (df['trip_duration_days'] > 1)).astype(int)
    
    # 4. "High efficiency" - Kevin mentions bonuses for high miles/day (but not too high)
    df['high_efficiency'] = ((df['miles_per_day'] >= 150) & 
                            (df['miles_per_day'] <= 300) & 
                            (df['trip_duration_days'] > 1)).astype(int)
    
    # 5. "Low spending penalty" - Dave mentions submitting tiny receipts is worse than nothing
    df['tiny_receipts_penalty'] = ((df['total_receipts_amount'] > 0) & 
                                   (df['total_receipts_amount'] < 50)).astype(int)
    
    # 6. "Optimal spending range" - Kevin mentions $600-800 gets good treatment
    df['optimal_spending_range'] = ((df['total_receipts_amount'] >= 600) & 
                                    (df['total_receipts_amount'] <= 800)).astype(int)
    
    # 7. "Vacation penalty" - Kevin mentions 8+ day trips with high spending get penalized
    df['vacation_penalty'] = ((df['trip_duration_days'] >= 8) & 
                             (df['receipts_per_day'] > 100)).astype(int)
    
    # 8. "Rounding bug" - Lisa mentions receipts ending in .49 or .99 get extra
    df['receipts_cents'] = (df['total_receipts_amount'] * 100) % 100
    df['favorable_rounding'] = ((df['receipts_cents'] == 49) | 
                               (df['receipts_cents'] == 99)).astype(int)
    
    # 9. Kevin's "sweet spot combo" - 5 days + 180+ miles/day + <$100/day
    df['kevin_sweet_combo'] = ((df['trip_duration_days'] == 5) & 
                              (df['miles_per_day'] >= 180) & 
                              (df['receipts_per_day'] < 100)).astype(int)
    
    # 10. "Short trip, high mileage" pattern
    df['short_high_mileage'] = ((df['trip_duration_days'] <= 2) & 
                               (df['miles_traveled'] >= 100)).astype(int)
    
    # 11. "Long trip, low mileage" pattern  
    df['long_low_mileage'] = ((df['trip_duration_days'] >= 6) & 
                             (df['miles_per_day'] < 50)).astype(int)
    
    # 12. Trip efficiency categories based on interview clustering ideas
    df['trip_type'] = 'medium'
    df.loc[(df['trip_duration_days'] <= 2) & (df['miles_per_day'] >= 100), 'trip_type'] = 'quick_high_mileage'
    df.loc[(df['trip_duration_days'] >= 6) & (df['miles_per_day'] < 50), 'trip_type'] = 'long_low_mileage'
    df.loc[(df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6) & 
           (df['miles_per_day'] >= 100) & (df['miles_per_day'] <= 200), 'trip_type'] = 'balanced'
    
    # 13. Calculate reimbursement efficiency metrics
    base_per_diem = 100  # Assume $100/day base
    df['base_expected'] = df['trip_duration_days'] * base_per_diem
    df['reimbursement_ratio'] = df['reimbursement'] / df['base_expected']
    df['excess_reimbursement'] = df['reimbursement'] - df['base_expected']
    
    # 14. Mileage tiers (Lisa mentioned tiered mileage rates)
    df['mileage_tier'] = 'low'
    df.loc[df['miles_traveled'] >= 100, 'mileage_tier'] = 'medium'
    df.loc[df['miles_traveled'] >= 300, 'mileage_tier'] = 'high'
    df.loc[df['miles_traveled'] >= 600, 'mileage_tier'] = 'very_high'
    
    return df

def test_interview_hypothesis(df, feature_name, description):
    """Test a specific feature hypothesis"""
    print(f"\n=== Testing: {description} ===")
    
    if feature_name not in df.columns:
        print(f"Feature {feature_name} not found!")
        return
    
    # Compare reimbursement for cases with/without this feature
    has_feature = df[df[feature_name] == 1]
    no_feature = df[df[feature_name] == 0]
    
    if len(has_feature) == 0 or len(no_feature) == 0:
        print("Not enough data points for comparison")
        return
    
    has_feature_reimb = has_feature['reimbursement']
    no_feature_reimb = no_feature['reimbursement']
    
    # Calculate statistics
    has_mean = has_feature_reimb.mean()
    no_mean = no_feature_reimb.mean()
    difference = has_mean - no_mean
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(has_feature_reimb, no_feature_reimb)
    
    print(f"Cases with feature: {len(has_feature)} (avg reimbursement: ${has_mean:.2f})")
    print(f"Cases without feature: {len(no_feature)} (avg reimbursement: ${no_mean:.2f})")
    print(f"Difference: ${difference:.2f}")
    print(f"T-test p-value: {p_value:.6f}")
    
    if p_value < 0.001:
        significance = "*** HIGHLY SIGNIFICANT"
    elif p_value < 0.01:
        significance = "** SIGNIFICANT"
    elif p_value < 0.05:
        significance = "* MARGINALLY SIGNIFICANT"
    else:
        significance = "NOT SIGNIFICANT"
    
    print(f"Statistical significance: {significance}")
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(has_feature) - 1) * has_feature_reimb.std()**2 + 
                         (len(no_feature) - 1) * no_feature_reimb.std()**2) / 
                        (len(has_feature) + len(no_feature) - 2))
    cohens_d = difference / pooled_std
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    
    return {
        'feature': feature_name,
        'description': description,
        'n_with': len(has_feature),
        'n_without': len(no_feature),
        'mean_with': has_mean,
        'mean_without': no_mean,
        'difference': difference,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }

def analyze_categorical_features(df):
    """Analyze categorical features like trip_type and mileage_tier"""
    print("\n=== Categorical Feature Analysis ===")
    
    # Trip type analysis
    print("\n--- Trip Type Analysis ---")
    trip_type_stats = df.groupby('trip_type')['reimbursement'].agg(['count', 'mean', 'std']).round(2)
    print(trip_type_stats)
    
    # ANOVA test for trip types
    trip_types = df['trip_type'].unique()
    groups = [df[df['trip_type'] == tt]['reimbursement'] for tt in trip_types]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA p-value for trip types: {p_value:.6f}")
    
    # Mileage tier analysis
    print("\n--- Mileage Tier Analysis ---")
    mileage_stats = df.groupby('mileage_tier')['reimbursement'].agg(['count', 'mean', 'std']).round(2)
    print(mileage_stats)
    
    # ANOVA test for mileage tiers
    mileage_tiers = df['mileage_tier'].unique()
    groups = [df[df['mileage_tier'] == mt]['reimbursement'] for mt in mileage_tiers]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA p-value for mileage tiers: {p_value:.6f}")

def correlation_analysis(df):
    """Analyze correlations between features and reimbursement"""
    print("\n=== Correlation Analysis ===")
    
    # Select numeric features
    numeric_features = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_per_day', 'receipts_per_day', 'reimbursement_ratio',
        'excess_reimbursement'
    ]
    
    corr_with_reimbursement = df[numeric_features + ['reimbursement']].corr()['reimbursement'].abs().sort_values(ascending=False)
    
    print("Correlations with reimbursement (absolute values):")
    for feature, corr in corr_with_reimbursement.items():
        if feature != 'reimbursement':
            print(f"{feature}: {corr:.3f}")

def regression_analysis(df):
    """Perform regression analysis with interview-based features"""
    print("\n=== Regression Analysis with Interview Features ===")
    
    # Select features for regression
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_per_day', 'receipts_per_day',
        'is_5_day_trip', 'is_sweet_spot_length', 'in_efficiency_sweet_spot',
        'high_efficiency', 'tiny_receipts_penalty', 'optimal_spending_range',
        'favorable_rounding', 'kevin_sweet_combo', 'short_high_mileage'
    ]
    
    X = df[feature_cols]
    y = df['reimbursement']
    
    # Fit regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reg = LinearRegression()
    reg.fit(X_scaled, y)
    
    y_pred = reg.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    
    print(f"R² score: {r2:.3f}")
    
    # Feature coefficients
    print("\nFeature coefficients (standardized):")
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': reg.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    for _, row in coef_df.iterrows():
        print(f"{row['feature']}: {row['coefficient']:.2f}")
    
    return reg, scaler, coef_df

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} cases")
    
    print("\n=== Basic Data Overview ===")
    print(df.describe())
    
    # Create interview-based features
    print("\nCreating interview-based features...")
    df_features = create_interview_features(df)
    
    # Test each hypothesis from interviews
    hypotheses = [
        ('is_5_day_trip', "5-day trips get bonuses (Lisa's observation)"),
        ('is_sweet_spot_length', "4-6 day trips are optimal (Jennifer's pattern)"),
        ('in_efficiency_sweet_spot', "180-220 miles/day sweet spot (Kevin's analysis)"),
        ('high_efficiency', "High efficiency trips get bonuses (Marcus's theory)"),
        ('tiny_receipts_penalty', "Tiny receipts are penalized (Dave's experience)"),
        ('optimal_spending_range', "600-800 spending range optimal (Kevin's finding)"),
        ('vacation_penalty', "Long trips + high spending penalized (Kevin's theory)"),
        ('favorable_rounding', "Receipts ending in .49/.99 get bonuses (Lisa's bug)"),
        ('kevin_sweet_combo', "5 days + 180+ miles/day + <$100/day combo"),
        ('short_high_mileage', "Short high-mileage trips treated special"),
    ]
    
    results = []
    for feature, description in hypotheses:
        result = test_interview_hypothesis(df_features, feature, description)
        if result:
            results.append(result)
    
    # Analyze categorical features
    analyze_categorical_features(df_features)
    
    # Correlation analysis
    correlation_analysis(df_features) 
    
    # Regression analysis
    reg_model, scaler, coefficients = regression_analysis(df_features)
    
    # Summary of significant findings
    print("\n=== SUMMARY OF SIGNIFICANT FINDINGS ===")
    significant_results = [r for r in results if r['significant']]
    significant_results.sort(key=lambda x: abs(x['difference']), reverse=True)
    
    print(f"Found {len(significant_results)} statistically significant patterns:")
    for result in significant_results:
        print(f"• {result['description']}")
        print(f"  Effect: ${result['difference']:.2f} difference (p={result['p_value']:.4f})")
        print(f"  Sample sizes: {result['n_with']} vs {result['n_without']} cases")
    
    if not significant_results:
        print("No statistically significant patterns found in the interview theories.")
        print("This could mean:")
        print("1. The theories are incorrect")
        print("2. The sample size is too small to detect effects") 
        print("3. The effects exist but require more complex interactions")
    
    return df_features, results, reg_model

if __name__ == "__main__":
    data, hypothesis_results, model = main() 
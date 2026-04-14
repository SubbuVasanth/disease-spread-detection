"""
analysis.py
DiseaseSpread — SRM IST Mini Project 2026
-----------------------------------------
Statistical analysis using real EpiClim columns:
cases, deaths, temp_celsius, rainfall_mm,
leaf_area_index, density_per_sqkm, season
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('district').agg(
        total_cases   = ('cases', 'sum'),
        mean_cases    = ('cases', 'mean'),
        max_cases     = ('cases', 'max'),
        total_deaths  = ('deaths', 'sum'),
        avg_temp_c    = ('temp_celsius', 'mean'),
        avg_rainfall  = ('rainfall_mm', 'mean'),
    ).round(2).reset_index().sort_values('total_cases', ascending=False)


def disease_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('disease')['cases'].agg(['sum','mean','count']).round(2)


def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    order = ['Winter','Summer','Monsoon','Post-Monsoon']
    result = df.groupby('season')['cases'].agg(['mean','sum','count']).round(2)
    return result.reindex([s for s in order if s in result.index])


def climate_disease_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between climate variables and case counts."""
    climate_vars = ['temp_celsius','rainfall_mm','leaf_area_index','density_per_sqkm']
    results = []
    for var in climate_vars:
        if var in df.columns:
            clean = df[[var,'cases']].dropna()
            r, p = stats.pearsonr(clean[var], clean['cases'])
            results.append({
                'variable':    var,
                'correlation': round(r, 4),
                'p_value':     round(p, 4),
                'significant': '✅ Yes' if p < 0.05 else '❌ No'
            })
    return pd.DataFrame(results)


def detect_surges(df: pd.DataFrame, multiplier: float = 2.0) -> pd.DataFrame:
    """Flag records where cases exceed mean + multiplier*std per district."""
    df = df.copy()
    df['surge'] = False
    for district, group in df.groupby('district'):
        threshold = group['cases'].mean() + multiplier * group['cases'].std()
        df.loc[group[group['cases'] > threshold].index, 'surge'] = True
    surges = df[df['surge']].sort_values('cases', ascending=False)
    print(f"[analysis] Surge events detected: {len(surges)}")
    return surges


def top_risk_districts(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    from preprocessing import compute_ors
    return compute_ors(df).head(top_n)[['district','ORS','risk_level']]


if __name__ == '__main__':
    print("Run preprocessing.py first to generate merged_clean.csv, then import this module.")
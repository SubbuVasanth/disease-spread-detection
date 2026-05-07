"""
preprocessing.py
DiseaseSpread — SRM IST Mini Project 2026
-----------------------------------------
Loads EpiClim CSV + Census 2011, cleans,
merges, and engineers features.

EpiClim columns:
  week_of_outbreak, state_ut, district, Disease,
  Cases, Deaths, day, mon, year,
  Latitude, Longitude, preci, LAI, Temp (Kelvin)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


# ── Census 2011 — Tamil Nadu ──────────────────────────────────────────────────
# Source: Office of the Registrar General & Census Commissioner, India

CENSUS_2011 = {
    'district': [
        'Chennai','Coimbatore','Madurai','Tiruchirappalli','Salem',
        'Tirunelveli','Vellore','Erode','Thanjavur','Dindigul',
        'Tiruppur','Kanchipuram','Krishnagiri','Cuddalore','Nagapattinam',
        'Namakkal','Theni','Villupuram','The Nilgiris','Pudukkottai',
        'Ramanathapuram','Sivaganga','Thoothukudi','Tiruvannamalai','Virudhunagar'
    ],
    'population': [
        7088000, 3458127, 3038252, 2722290, 3482056,
        3072880, 3927780, 2251744, 2405890, 2159775,
        1795974, 3998252, 1879809, 2605914, 1616450,
        1726601, 1243003, 3458873, 735394,  1618345,
        1353445, 1339101, 1750176, 2464875, 1942288
    ],
    'area_sq_km': [
        426,  4723, 3741, 4404, 5245,
        6823, 6077, 5714, 3481, 6267,
        5189, 4432, 5051, 3703, 2715,
        3363, 3242, 7194, 2545, 4663,
        4175, 4189, 4621, 6191, 4283
    ]
}

SEASON_MAP = {
    12: 'Winter',      1: 'Winter',       2: 'Winter',
    3:  'Summer',      4: 'Summer',       5: 'Summer',
    6:  'Monsoon',     7: 'Monsoon',      8: 'Monsoon',  9: 'Monsoon',
    10: 'Post-Monsoon', 11: 'Post-Monsoon'
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_epiclim(filepath: str, state_filter: str = 'Tamil') -> pd.DataFrame:
    """Load and filter EpiClim CSV by state name substring."""
    df = pd.read_csv(filepath)
    print(f"[load] Full EpiClim shape: {df.shape}")

    df = df[df['state_ut'].str.contains(state_filter, case=False, na=False)].copy()
    df = df.reset_index(drop=True)
    print(f"[load] After '{state_filter}' filter: {df.shape}")
    return df


def build_census() -> pd.DataFrame:
    """Return Census 2011 Tamil Nadu DataFrame with density column."""
    df = pd.DataFrame(CENSUS_2011)
    df['density_per_sqkm'] = (df['population'] / df['area_sq_km']).round(2)
    df['district_key'] = df['district'].str.strip().str.lower()
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def rename_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, convert Kelvin → Celsius, create date."""
    df = df.rename(columns={
        'state_ut'        : 'state',
        'week_of_outbreak': 'week_label',
        'Disease'         : 'disease',
        'Cases'           : 'cases',
        'Deaths'          : 'deaths',
        'Latitude'        : 'latitude',
        'Longitude'       : 'longitude',
        'preci'           : 'rainfall_mm',
        'LAI'             : 'leaf_area_index',
    })
    # Kelvin → Celsius
    df['cases'] = pd.to_numeric(df['cases'], errors='coerce')
    df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce')
    df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
    df['leaf_area_index'] = pd.to_numeric(df['leaf_area_index'], errors='coerce')
    
    # Safe temp parsing
    df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
    df['temp_celsius'] = (df['Temp'] - 273.15).round(2)
    df = df.drop(columns=['Temp'])

    # Date column
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' +
        df['mon'].astype(str).str.zfill(2) + '-' +
        df['day'].astype(str).str.zfill(2),
        errors='coerce'
    )
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values."""
    before = df.isnull().sum().sum()
    df['cases']           = df.groupby(['district','disease'])['cases'].transform(
        lambda x: x.fillna(x.median()))
    df['deaths']          = df['deaths'].fillna(0)
    df['rainfall_mm']     = df['rainfall_mm'].fillna(df['rainfall_mm'].median())
    df['temp_celsius']    = df['temp_celsius'].fillna(df['temp_celsius'].median())
    df['leaf_area_index'] = df['leaf_area_index'].fillna(df['leaf_area_index'].median())
    after = df.isnull().sum().sum()
    print(f"[clean] Missing values: {before} → {after}")
    return df


# ── Merge & Normalize ─────────────────────────────────────────────────────────

def merge_census(df: pd.DataFrame, df_census: pd.DataFrame) -> pd.DataFrame:
    """Merge EpiClim with Census 2011 on district name."""
    df['district_key'] = df['district'].str.strip().str.lower()
    merged = df.merge(
        df_census[['district_key','population','density_per_sqkm']],
        on='district_key', how='left'
    )
    # Fallback for unmatched districts
    median_density = df_census['density_per_sqkm'].median()
    median_pop     = df_census['population'].median()
    merged['density_per_sqkm'] = merged['density_per_sqkm'].fillna(median_density)
    merged['population']        = merged['population'].fillna(median_pop)
    merged = merged.drop(columns=['district_key'])
    print(f"[merge] Shape after census merge: {merged.shape}")
    return merged


def normalize_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Add cases per 100K population."""
    df['cases_per_100k'] = (df['cases'] / df['population'] * 100_000).round(4)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add season, quarter, and rolling average."""
    df['season']  = df['mon'].map(SEASON_MAP)
    df['quarter'] = ((df['mon'] - 1) // 3 + 1)

    df = df.sort_values(['district','disease','year','mon','day'])
    df['cases_4wk_avg'] = (
        df.groupby(['district','disease'])['cases']
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
        .round(2)
    )
    return df


# ── ORS ───────────────────────────────────────────────────────────────────────

def compute_ors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Outbreak Risk Score per district.
    Weights: cases=40%, rainfall=25%, temp=20%, density=15%
    """
    summary = df.groupby('district').agg(
        avg_cases    = ('cases', 'mean'),
        avg_rainfall = ('rainfall_mm', 'mean'),
        avg_temp     = ('temp_celsius', 'mean'),
        avg_density  = ('density_per_sqkm', 'mean')
    ).reset_index()

    scaler  = MinMaxScaler()
    factors = ['avg_cases','avg_rainfall','avg_temp','avg_density']
    summary[factors] = scaler.fit_transform(summary[factors])

    summary['ORS'] = (
        0.40 * summary['avg_cases'] +
        0.25 * summary['avg_rainfall'] +
        0.20 * summary['avg_temp'] +
        0.15 * summary['avg_density']
    ).round(4)

    summary['risk_level'] = pd.cut(
        summary['ORS'], bins=[0, 0.33, 0.66, 1.0],
        labels=['🟢 Low','🟡 Medium','🔴 High']
    )
    return summary.sort_values('ORS', ascending=False).reset_index(drop=True)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    epiclim_path: str,
    save_path: str = None
) -> pd.DataFrame:
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'processed_data', 'merged_clean.csv')

    print("=" * 55)
    print("DiseaseSpread — Preprocessing Pipeline")
    print("=" * 55)

    df       = load_epiclim(epiclim_path)
    df       = rename_and_convert(df)
    df       = handle_missing(df)
    df_census = build_census()
    df       = merge_census(df, df_census)
    df       = normalize_cases(df)
    df       = add_features(df)

    df.to_csv(save_path, index=False)
    print(f"\n✅ Saved → {save_path}  |  Shape: {df.shape}")
    return df


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    default_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'Data', 'epiclim_data.csv')
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    df = run_pipeline(path)
    print("\nPreview:")
    print(df[['district','disease','year','mon','cases','temp_celsius',
              'rainfall_mm','density_per_sqkm','ORS' if 'ORS' in df.columns else 'cases_per_100k']].head(5))
    ors = compute_ors(df)
    print("\n🏆 Outbreak Risk Scores:")
    print(ors[['district','ORS','risk_level']].to_string(index=False))
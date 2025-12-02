import pandas as pd

# Minimal normalize_columns copied/adapted from main.py for repro

def normalize_columns(df: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    def find_variant(possible):
        for p in possible:
            if p.lower() in lower_cols:
                return lower_cols[p.lower()]
        for col in df.columns:
            lower_col = col.lower()
            for p in possible:
                if p.lower() in lower_col:
                    return col
        return None

    status_col = find_variant(['Status', 'studentstatus', 'student_status', 'enrollment_status', 'state'])
    hasjob_col = find_variant(['hasjob', 'HasSideJob', 'has_side_jobs', 'hasjobs', 'hassidejobs', 'employed', 'HasSideJobs'])
    income_col = find_variant(['MonthlyFamilyIncome', 'income', 'pay', 'wage', 'compensation'])

    if status_col:
        col_map[status_col] = 'Status'
    if hasjob_col:
        col_map[hasjob_col] = 'HasSideJob'
    if income_col:
        col_map[income_col] = 'MonthlyFamilyIncome'

    if col_map:
        df = df.rename(columns=col_map)

    if not for_prediction and 'Status' in df.columns:
        if pd.api.types.is_numeric_dtype(df['Status']):
            df['Status'] = df['Status'].fillna(0).astype(int).clip(0, 1)
        else:
            df['Status'] = df['Status'].astype(str).str.lower().str.strip().apply(lambda x: 1 if 'enrolled' in x else 0).astype(int)

    if 'HasSideJob' in df.columns or 'hasjob' in df.columns:
        try:
            if 'hasjob' in df.columns and 'HasSideJob' not in df.columns:
                df = df.rename(columns={'hasjob': 'HasSideJob'})
            df['HasSideJob'] = df['HasSideJob'].map(lambda v: 1 if str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't') else (0 if str(v).strip().lower() in ('0', 'false', 'no', 'n', 'f') else v))
            df['HasSideJob'] = pd.to_numeric(df['HasSideJob'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass

    if 'MonthlyFamilyIncome' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['MonthlyFamilyIncome']):
            df['MonthlyFamilyIncome'] = df['MonthlyFamilyIncome'].astype('category').cat.codes
        df['MonthlyFamilyIncome'] = (df['MonthlyFamilyIncome'] - df['MonthlyFamilyIncome'].min()) / (df['MonthlyFamilyIncome'].max() - df['MonthlyFamilyIncome'].min())

    return df


if __name__ == '__main__':
    path = 'students_dummy_final.csv'
    df = pd.read_csv(path)
    print('Columns:', df.columns.tolist())

    # Reproduce selections
    combos = [
        ('Age', 'HasSideJob'),
        ('Age', 'Gender')
    ]

    for col, col2 in combos:
        temp_df = df.copy()
        temp_df = normalize_columns(temp_df, for_prediction=False)
        print('\n--- Testing', col, 'vs', col2, '---')
        # Check actual column names after normalization
        print('Available columns after normalize:', temp_df.columns.tolist())
        if col not in temp_df.columns:
            print(f"Column {col} not found")
            continue
        if col2 not in temp_df.columns:
            print(f"Column {col2} not found")
            continue
        y = temp_df[col]
        x = temp_df[col2]
        print(f"dtypes -> {col}: {y.dtype}, {col2}: {x.dtype}")
        print(f"is_numeric -> {col}: {pd.api.types.is_numeric_dtype(y)}, {col2}: {pd.api.types.is_numeric_dtype(x)}")
        print(f"unique {col2} values (first 10):", x.unique()[:10])
        # Simulate conversion behavior used in analyze_relationship
        def to_codes(s):
            if s.dtype == 'object':
                return s.astype('category').cat.codes
            return s
        x_codes = to_codes(x)
        print(f"After possible conversion, dtype {col2}: {x_codes.dtype}, unique codes (first 10): {x_codes.unique()[:10]}")

    print('\nDone.')

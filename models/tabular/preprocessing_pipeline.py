import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_pipeline():
    # 1. Columns
    columns_to_drop = [
        'patient_id', 'dataset', 'acquisition_times', 'mastectomy_post_nac', 'days_to_follow_up', 'days_to_recurrence',
        'days_to_metastasis', 'days_to_death', 'oncotype_score', 'nottingham_grade', 'patient_size', 'view',
        'bilateral_mri', 'num_phases', 'fat_suppressed', 'field_strength', 'image_rows', 'image_columns',
        'num_slices', 'pixel_spacing', 'slice_thickness', 'site', 'manufacturer', 'scanner_model', 'high_bit',
        'window_center', 'window_width', 'echo_time', 'repetition_time', 'acquisition_times', 'acquisition_date',
        'tcia_series_uid', 'nac_agent'
    ]

    columns_fill_minus1 = ['anti_her2_neu_therapy', 'endocrine_therapy', 'mammaprint', 'er', 'pr', 'hr', 'her2']
    columns_fill_0 = ['bilateral_breast_cancer', 'multifocal_cancer']
    columns_fill_median = ['age', 'weight']
    columns_fill_unknown = ['ethnicity', 'bmi_group', 'tumor_subtype', 'menopause']

    # These categorical feature names are generated after imputation; prefix shows the imputer used
    categorical_cols = [
        'fill_minus1__mammaprint', 'fill_minus1__anti_her2_neu_therapy', 'fill_minus1__endocrine_therapy',
        'fill_unknown__bmi_group', 'fill_unknown__menopause', 'fill_unknown__tumor_subtype',
        'fill_minus1__her2', 'fill_minus1__hr', 'fill_minus1__er', 'fill_minus1__pr',
        'fill_unknown__ethnicity'
    ]

    # 2. Column dropper
    dropper = FunctionTransformer(lambda X: X.drop(columns=columns_to_drop))

    # 3. Imputation
    column_imputer = ColumnTransformer(transformers=[
        ("fill_minus1", SimpleImputer(strategy="constant", fill_value=-1), columns_fill_minus1),
        ("fill_0", SimpleImputer(strategy="constant", fill_value=0), columns_fill_0),
        ("fill_median", SimpleImputer(strategy="median"), columns_fill_median),
        ("fill_unknown", SimpleImputer(strategy="constant", fill_value="unknown"), columns_fill_unknown),
    ], remainder='passthrough')

    # 4. Replace typos / aliases in the 'nac_agent' column
    def replace_nac_agent(df):
        df = df.copy()
        df['nac_agent'] = df['nac_agent'].replace({
            'AC + T (estimated)': 'Anthracycline + Taxane',
            'AC (estimated)': 'Anthracycline'
        })
        df['nac_agent'] = df['nac_agent'].str.replace('Anthracyline', 'Anthracycline', regex=False)
        return df

    # 5. Convert 'nac_agent' into binary indicator columns for each substance of interest
    def encode_agent(agent, substance):
        return 1 if substance in agent else 0

    def binary_encode_nac_agent(df):
        df = df.copy()
        substances = [
            'Anthracycline', 'Taxane', 'Paclitaxel', 'AMG 386', 'Neratinib', 'Ganitumab',
            'Ganetespib', 'ABT 888', 'Carboplatin', 'Pembrolizumab', 'T-DM1', 'Pertuzumab',
            'MK-2206', 'Trastuzumab', 'FEC100'
        ]
        for substance in substances:
            df[substance] = df['nac_agent'].apply(lambda x: encode_agent(x, substance))
        return df

    # 6. Cast categorical columns to string so OneHotEncoder can handle them
    def cast_categories(X):
        df_casted = pd.DataFrame(X, columns=column_imputer.get_feature_names_out())
        for col in categorical_cols:
            if col in df_casted.columns:
                df_casted[col] = df_casted[col].astype(str)
        return df_casted

    cast_categoricals = FunctionTransformer(cast_categories)

    # 7. One‑hot encode the selected categorical columns
    onehot = ColumnTransformer(transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ], remainder='passthrough')

    # 8. Final pipeline assembly
    pipeline = Pipeline(steps=[
        ("replace_nac_agent", FunctionTransformer(replace_nac_agent)),
        ("binary_encode_nac_agent", FunctionTransformer(binary_encode_nac_agent)),
        ("drop_columns", dropper),
        ("impute_columns", column_imputer),
        ("cast_categoricals", cast_categoricals),
        ("encode_categoricals", onehot)
    ])

    return pipeline

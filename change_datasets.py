import pandas as pd


df_bank = pd.read_csv('datasets/bank-marketing.csv')
df_bio = pd.read_csv('datasets/bio-response.csv')
df_blood = pd.read_csv('datasets/blood-transfusion-service-center.csv')
df_breast_cancer = pd.read_csv('datasets/breast-cancer.csv')
df_climate = pd.read_csv('datasets/climate-model-simulation-crashes.csv')
df_credit = pd.read_csv('datasets/credit-g.csv')
df_diabetes = pd.read_csv('datasets/diabetes.csv')
df_eucalyptus = pd.read_csv('datasets/eucalyptus.csv')
df_iris = pd.read_csv('datasets/iris.csv')
df_phishing = pd.read_csv('datasets/phishing-websites.csv')
df_transplant = pd.read_csv('datasets/transplant.csv')

df_bank.rename(columns={
    'V1': 'age',
    'V2': 'job',
    'V3': 'marital status',
    'V4': 'education',
    'V5': 'credit_default',
    'V6': 'balance',
    'V7': 'housing_loan',
    'V8': 'personal_loan',
    'V9': 'contact_type',
    'V10': 'day',
    'V11': 'month',
    'V12': 'duration_contact',
    'V13': 'campaign',
    'V14': 'pday',
    'V15': 'previous',
    'V16': 'poutcame',
    'Class': 'Target',
}, inplace=True)


df_bio.rename(columns={
    'target': 'Target'
}, inplace=True)


df_blood.rename(columns={
    'V1': 'recency',
    'V2': 'frequency',
    'V3': 'monetary',
    'V4': 'time',
    'Class': 'Target',
}, inplace=True)

df_blood['Target'] = df_blood['Target'].replace({2: 1, 1: 0})


df_breast_cancer.rename(columns={
    'Class': 'Target'
}, inplace=True)


df_climate.rename(columns={
    'outcome': 'Target',
}, inplace=True)


df_credit.rename(columns={
    'class': 'Target',
}, inplace=True)


df_diabetes.rename(columns={
    'preg': 'n_pregnancy',
    'plas': 'plasma',
    'pres': 'diastolic_blood_pressure',
    'skin': 'skin_thickness',
    'insu': 'insulin',
    'mass': 'body_mass',
    'pedi': 'pedigree',
    'age': 'age',
    'class': 'Target',
}, inplace=True)


df_eucalyptus.rename(columns={
    'Abbrev': 'site_abbreviation',
    'Rep': 'site_rep',
    'Locality': 'locality',
    'Map_Ref': 'map_ref',
    'Latitude': 'latitude',
    'Altitude': 'altitude',
    'Rainfall': 'rainfall',
    'Frosts': 'frosts',
    'Year': 'year',
    'Sp': 'species_code',
    'PMCno': 'seedlot_number',
    'DBH': 'best_diameter_base_height',
    'Ht': 'height',
    'Surv': 'survival',
    'Vig': 'vigour',
    'Ins_res': 'insect_resistance',
    'Stem_Fm': 'stem_form',
    'Crown_Fm': 'crown_form',
    'Brnch_Fm': 'branch_form',
    'Utility': 'Target',
}, inplace=True)


df_iris.rename(columns={
    'class': 'Target',
}, inplace=True)


df_phishing.rename(columns={
    'Result': 'Target',
}, inplace=True)


df_transplant.rename(columns={
    'obs': 'hospital',
    'e': 'expected_number_deaths',
    'z': 'number_deaths',
    'binaryClass': 'Target',
}, inplace=True)
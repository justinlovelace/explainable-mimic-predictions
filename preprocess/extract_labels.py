from os.path import join
import pandas as pd
import yaml
import numpy as np
from tqdm import tqdm
import os
import argparse
import utils
import clean_text

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='rootpath/resources/params.json', help="Directory containing params.json")

def preprocess_all_notes(params):
    data_dir = params.data_dir

    # get relevant data from NOTES table
    print('\nImporting data from NOTEEVENTS...')
    path_notes = join(data_dir, 'NOTEEVENTS.csv.gz')
    df_notes = pd.read_csv(path_notes)

    df_notes = df_notes[(df_notes['ISERROR'] != 1)]
    df_notes = df_notes[
        ['ROW_ID', 'HADM_ID', 'SUBJECT_ID', 'CHARTDATE', 'CHARTTIME', 'TEXT', 'CATEGORY']]

    clean_notes_all = clean_text.preprocess(df_notes)
    local_dir = params.local_data
    datasetPath = os.path.join(local_dir, 'df_MASTER_NOTES_ALL.csv')
    clean_notes_all.to_csv(datasetPath, index=False)


def get_notes(params, clean_notes_all):
    data_dir = params.data_dir
    local_dir = params.local_data

    clean_notes_all['CHARTDATE'] = clean_notes_all['CHARTDATE'].astype('datetime64[ns]')
    clean_notes_all['CHARTTIME'] = clean_notes_all['CHARTTIME'].astype('datetime64[ns]')
    print(clean_notes_all.dtypes)

    print('\nImporting data from ICUSTAYS...')
    path_icu = os.path.join(data_dir, 'ICUSTAYS.csv.gz')
    df_icu = pd.read_csv(path_icu)
    df_icu = df_icu[['ICUSTAY_ID', 'HADM_ID', 'SUBJECT_ID', 'INTIME', 'OUTTIME']]
    df_icu['INTIME'] = df_icu['INTIME'].astype('datetime64[ns]')
    df_icu['OUTTIME'] = df_icu['OUTTIME'].astype('datetime64[ns]')
    print(df_icu.dtypes)

    print('\nDropping ICUSTAYS with missing times...')
    df_icu = df_icu[df_icu.OUTTIME.isnull() == False]
    df_icu = df_icu[df_icu.INTIME.isnull() == False]

    print('\nImporting data from ADMISSIONS...')
    path_adm = os.path.join(data_dir, 'ADMISSIONS.csv.gz')
    df_adm = pd.read_csv(path_adm)
    df_adm = df_adm[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'DISCHARGE_LOCATION', 'HOSPITAL_EXPIRE_FLAG', 'DEATHTIME']]
    df_adm['HOSPITAL_EXPIRE_FLAG'] = df_adm['HOSPITAL_EXPIRE_FLAG'].astype('bool')
    print(df_adm.dtypes)

    print('\nImporting data from PATIENTS...')
    path_patients = os.path.join(data_dir, 'PATIENTS.csv.gz')
    df_patients = pd.read_csv(path_patients)
    df_patients = df_patients[['SUBJECT_ID', 'DOB', 'DOD', 'EXPIRE_FLAG']]
    df_patients['EXPIRE_FLAG'] = df_patients['EXPIRE_FLAG'].astype('bool')
    print(df_patients.dtypes)

    print('\nMerging ADMISSIONS and PATIENTS...')
    df_adm_pt = pd.merge(df_adm, df_patients, how='inner', on=['SUBJECT_ID'])
    ages = (df_adm_pt['ADMITTIME'].astype('datetime64[ns]') - df_adm_pt['DOB'].astype('datetime64[ns]')).dt.days / 365
    df_adm_pt['AGE'] = [age if age >= 0 else 91.4 for age in ages]
    df_adm_pt.drop(['DOB'], axis=1, inplace=True)
    # Removing minors from the data
    num_adm = len(df_adm_pt)
    df_adm_pt = df_adm_pt[(df_adm_pt['AGE'] >= 18)]
    df_adm_pt.drop(['AGE'], axis=1, inplace=True)
    print('Dropped ' + str(num_adm - len(df_adm_pt)) + ' minors')

    print('Merging ICUSTAYS with ADMISSIONS and PATIENTS...')
    df_icu_adm_pt = pd.merge(df_icu, df_adm_pt, how='inner', on=['HADM_ID', 'SUBJECT_ID'])
    print(df_icu_adm_pt.dtypes)

    print('\nMerging ICUSTAYS and Preprocessed Notes...')
    df_icu_notes = pd.merge(df_icu_adm_pt, clean_notes_all, how='inner', on=['HADM_ID', 'SUBJECT_ID'])
    print(df_icu_notes.dtypes)

    df_icu_notes = df_icu_notes[(df_icu_notes.CHARTDATE < df_icu_notes.OUTTIME.astype('datetime64[D]')) | (df_icu_notes.CHARTTIME < df_icu_notes.OUTTIME)]
    print('Total number of notes: ' + str(len(df_icu_notes)))

    datasetPath = os.path.join(local_dir, 'df_MASTER_NOTES_ICU.csv')
    df_icu_notes.to_csv(datasetPath, index=False)

    print("\nCalculating counts of notes...")
    df_icu_notes_count = df_icu_notes.groupby('ICUSTAY_ID')['TEXT'].size().reset_index(name='counts')
    # print(df_icu_notes_count)
    df_icu_notes_count = df_icu_notes_count[(df_icu_notes_count['counts'] > 2)]


    print('\nConcatenating notes...')
    df_icu_notes.sort_values(['ICUSTAY_ID', 'CHARTDATE'], ascending=[True, True])
    df_icu_notes = df_icu_notes.groupby('ICUSTAY_ID')['TEXT'].apply(lambda x: '\n'.join(map(str, x)))
    df_icu_notes = df_icu_notes.reset_index()
    print('Total number of icustays: ' + str(len(df_icu_notes)))

    print("\nDropping stays with less than 3 notes...")
    df_icu_notes = pd.merge(df_icu_notes, df_icu_notes_count, how='inner', on=['ICUSTAY_ID'])
    df_icu_notes = df_icu_notes[['ICUSTAY_ID', 'TEXT']]
    df_icu_notes = pd.merge(df_icu_notes, df_icu_adm_pt, how='inner', on=['ICUSTAY_ID'])
    print(df_icu_notes.dtypes)
    df_icu_notes = df_icu_notes[['ICUSTAY_ID', 'HADM_ID', 'SUBJECT_ID', 'INTIME', 'OUTTIME', 'ADMITTIME', 'DISCHTIME', 'DISCHARGE_LOCATION', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG', 'DOD', 'EXPIRE_FLAG', 'TEXT']]
    print('Total number of icustays: ' + str(len(df_icu_notes)))

    return df_icu_notes

def add_features(df_MASTER_DATA):
    """
    the function adds target labels to the dataset
    :param df_MASTER_DATA:
    :return:
    """
    print('\n Adding target variables...')

    df_MASTER_DATA['INTIME'] = df_MASTER_DATA['INTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['OUTTIME'] = df_MASTER_DATA['OUTTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['ADMITTIME'] = df_MASTER_DATA['ADMITTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['DISCHTIME'] = df_MASTER_DATA['DISCHTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['DEATHTIME'] = df_MASTER_DATA['DEATHTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['DOD'] = df_MASTER_DATA['DOD'].astype('datetime64[ns]')
    df_MASTER_DATA = df_MASTER_DATA.sort_values(['SUBJECT_ID', 'INTIME', 'OUTTIME'], ascending=[True, True, True])
    df_MASTER_DATA.reset_index(inplace=True, drop=True)

    # Add targetr column to show if readmitted within different timeframes
    df_MASTER_DATA = df_MASTER_DATA.assign(Mortality_30days=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(Mortality_InHospital=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_30days=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_Bounceback=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(Time_To_readmission=np.nan)
    df_MASTER_DATA = df_MASTER_DATA.assign(Time_To_death=np.nan)

    # total number of admissions
    num_adms = df_MASTER_DATA.shape[0]
    indexes_to_drop = []
    for idx in tqdm(range(0, num_adms)):
        # Drops icustay from cohort if the patient dies during the stay
        if df_MASTER_DATA.HOSPITAL_EXPIRE_FLAG[idx] and (df_MASTER_DATA.DEATHTIME[idx] <= df_MASTER_DATA.OUTTIME[idx] or
                                                  (df_MASTER_DATA.DISCHTIME[idx] <= df_MASTER_DATA.OUTTIME[idx] and df_MASTER_DATA.DISCHARGE_LOCATION[idx] == 'DEAD/EXPIRED')):
            indexes_to_drop.append(idx)
        # Calculates readmissions labels
        if idx > 0 and df_MASTER_DATA.SUBJECT_ID[idx] == df_MASTER_DATA.SUBJECT_ID[idx - 1]:
            # previous icu discharge time
            prev_outtime = df_MASTER_DATA.OUTTIME[idx - 1]
            # current icu admit time
            curr_intime = df_MASTER_DATA.INTIME[idx]

            readmit_time = curr_intime - prev_outtime
            df_MASTER_DATA.loc[idx - 1, 'Time_To_readmission'] = readmit_time.seconds / (3600 * 24) + readmit_time.days

            if readmit_time.days <= 30:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_30days'] = 1
                # Check bouncebacks
            if df_MASTER_DATA.HADM_ID[idx] == df_MASTER_DATA.HADM_ID[idx - 1]:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_Bounceback'] = 1

        # Checks in hospital mortality
        if df_MASTER_DATA.HOSPITAL_EXPIRE_FLAG[idx]:
            df_MASTER_DATA.loc[idx, 'Mortality_InHospital'] = 1
            # current icu discharge time
            outtime = df_MASTER_DATA.OUTTIME[idx]
            # current death time
            deathtime = df_MASTER_DATA.DEATHTIME[idx]
            time_to_death = deathtime - outtime
            df_MASTER_DATA.loc[idx, 'Time_To_death'] = time_to_death.seconds / (3600 * 24) + time_to_death.days
            if time_to_death.days <= 30:
                df_MASTER_DATA.loc[idx, 'Mortality_30days'] = 1

        # Checks out of hospital mortality
        if df_MASTER_DATA.EXPIRE_FLAG[idx]:
            # current icu discharge time
            outtime = df_MASTER_DATA.OUTTIME[idx]
            # current death time
            dod = df_MASTER_DATA.DOD[idx]
            time_to_death = dod - outtime
            df_MASTER_DATA.loc[idx, 'Time_To_death'] = time_to_death.seconds / (3600 * 24) + time_to_death.days
            if time_to_death.days <= 30:
                df_MASTER_DATA.loc[idx, 'Mortality_30days'] = 1
    df_MASTER_DATA.drop(df_MASTER_DATA.index[indexes_to_drop], inplace=True)
    print('Total stays: ' + str(len(df_MASTER_DATA)))
    print('30 day readmission: ' + str(df_MASTER_DATA['IsReadmitted_30days'].sum()))
    print('Bounceback readmission: ' + str(df_MASTER_DATA['IsReadmitted_Bounceback'].sum()))
    print('30 day mortality: ' + str(df_MASTER_DATA['Mortality_30days'].sum()))
    print('In hospital mortality: ' + str(df_MASTER_DATA['Mortality_InHospital'].sum()))
    return df_MASTER_DATA


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.config_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    local_dir = params.local_data

    # Preprocess notes for CNN
    preprocess_all_notes(params)

    # Create labeled data for CNN
    print('\nImporting data from preprocessed notes...')
    path_clean_notes = os.path.join(local_dir, 'df_MASTER_NOTES_ALL.csv')
    clean_notes_all = pd.read_csv(path_clean_notes)
    notes = get_notes(params, clean_notes_all)

    master_notes = add_features(notes)
    print(master_notes[:5]['TEXT'])
    datasetPath = os.path.join(local_dir, 'df_MASTER_DATA_ALL_LABELS.csv')
    master_notes.to_csv(datasetPath, index=False)

"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
import os
import pandas as pd
import gensim, logging
import argparse
from tqdm import tqdm
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='rootpath/resources/params.json', help="Directory containing params.json")

class MySentences(object):
    def __init__(self, df_notes):
        self.df_notes = df_notes

    def __iter__(self):
        for index, row in self.df_notes.iterrows():
            text = row['TEXT']
            yield text.split()

def train_embeddings(params, df_notes, folder):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = MySentences(df_notes)  #iterator
    print('Training model...')
    model = gensim.models.Word2Vec(sentences, min_count=5, workers=16, size=100)

    word_vectors = model.wv
    local_dir = params.local_data
    datasetPath = os.path.join(local_dir, folder, 'vectors.kv')
    word_vectors.save(datasetPath)

    local_dir = params.local_data
    datasetPath = os.path.join(local_dir, folder, 'w2v_model')
    model.save(datasetPath)

def get_starspace_notes(params, clean_notes, folder):
    print('Loading data...')
    local_dir = params.local_data
    data_dir = params.data_dir

    print('Concatenating text...')
    clean_notes = clean_notes.groupby('HADM_ID')['TEXT'].apply(lambda x: ' '.join(x))
    clean_notes = clean_notes.reset_index()

    print('Loading ICD Diagnosis codes...')
    datasetPath = os.path.join(data_dir, 'DIAGNOSES_ICD.csv.gz')
    df_diag = pd.read_csv(datasetPath)
    df_diag['ICD9_CODE'] = '__label__<diag>' + df_diag['ICD9_CODE'].astype(str)
    df_diag = df_diag.groupby('HADM_ID')['ICD9_CODE'].apply(lambda x: ' '.join(x))
    df_diag = df_diag.reset_index()

    print(df_diag.head())
    print(clean_notes.head())
    df_master = pd.merge(df_diag, clean_notes, how='inner', on=['HADM_ID'])

    print('Loading ICD Procedure codes...')
    datasetPath = os.path.join(data_dir, 'PROCEDURES_ICD.csv.gz')
    df_proc = pd.read_csv(datasetPath)
    df_proc['ICD9_CODE'] = '__label__<proc>' + df_proc['ICD9_CODE'].astype(str)
    df_proc = df_proc.groupby('HADM_ID')['ICD9_CODE'].apply(lambda x: ' '.join(x))
    df_proc = df_proc.reset_index()

    df_master = pd.merge(df_master, df_proc, how='left', on=['HADM_ID'])
    df_master["ICD9_BOTH"] = df_master["ICD9_CODE_x"].map(str) + ' ' + df_master["ICD9_CODE_y"].map(str)

    datasetPath = os.path.join(local_dir, folder, 'df_starspace.csv')
    df_master.to_csv(datasetPath, index=False)

    return df_master


def write_starspace_file(params, df_master, folder):
    local_dir = params.local_data
    filePath = os.path.join(local_dir, folder, 'starspace.txt')
    total = len(df_master.index)
    with tqdm(total=total) as pbar:
        with open(filePath, 'w') as f:
            for index, row in tqdm(df_master.iterrows()):
                pbar.update(1)
                text = [x for x in row['TEXT'].split()]
                both = [x for x in row['ICD9_BOTH'].split() if x not in ['Nan', 'NaN', 'nan', 'NAN']]
                print(' '.join(text) + ' ' + ' '.join(both), file=f)

if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.config_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    local_dir = params.local_data
    print('Loading NOTES...')
    datasetPath = os.path.join(local_dir, 'df_MASTER_NOTES_ALL.csv')
    df_notes = pd.read_csv(datasetPath)
    for i in range(5):
        datasetPath = os.path.join(local_dir, 'fold'+str(i), 'df_test_subjects.csv')
        df_test = pd.read_csv(datasetPath)
        df_temp_notes = df_notes[~(df_notes.SUBJECT_ID.isin(df_test.SUBJECT_ID))]
        # train word2vec ebeddings
        train_embeddings(params, df_temp_notes, 'fold'+str(i))

        # create files to train starspace embeddings
        df_starspace = get_starspace_notes(params, df_temp_notes, 'fold'+str(i))
        write_starspace_file(params, df_starspace, 'fold'+str(i))
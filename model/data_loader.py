import random
import numpy as np
import os
from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm
import torch

class DataLoader(object):
    """
    Handles the data. Stores the vocabulary and their mappings to indices.
    """

    def __init__(self, data_dir, params):
        """
        Loads vocabulary
        class.
        Args:
            data_dir: (string) directory containing the data
            params: (Params) hyperparameters of the training process.
        """

        # loading vocab
        self.params = params
        self.data_dir = data_dir

        self.vocab_w2v = {}
        self.weights_w2v = []
        self.index_to_word_w2v = {}

        self.vocab_sp = {}
        self.weights_sp = []
        self.index_to_word_sp = {}

        vocab_path = os.path.join(data_dir, 'fold' + str(params.fold), 'vectors.kv')
        word_vectors = KeyedVectors.load(vocab_path)

        for idx, key in enumerate(word_vectors.vocab):
            self.vocab_w2v[key] = idx
            self.weights_w2v.append(word_vectors[key])
            self.index_to_word_w2v[idx] = key
        self.vocab_w2v['<UNK>'] = idx + 1
        self.unk_ind_w2v = idx + 1
        self.index_to_word_w2v[idx+1] = '<UNK>'
        vec = np.random.randn(100)
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        self.weights_w2v.append(vec.astype(np.float32))
        self.vocab_w2v['<PAD>'] = idx + 2
        self.pad_ind_w2v = idx + 2
        self.index_to_word_w2v[idx + 2] = '<PAD>'
        self.weights_w2v.append(np.zeros(100, dtype=np.float32))
        print('generating sp vocab')
        vocab_path = os.path.join(data_dir, 'fold' + str(params.fold), 'diag_proc.tsv')
        with open(vocab_path, 'r') as inp:
            idx = 0
            for line in inp:
                words = line.strip().split()
                if '<diag>' in words[0] or '<proc>' in words[0]:
                    continue
                self.vocab_sp[words[0]] = idx
                self.index_to_word_sp[idx] = words[0]
                idx += 1
                vec = np.array(list(map(float, words[1:])), dtype=np.float32)
                # vec = vec / float(np.linalg.norm(vec) + 1e-6)

        self.vocab_sp['<UNK>'] = idx
        self.unk_ind_sp = idx
        self.index_to_word_sp[idx] = '<UNK>'
        vec = np.random.randn(300)
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        self.weights_sp.append(vec.astype(np.float32))
        self.vocab_sp['<PAD>'] = idx + 1
        self.pad_ind_sp = idx + 1
        self.index_to_word_sp[idx + 1] = '<PAD>'
        self.weights_sp.append(np.zeros(300, dtype=np.float32))

        self.weights_w2v = np.stack(self.weights_w2v, axis=0)
        self.weights_sp = np.stack(self.weights_sp, axis=0)

        # adding dataset parameters to param (e.g. vocab size, )
        params.vocab_size_w2v = len(self.vocab_w2v)
        params.vocab_size_sp = len(self.vocab_sp)

    def pad_or_truncate(self, note, pad_ind):
        if (len(note) > self.params.doc_length):
            return note[-self.params.doc_length:], [0.0]*self.params.doc_length
        else:
            attn_mask = [0.0] * len(note) + [float('-inf')] * (self.params.doc_length - len(note))
            note.extend([pad_ind] * (self.params.doc_length - len(note)))
            return note, attn_mask

    def load_notes_labels(self, file, d):
        """
        Loads notes and labels and stores them in the provided dict d.
        Args:
            file: (string) file with sentences with notes
            d: (dict) a dictionary in which the loaded data is stored
        """

        df_notes = pd.read_csv(file)

        if (self.params.debug):
            df_notes = df_notes.head(self.params.batch_size * 2)

        total = len(df_notes.index)
        notes_w2v = []
        w2v_attn_mask = []
        notes_sp = []
        sp_attn_mask = []
        labels = []
        ids = []
        print(df_notes.columns)
        assert self.params.task in df_notes.columns
        with tqdm(total=total) as pbar:
            for index, row in tqdm(df_notes.iterrows()):
                pbar.update(1)
                text = row['TEXT']
                # print(text)
                note_w2v = [self.vocab_w2v[token] if token in self.vocab_w2v else self.vocab_w2v['<UNK>'] for token in text.split()]
                note_sp = [self.vocab_sp[token] if token in self.vocab_sp else self.vocab_sp['<UNK>'] for token in text.split()]
                label = [row[self.params.task]]
                id = row['ICUSTAY_ID']

                note_w2v_ind, note_w2v_mask = self.pad_or_truncate(note_w2v, self.pad_ind_w2v)
                notes_w2v.append(note_w2v_ind)
                w2v_attn_mask.append(note_w2v_mask)
                note_sp_ind, note_sp_mask = self.pad_or_truncate(note_sp, self.pad_ind_sp)
                notes_sp.append(note_sp_ind)
                sp_attn_mask.append(note_sp_mask)
                labels.append(label)
                ids.append(id)

        # checks to ensure there is a label for each note
        assert len(labels) == len(notes_w2v)
        assert len(notes_w2v) == len(notes_sp)
        assert len(notes_w2v) == len(w2v_attn_mask)
        assert len(sp_attn_mask) == len(notes_sp)
        assert len(notes_sp) == len(ids)

        # storing data in dict d

        d['data_w2v'] = notes_w2v
        d['mask_w2v'] = w2v_attn_mask
        d['data_sp'] = notes_sp
        d['mask_sp'] = sp_attn_mask
        d['labels'] = labels
        d['size'] = len(notes_w2v)
        d['ids'] = ids

    def load_data(self, splits, data_dir):
        """
        Loads the data for specified splits data_dir.
        Args:
            splits: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset
        Returns:
            data: (dict) contains the data with labels for each type in types
        """
        data = {}

        for split in ['train', 'val', 'test']:
            if split in splits:
                file = os.path.join(data_dir, 'fold' + str(self.params.fold), 'df_' + split + '_subjects.csv')
                data[split] = {}
                self.load_notes_labels(file, data[split])
        return data

    def data_iterator(self, data, params, shuffle=False):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.
        Args:
            data: (dict) contains data
            params: (Params) hyperparameters of the training process
            shuffle: (bool) whether the data should be shuffled
        Yields:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data

        order = list(range(data['size']))
        if shuffle:
            random.shuffle(order)

        # one pass over data
        for i in range((data['size']) // params.batch_size):
            # fetch sentences and tags
            batch_notes_w2v = np.array(
                [data['data_w2v'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_notes_sp = np.array(
                [data['data_sp'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_w2v_mask = np.array(
                [data['mask_w2v'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_sp_mask = np.array(
                [data['mask_sp'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_tags = np.array(
                [data['labels'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])

            batch_ids = [data['ids'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]]

            # since all data are indices, we convert them to torch LongTensors
            batch_data_w2v, batch_data_sp, batch_labels = torch.tensor(batch_notes_w2v, dtype=torch.long, device='cuda'), torch.tensor(
                batch_notes_sp, dtype=torch.long, device='cuda'), torch.tensor(batch_tags, dtype=torch.float, device='cuda')

            if 'attn' in self.params.model:
                batch_w2v_mask, batch_sp_mask = torch.tensor(batch_w2v_mask, dtype=torch.float, device='cuda'), torch.tensor(batch_sp_mask, dtype=torch.float, device='cuda')
                yield [batch_data_w2v, batch_w2v_mask], [batch_data_sp, batch_sp_mask], batch_labels, batch_ids
            else:
                yield batch_data_w2v, batch_data_sp, batch_labels, batch_ids

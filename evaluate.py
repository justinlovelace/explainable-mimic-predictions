"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.models as models
from model.data_loader import DataLoader
import torch.nn as nn
import pandas as pd
from tqdm import trange
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='rootpath/resources/params.json', help="Directory containing params.json")

def evaluate_attn(model, loss_fn, data_iterator, params, num_steps, data_loader, model_dir):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        data_loader: data_loader that contains index to word mappings
        model_dir: model directory where attention rankings will be saved
    """
    # set model to evaluation mode

    model.eval()

    # summary for current eval loop
    master_list = []

    # compute metrics over the dataset

    running_metrics = utils.TestMetrics()
    # Use tqdm for progress bar
    with torch.no_grad():
        t = trange(num_steps)
        for _ in t:
            # fetch the next evaluation batch
            train_batch_w2v, train_batch_sp, labels_batch, ids = next(data_iterator)
            if 'w2v' in params.emb:
                output_batch, attn_weights_w2v = model(train_batch_w2v, interpret=True)
                batch_word_indexes = train_batch_w2v[0].tolist()
                batch_text = []
                for word_indexes in batch_word_indexes:
                    unigrams, bigrams, trigrams = [], [], []
                    for ind in range(len(word_indexes)):
                        if ind < 2:
                            pre_context = data_loader.index_to_word_w2v[word_indexes[ind - 1]]
                        elif ind < 1:
                            pre_context = ''
                        else:
                            pre_context = data_loader.index_to_word_w2v[word_indexes[ind - 2]] + ' ' + \
                                          data_loader.index_to_word_w2v[word_indexes[ind - 1]]
                        if ind + 4 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 3]])
                            trigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 3]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 4]])
                        elif ind + 3 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 3]])
                            trigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 3]])
                        elif ind + 2 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                            trigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 2]] + '] ')
                        elif ind + 1 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ')
                        else:
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ')

                    batch_text.append(
                        unigrams + bigrams + ['<CONV_PAD>'] + trigrams + ['<CONV_PAD>'] + ['<CONV_PAD>'])
                output_list = output_batch.tolist()
                attn_weights_list = [x.tolist() for x in attn_weights_w2v]
                labels_batch_list = labels_batch.tolist()
                assert len(ids) == len(batch_text)
                assert len(ids) == len(labels_batch_list)
                assert len(ids) == len(output_list)
                assert len(ids) == len(attn_weights_list[0])
                for head in range(len(attn_weights_list)):
                    for index in range(len(ids)):
                        temp_list = []
                        temp_list.append(ids[index])
                        temp_list.append('w2v')
                        temp_list.append(head)
                        temp_list.append(labels_batch_list[index][0])
                        temp_list.append(output_list[index][0])
                        attn_words = list(zip(attn_weights_list[head][index], batch_text[index]))
                        attn_words.sort(reverse=True)
                        new_attn_words = [x for t in attn_words[:50] for x in t]
                        temp_list.extend(new_attn_words)
                        master_list.append(temp_list)
            elif 'sp' in params.emb:
                output_batch, attn_weights_sp = model(train_batch_sp, interpret=True)
                batch_word_indexes = train_batch_sp[0].tolist()
                batch_text = []
                for word_indexes in batch_word_indexes:
                    unigrams, bigrams, trigrams = [], [], []
                    for ind in range(len(word_indexes)):
                        if ind < 2:
                            pre_context = data_loader.index_to_word_sp[word_indexes[ind - 1]]
                        elif ind < 1:
                            pre_context = ''
                        else:
                            pre_context = data_loader.index_to_word_sp[word_indexes[ind - 2]] + ' ' + \
                                          data_loader.index_to_word_sp[word_indexes[ind - 1]]

                        if ind + 4 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 3]])
                            trigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 3]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 4]])
                        elif ind + 3 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 3]])
                            trigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 3]])
                        elif ind + 2 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]])
                            trigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 2]] + '] ')
                        elif ind + 1 < len(word_indexes):
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + '] ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]])
                            bigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + ' ' +
                                data_loader.index_to_word_sp[word_indexes[ind + 1]] + '] ')
                        else:
                            unigrams.append(
                                pre_context + ' [' + data_loader.index_to_word_sp[word_indexes[ind]] + '] ')

                    batch_text.append(unigrams + bigrams + trigrams)
                output_list = output_batch.tolist()
                attn_weights_list = [x.tolist() for x in attn_weights_sp]
                labels_batch_list = labels_batch.tolist()
                assert len(ids) == len(batch_text)
                assert len(ids) == len(labels_batch_list)
                assert len(ids) == len(output_list)
                assert len(ids) == len(attn_weights_list[0])
                for head in range(len(attn_weights_list)):
                    for index in range(len(ids)):
                        temp_list = []
                        temp_list.append(ids[index])
                        temp_list.append(head)
                        temp_list.append('sp300')
                        temp_list.append(labels_batch_list[index][0])
                        temp_list.append(output_list[index][0])
                        attn_words = list(zip(attn_weights_list[head][index], batch_text[index]))
                        attn_words.sort(reverse=True)
                        new_attn_words = [x for t in attn_words[:50] for x in t]

                        temp_list.extend(new_attn_words)
                        master_list.append(temp_list)
                output_batch = model(train_batch_sp)
            loss_fn(output_batch, labels_batch)
            running_metrics.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())

    df_attn_scores = pd.DataFrame(master_list, columns=["ICUSTAY_ID", 'head', 'embedding', params.task + "_label", params.task + "_prediction"] + [
        'attn_' + str(i // 2) if i % 2 == 0 else 'words_' + str(i // 2) for i in range(100)])
    print(df_attn_scores.dtypes)
    df_attn_scores.sort_values(by=[params.task + "_prediction"], ascending=False, inplace=True)
    print(df_attn_scores.head(5))
    datasetPath = os.path.join(model_dir, 'df_attn.csv')
    df_attn_scores.to_csv(datasetPath, index=False)
    logging.info('AUCROC' + str(running_metrics()))
    metrics = running_metrics()
    return {'AUCROC': metrics[0], "AUCPR": metrics[1]}


def evaluate(model, loss_fn, data_iterator, params, num_steps):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode

    model.eval()
    # compute metrics over the dataset
    running_auc = utils.OutputAUC()
    running_metrics = utils.TestMetrics()
    # Use tqdm for progress bar
    with torch.no_grad():
        t = trange(num_steps)
        for _ in t:
            # fetch the next evaluation batch
            train_batch_w2v, train_batch_sp, labels_batch, ids = next(data_iterator)
            if 'w2v' in params.emb:
                output_batch = model(train_batch_w2v)
            elif 'sp' in params.emb:
                output_batch = model(train_batch_sp)
            else:
                output_batch = model(train_batch_w2v, train_batch_sp)
            loss_fn(output_batch, labels_batch)
            running_auc.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            running_metrics.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())

    logging.info('AUCROC' + str(running_auc()))
    logging.info('METRICS' + str(running_metrics()))
    metrics = running_metrics()
    return {'AUCROC': metrics[0], "AUCPR": metrics[1]}



if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    torch.set_num_threads(4)
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.config_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    np.random.seed(230)
    np.random.seed(230)
    if params.cuda:
        torch.cuda.manual_seed_all(230)

    # Set the logger
    job_name = "emb{}_lr{}_k{}_bs{}_f{}_dr{}_{}_{}".format(params.emb, params.learning_rate, params.kernels,
                                                           params.batch_size, params.filters, params.dropout,
                                                           params.model, params.task)
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(params.save_path, 'fold' + str(params.fold), job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(params.data_dir, params)

    data = data_loader.load_data(['test'], params.data_dir)
    test_data = data['test']

    # specify the train and val dataset sizes
    params.test_size = test_data['size']

    logging.info("- done.")

    # Define the model and optimizer
    device = torch.device("cuda:0" if params.cuda else sys.exit("gpu unavailable"))
    if params.model == "cnn_text":
        if params.emb == 'w2v':
            model = models.CNN_Text(data_loader.weights_w2v, params)
        elif 'sp' in params.emb:
            model = models.CNN_Text(data_loader.weights_sp, params)
    elif params.model == "cnn_text_attn":
        if 'w2v' in params.emb:
                model = models.CNN_Text_Attn(data_loader.weights_w2v, params)
        elif 'sp' in params.emb:
            model = models.CNN_Text_Attn(data_loader.weights_sp, params)

    print(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # fetch loss function and metrics
    loss_fn = torch.nn.BCELoss()
    metrics = models.metrics

    # Train the model
    logging.info("Starting evaluation ")
    utils.load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model, parallel=False)
    num_steps = test_data['size'] // params.batch_size
    test_data_iterator = data_loader.data_iterator(test_data, params, shuffle=False)
    if 'attn' in params.model:
        test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)

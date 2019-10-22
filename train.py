"""Train the model"""

import argparse
import logging
import os


import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import model.models as models
from model.data_loader import DataLoader
from evaluate import evaluate
import torch.nn as nn
import random
import sys
import utils



parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='rootpath/resources/params.json', help="Directory containing params.json")

def train(model, optimizer, loss_fn, data_iterator, params, num_steps):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    running_auc = utils.OutputAUC()
    for i in t:
        # fetch the next training batch
        train_batch_w2v, train_batch_sp, labels_batch, ids = next(data_iterator)
        if 'w2v' in params.emb :
            output_batch = model(train_batch_w2v)
        elif 'sp' in params.emb:
            output_batch = model(train_batch_sp)
        else:
            output_batch = model(train_batch_w2v, train_batch_sp)
        loss = loss_fn(output_batch, labels_batch)
        loss = loss / params.grad_acc  # Normalize our loss (if averaged)

        running_auc.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
        loss.backward()

        if i % params.grad_acc == 0:
            # performs updates using calculated gradients
            optimizer.step()
            # clear previous gradients
            optimizer.zero_grad()

        # update the average loss
        loss_avg.update(float(loss.data.item()))
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    logging.info('Train AUC: '+str(running_auc()))
    return loss_avg()



def train_and_evaluate(model, train_data, val_data, test_data, optimizer, loss_fn, params, model_dir, data_loader):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    best_val_auc = 0
    num_epochs = 0
    epoch = 0

    while num_epochs < params.patience:
        # Run one epoch
        logging.info(model_dir)
        logging.info("Epoch {}".format(epoch + 1))
        logging.info("Patience: {}".format(num_epochs + 1))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = train_data['size'] // params.batch_size
        train_data_iterator = data_loader.data_iterator(train_data, params, shuffle=True)
        train(model, optimizer, loss_fn, train_data_iterator, params, num_steps)

        # Evaluate for one epoch on validation set
        num_steps = val_data['size'] // params.batch_size
        val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, params, num_steps)

        is_best = np.mean(val_metrics['AUCROC']) >= best_val_auc
        model_to_save = model
        state_dict = model_to_save.state_dict()
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': state_dict,
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best auc")
            best_val_auc = np.mean(val_metrics['AUCROC'])

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            num_epochs = -1

        num_epochs += 1
        epoch += 1

        # Save latest val metrics in a json file in the model directory
        logging.info("Current AUC: " + str(val_metrics['AUCROC']) + '\n' + str(np.mean(val_metrics['AUCROC'])))
        logging.info("Best AUC: " + str(best_val_auc))
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    if params.test_mode:
        utils.load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model)
        num_steps = test_data['size'] // params.batch_size
        test_data_iterator = data_loader.data_iterator(test_data, params, shuffle=False)
        test_metrics = evaluate(model, loss_fn, test_data_iterator, params, num_steps)

    logging.info("TEST METRICS: " + str(test_metrics['AUCROC']))
    logging.info("MEAN TEST METRICS: " + str(np.mean(test_metrics['AUCROC'])))





if __name__ == '__main__':
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
    random.seed(230)
    np.random.seed(230)
    if params.cuda:
        torch.cuda.manual_seed_all(230)

    job_name = "emb{}_lr{}_k{}_bs{}_f{}_dr{}_{}_{}".format(params.emb, params.learning_rate, params.kernels,
                                                           params.batch_size, params.filters, params.dropout,
                                                           params.model, params.task)
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(params.save_path, 'fold' + str(params.fold), job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(params.local_data, params)

    data = data_loader.load_data(['train', 'val', 'test'], params.local_data)
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Define the model and optimizer
    device = torch.device("cuda:0" if params.cuda else sys.exit("gpu unavailable"))
    if params.model == "cnn_text":
        if params.emb=='w2v':
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
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = torch.nn.BCELoss()

    # Train the model
    logging.info("Starting training ")
    train_and_evaluate(model, train_data, val_data, test_data, optimizer, loss_fn, params, model_dir,
                       data_loader)

    print(model_dir)

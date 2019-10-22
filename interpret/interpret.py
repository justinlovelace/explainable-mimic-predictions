"""Evaluates the model"""

import argparse
import os

import numpy as np
import torch
import utils
import pandas as pd
import math
from wordcloud import WordCloud
import re

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='/home/ugrads/j/justinlovelace/MIMIC/ICU_Readmit/resources', help="Directory containing params.json")

def create_global_wordcloud(model_dir, save_dir, high_risk=True, ratio=0.05):
    img_directory = os.path.join(save_dir, 'visualizations', 'global')
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
    df_attn_weights = pd.read_csv(os.path.join(model_dir, 'df_attn.csv'))
    datasetPath = os.path.join(save_dir, 'df_master.csv')
    # df_attn_weights.to_csv(datasetPath, index=False)
    target_column = [x for x in df_attn_weights.columns if 'prediction' in x]
    num_rows = math.floor(len(df_attn_weights)*ratio)
    if high_risk:
        df_attn_weights = df_attn_weights.nlargest(n=num_rows, columns=target_column, keep='first')
    else:
        df_attn_weights = df_attn_weights.nsmallest(n=num_rows, columns=target_column, keep='first')

    freq_dict = {}
    count_dict = {}
    weight_dict = {}
    for index, row in df_attn_weights.iterrows():
        for i in range(3):
            word_window = row['words_'+str(i)]
            word = re.search('\[(.*)\]', word_window)
            if word:
                word = word.group(1)
            else:
                continue
            if word in freq_dict:
                freq_dict[word] += row['attn_'+str(i)]
                count_dict[word] += 1
                weight_dict[word].append(row['attn_'+str(i)])
            else:
                freq_dict[word] = row['attn_' + str(i)]
                count_dict[word] = 1
                weight_dict[word] = [row['attn_' + str(i)]]

    master_list=[]
    for word in freq_dict:
        master_list.append([word, np.mean(weight_dict[word]), count_dict[word], freq_dict[word]])


    df_word_cloud_data = pd.DataFrame(master_list, columns=["Word", 'Avg_weight', 'Frequency', 'Avg_weight X Frequency'])
    print(df_word_cloud_data.dtypes)
    df_word_cloud_data.sort_values(by=['Avg_weight X Frequency'], ascending=False, inplace=True)
    print(df_word_cloud_data.head(5))
    datasetPath = os.path.join(save_dir, 'df_' + 'global_high_risk.csv' if high_risk else 'global_low_risk.csv')
    # df_word_cloud_data.to_csv(datasetPath, index=False)

    # lower max_font_size
    wordcloud = WordCloud(relative_scaling=0.5, prefer_horizontal=1, max_font_size=512, random_state=0, width=1920, height=1080, max_words=20, background_color='white', color_func=lambda *args, **kwargs: "black")
    wordcloud.generate_from_frequencies(freq_dict)

    imgname = 'global_high_risk.png' if high_risk else 'global_low_risk.png'
    wordcloud.to_file(os.path.join(img_directory, imgname))

def create_individual_wordclouds(model_dir, save_dir, high_risk=True, ratio=0.005):
    img_directory = os.path.join(save_dir, 'visualizations', 'individual')
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
    df_attn_weights = pd.read_csv(os.path.join(model_dir, 'df_attn.csv'))
    target_column = next(x for x in df_attn_weights.columns if 'prediction' in x)
    num_rows = math.floor(len(df_attn_weights)*ratio)
    if high_risk:
        df_attn_weights = df_attn_weights.nlargest(n=num_rows, columns=target_column, keep='first')
    else:
        df_attn_weights = df_attn_weights.nsmallest(n=num_rows, columns=target_column, keep='first')


    for index, row in df_attn_weights.iterrows():
        freq_dict = {}
        for i in range(50):
            word = row['words_'+str(i)]
            if word in freq_dict:
                freq_dict[word] += row['attn_'+str(i)]
            else:
                freq_dict[word] = row['attn_' + str(i)]
        wordcloud = WordCloud(relative_scaling=0.5, prefer_horizontal=1, max_font_size=512, random_state=0, width=1920,
                              height=1080, max_words=10, background_color='white', color_func=lambda *args, **kwargs: "black")
        wordcloud.generate_from_frequencies(freq_dict)
        imgname = "ICUSTAY_ID_{}_prediction_{}_label{}_risk_{}.png".format(row['ICUSTAY_ID'], row[target_column], row[next(x for x in df_attn_weights.columns if 'label' in x)], 'high' if high_risk else 'low')
        wordcloud.to_file(os.path.join(img_directory, imgname))
        # lower max_font_size


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

    save_dir = params.visualizations_path
    # use GPU if available
    params.cuda = torch.cuda.is_available()
    # Set the logger
    job_name = "emb{}_lr{}_k{}_bs{}_f{}_dr{}_{}_{}".format(params.emb, params.learning_rate, params.kernels,
                                                           params.batch_size, params.filters, params.dropout,
                                                           params.model, params.task)
    # Create a new folder in parent_dir with unique_name "job_name"
    print(job_name)
    model_dir = os.path.join(params.save_path, 'fold' + str(params.fold), job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    task_save_dir = os.path.join(save_dir, params.task, params.emb, 'fold' + str(params.fold))

    print('Creating high risk individual wordclouds...')
    create_individual_wordclouds(model_dir, task_save_dir)
    print('Creating low risk individual wordclouds...')
    create_individual_wordclouds(model_dir, task_save_dir, high_risk=False)
    print('Creating high risk global wordclouds...')
    create_global_wordcloud(model_dir, task_save_dir)
    print('Creating high risk global wordclouds...')
    create_global_wordcloud(model_dir, task_save_dir, high_risk=False)

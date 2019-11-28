# Explainable Prediction of Adverse Outcomes Using Clinical Notes

Code for the paper [Explainable Prediction of Adverse Outcomes Using Clinical Notes](https://arxiv.org/abs/1910.14095).

Modify all of the file paths in `resources/params.json` according to your local file structure.

## Processing Data 
Preprocess the notes and extract the target labels.

`python preprocess/extract_labels.py` 

Split the data into 5 splits to use for 5-fold cross validation.

`python preprocess/cross_val_splits.py` 

## Training Embeddings
In this work we use two different embedding techniques, Word2Vec and Starspace embeddings. Running word_embedding.py generates the Word2Vec embeddings and the files needed to generate the Starspace embeddings. Instructions for installing and training Starspace embeddings can be found in their [repo](https://github.com/facebookresearch/StarSpace). An example script with the settings we used to train the starspace embeddings can be found at preprocess/train_starspace.py.

`python preprocess/word_embedding.py` 

## Training Models
The training hyperparameters can be modified in `resources/params.json`. The models can be trained using the following command.

`python train.py` 

## Acknowledgments
We adapt our preprocessing code from that released by [DeepEHR](https://github.com/NYUMedML/DeepEHR).

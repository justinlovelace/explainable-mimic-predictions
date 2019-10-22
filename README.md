# Explainable Prediction of Adverse Outcomes Using Clinical Notes

Modify all of the file paths in `resources/params.json` according to your local file structure.

## Processing Data   
`python preprocess/extract_labels.py` 

`python preprocess/cross_val_splits.py` 

## Training Embeddings

`python preprocess/word_embedding.py` 

### Training Models
The hyperparameters can be modified in `resources/params.json`.

`python train.py` 
./starspace train -trainFile /train/file/path.txt -model /model/file/path.text -ngrams 1 -adagrad True -thread 20 -dropoutRHS 0.8 -dim 300 -lr 0.01 -epoch 20 -margin 0.05 -verbose true -loss hinge -initRandSd 0.01 -trainMode 0 -similarity "cosine" -negSearchLimit 100 -minCount 5 -maxNegSamples 100 -dropoutLHS 0.0 -minCountLabel 5 -fileFormat fastText

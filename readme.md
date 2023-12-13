# LSTM, Transformer and BERT tagger for NER problems

## Introduction
In this project, I trained a LSTM, a Transformer and a BERT model to solve the NER tagging problem on the dataset Conll2003.

## How to run
First, install the following packages:
+ `pytorch`
+ `seqeval`
+ `transformers`
+ `tensorboard`
You can run `pip install pytorch seqeval transformers tensorboard` to install them all. You can also use `conda install` command if you are in an activated anaconda environment, this method is more recommended.

Run `optimize.py`, this will automatically find the best parameters for LSTM and Transformer. Then modify `model_config.py` to set the parameters you find that works best or just some numbers you like. If you only want to train one model: you shoud run `train_xxx.py`. Replace `xxx` with the model name you like. Otherwise, run `main.py`. This will train the models. You can run `tensorboard` to monitor the training progress. The command is `tensorboard --logdir runs`. After training, run `write_lstm_and_transformer_out.py` and `write_bert_out.py` to write the outputs.

## Directory structure
### Sub-directories
This project includes 5 sub-directories.
+ **conll 2003** The dataset directory. It is empty when I submitted it.
+ **logs** The directory that stores all the outputs of each step. I did most of the experiments on the Universty's GPU farm, and used `nohup python xxx.py > xxx.log &` command to output all the logs. I didn't use logging package because `nohup` command is pretty handful. Also, I need to mention that I only put the last version of the logs in the directory, because I believe no one wants to see my countless errors during the experiment.
+ **models** The directory that stores all the trained models. It is empty when I submitted it.
+  **out** The directory that includes the results of the predictions. The format of each file is exactly the same as `test.txt`, I of changed the last element of each line to the predictions.
+ **runs** The directory that stores the tensorboard files, recording loss on the training set, f1, precision and recall scores on the valid set each epoch.

### Python scripts
I will introduce these scripts in logical order rather than filename order. 
+ **datapreprocess.py** This script loads data from the dataset and constructs the word map and label map. It also batches the data. This is for LSTM and Transformer.
+ **seqProc.py** This script provides the tokenizer and decoder funtion for LSTM and Transformer.
+ **dataset.py** This script provides the dataloader for BERT. I should have unified this with LSTM and Transformer, but due to time constraints I couldn't make it. Any way, both dataloaders run.
+ **models.py** The LSTMTagger, TransformerTagger and BERTTagger source codes written in pytorch. I will explain the details in the report.
+ **optimize.py** Using `optuna` package to search for the best parameters, including layer numbers, neuron numbers, loss weights etc. Both for LSTM and Transformer.
+ **model_config.py** Using `pickle` to save the model is easy yet not graceful. I saved the models with 
`torch.save(model.state_dict(), f"models/{model_type}_{epoch}.pt")`
However, this makes it model has to be created before dict is loaded. Modifying all the files is not a good experience. So I stored the parameters in the config file so that only modifying the config file is enough for every step of the experiment.
+ **evaluate.py** Provides a general evaluate function for different models and even for different `seqeval` functions. It is super cool.
+ **train_lstm.py** Train LSTM model or provide `train_lstm()` function.
+ **train_transformer.py** Train Transformer model or provide `train_transformer()` function.
+ **train_bert.py** Train BERT model or provide `train_bert()` function.
+ **main.py** Train all the models.
+ **write_lstm_and_transformer_out.py** Predict LSTM and Transformer on the test set, then write the output files.
+ **write_bert_out.py** Predict BERT on the test set and write the outputs.
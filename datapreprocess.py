

import torch
from seqProc import *
DEVICE = torch.device("cuda:0")


def load_data(batch_size=512):
  dirprefix = "."
  datasetpath = dirprefix+"/conll2003/"
  training_data = []
  with open(datasetpath+"train.txt","r") as trainf:
    words = []
    tags = []
    for line in trainf.readlines():
      if line=="\n":
        training_data.append((words,tags))
        words=[]
        tags=[]
      else:
        args = line.strip().split()
        words.append(args[0])
        tags.append(args[-1])
        
        
  test_data = []
  with open(datasetpath+"test.txt","r") as trainf:
    words = []
    tags = []
    for line in trainf.readlines():
      if line=="\n":
        test_data.append((words,tags))
        words=[]
        tags=[]
      else:
        args = line.strip().split()
        words.append(args[0])
        tags.append(args[-1])
        
        
  valid_data = []
  with open(datasetpath+"valid.txt","r") as trainf:
    words = []
    tags = []
    for line in trainf.readlines():
      if line=="\n":
        valid_data.append((words,tags))
        words=[]
        tags=[]
      else:
        args = line.strip().split()
        words.append(args[0])
        tags.append(args[-1])
        
        
        
  word_to_ix = {}
  tag_to_ix = {}
  # For each words-list (sentence) and tags-list in each tuple of training_data
  max_length = 0
  for sent, tags in training_data:
      max_length = max(max_length,len(sent))
      for word in sent:
          if word not in word_to_ix:  # word has not been assigned an index yet
              word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
      for tag in tags:
          if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
  for sent, tags in test_data:
      max_length = max(max_length,len(sent))
      for word in sent:
          if word not in word_to_ix:  # word has not been assigned an index yet
              word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
      for tag in tags:
          if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
  for sent, tags in valid_data:
      max_length = max(max_length,len(sent))
      for word in sent:
          if word not in word_to_ix:  # word has not been assigned an index yet
              word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
      for tag in tags:
          if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

  word_to_ix["<PAD>"] = 9
  tag_to_ix["<PAD>"] = 9
  ix_to_tag = {v: k for k, v in tag_to_ix.items()}



  length_train = len(training_data)
  all_data = training_data+test_data+valid_data
  # all_data = training_data
  padded_sents = torch.nn.utils.rnn.pad_sequence([prepare_sequence(sent,word_to_ix) for sent,tags in all_data],batch_first=True).to(DEVICE)
  padded_tags = torch.nn.utils.rnn.pad_sequence([prepare_sequence(tags,tag_to_ix) for sent,tags in all_data],batch_first=True).to(DEVICE)
  padded_sents_train = padded_sents[:length_train]
  padded_tags_train = padded_tags[:length_train]

  batched_X = padded_sents_train.split(batch_size)
  batched_Y = padded_tags_train.split(batch_size)
  return batched_X,batched_Y,valid_data,test_data,word_to_ix,tag_to_ix,ix_to_tag
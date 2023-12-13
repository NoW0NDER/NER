import torch
import torch.nn.functional as F
from seqeval.metrics import accuracy_score, f1_score, classification_report
from torch.utils import data
from tqdm import trange, tqdm
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule

from dataset import CoNLL2003DataSet, CoNLL2003Processor, vocab
from models import BertTagger, LSTMTagger, TransFormerTagger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    


batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CoNLL2003Processor()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

label_list = processor.get_labels()
tags_vals = label_list
label_map = {label: i for i, label in enumerate(label_list)}
train_data = processor.get_train()
val_data = processor.get_valid()
test_data = processor.get_test()
training_set = CoNLL2003DataSet(train_data, tokenizer, label_map, max_len=128)
eval_set = CoNLL2003DataSet(val_data, tokenizer, label_map, max_len=256)
test_set = CoNLL2003DataSet(test_data, tokenizer, label_map, max_len=256)

train_iter = data.DataLoader(dataset=training_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=12)
eval_iter = data.DataLoader(dataset=eval_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=12)
test_iter = data.DataLoader(dataset=test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=12)






model = BertTagger.from_pretrained('bert-base-cased', num_labels=len(label_map))
model.load_state_dict(torch.load('models/bert.pt'))
model.to(device)
nb_eval_steps = 0
predictions, true_labels = [], []
input_ids = []
for batch in tqdm(test_iter):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
    input_ids.extend(b_input_ids)
    with torch.no_grad():
        tmp_eval_loss, logits, reduced_labels = model(b_input_ids,
                                                        token_type_ids=b_token_type_ids,
                                                        attention_mask=b_input_mask,
                                                        labels=b_labels,
                                                        label_masks=b_label_masks)
    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()
    reduced_labels = reduced_labels.to('cpu').numpy()
    labels_to_append = []
    predictions_to_append = []
    for prediction, r_label in zip(logits, reduced_labels):
        preds = []
        labels = []
        for pred, lab in zip(prediction, r_label):
            if lab.item() == -1:  # masked label; -1 means do not collect this label
                continue
            preds.append(pred)
            labels.append(lab)
        predictions_to_append.append(preds)
        labels_to_append.append(labels)
    predictions.extend(predictions_to_append)
    true_labels.extend(labels_to_append)


import numpy as np
def decode_input(l):
    res = []
    for line in l:
        if 0 in line:
            line = line[:np.argmin(line)]
        decoded = tokenizer.convert_ids_to_tokens(line)
        # print(decoded)
        res.append(decoded)
    return res

def decode(l):
    res = []
    for line in l:
        res.append([tags_vals[i] for i in line])
    return res    
pred_tags = decode(predictions)
test_tags = decode(true_labels)
input_ids = torch.stack(input_ids).cpu().numpy()
input_sents = decode_input(input_ids)
# for pred,test,sent in zip(pred_tags,test_tags,input_sents):
#     print(pred,len(pred))
#     print(test,len(test))
#     print(sent,len(sent))
#     print("-----")
def write_bert():
    p_1d = [item for sublist in pred_tags for item in sublist]
    t_1d = [item for sublist in test_tags for item in sublist]
    print(len(p_1d),len(t_1d))
    idx = 0
    with open("conll2003/test.txt","r") as f1:
        with open("out/bert_text.txt","w") as f2:
            for line in f1:
                if line!="\n" and line!="":
                    words = line.strip().split()
                    if words[0].startswith("-DOCSTART-"):
                        f2.write(line)
                        continue
                    if t_1d[idx] == words[-1]:
                        words[-1] = p_1d[idx]+"\n"
                        idx+=1
                    else:
                        words[-1] = "*******\n"
                        idx+=1
                        print(idx)
                        print("line"+line)
                        print("t_1d[idx]"+t_1d[idx])
                        print("words[-1]"+words[-1])
                        print("error")
                        # break
                    f2.write(" ".join(words))
                else:
                    f2.write("\n")

# pred_tags = [tags_vals[p_ii] for p in predictions for p_i in p for p_ii in p_i]
# valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]


# write_bert()


# print(pred_tags[:10], test_tags[:10])
# print("Seq eval accuracy: {}".format(accuracy_score(test_tags, pred_tags)))
# print("F1-Score: {}".format(f1_score(test_tags, pred_tags)))
# print("Classification report: -- ")
# print(classification_report(test_tags, pred_tags))

def get_all_input_sents():
    with open("conll2003/test.txt","r") as f:
        sents = []
        sent = []
        for line in f:
            if line.startswith("-DOCSTART-"):
                continue
            if line=="\n":
                if sent == []:
                    continue
                sents.append(sent)
                sent = []
            else:
                sent.append(line.strip().split()[0])
    return sents
sents = get_all_input_sents()
for sent in sents:
    if len(sent)<=2:
        print(sent)
        break
print(len(sents))
print(len(input_ids))
print(len(pred_tags))
with open("logs/compare.txt","w") as f:
    for i in range(len(sents)):
        f.write(" ".join(input_sents[i])+"\n")
        f.write(" ".join(sents[i])+"\n")
        f.write(" ".join(pred_tags[i])+"\n")
        f.write(" ".join(test_tags[i])+"\n")
        if len(sents[i])!=len(pred_tags[i]) or len(sents[i])!=len(test_tags[i]):
            print(i)
            print(len(sents[i]),len(pred_tags[i]),len(test_tags[i]))
            print(input_sents[i])
            print(sents[i])
            print(pred_tags[i])
            print(test_tags[i])
            print("error")

        f.write("\n")
        f.write("**************************\n")


print("*****************starting to write to file*****************")
write_bert()
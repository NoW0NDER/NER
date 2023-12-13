import torch
import torch.nn.functional as F
from seqeval.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from torch.utils import data
from tqdm import trange, tqdm
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule

from torch.utils.tensorboard import SummaryWriter
from dataset import CoNLL2003DataSet, CoNLL2003Processor
from models import BertTagger

batch_size = 16
num_epochs = 10
lr = 5e-5
def train(train_iter, eval_iter, model, optimizer, scheduler, num_epochs):
    print("starting to train")
    model_type = "bert"
    writer = SummaryWriter(f"runs/{model_type}")
    max_grad_norm = 1.0  # should be a flag
    for _ in trange(num_epochs, desc="Epoch"):
        # TRAIN loop
        model = model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_iter)):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
            # forward pass
            loss, logits, labels = model(b_input_ids, token_type_ids=b_token_type_ids,
                                         attention_mask=b_input_mask, labels=b_labels,
                                         label_masks=b_label_masks)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            writer.add_scalar
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        acc,f1,prec,rec = eval(eval_iter, model)
        writer.add_scalar('Loss/train', tr_loss / nb_tr_steps, _)
        writer.add_scalar('f1_score/valid', f1, _)
        writer.add_scalar('precision_score/valid', prec, _)
        writer.add_scalar('recall_score/valid', rec, _)
        if _%5==0:
            torch.save(model.state_dict(), f"models/{model_type}_{_}.pt")
    writer.close()
    


def eval(iter_data, model):
    print("starting to evaluate")
    model = model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels = [], []
    for batch in tqdm(iter_data):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch

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
        true_labels.append(labels_to_append)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    # print(pred_tags[:10], valid_tags[:10])
    # print(len(pred_tags), len(valid_tags))
    acc = accuracy_score([valid_tags], [pred_tags])
    f1 = f1_score([valid_tags], [pred_tags])
    prec = precision_score([valid_tags], [pred_tags])
    rec = recall_score([valid_tags], [pred_tags])
    print("Seq eval accuracy: {}".format(acc))
    print("F1-Score: {}".format(f1))
    print("Classification report: -- ")
    print(classification_report([valid_tags], [pred_tags]))
    return acc,f1,prec,rec





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CoNLL2003Processor()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

label_list = processor.get_labels()
tags_vals = label_list
label_map = {label: i for i, label in enumerate(label_list)}
train_data = processor.get_train()
val_data = processor.get_valid()
test_data = processor.get_test()
training_set = CoNLL2003DataSet(train_data, tokenizer, label_map, max_len=256)
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
model.to(device)
param_optimizer = list(model.named_parameters())
num_train_optimization_steps = int(len(train_data) / batch_size) * num_epochs

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
]
warmup_steps = int(0.1 * num_train_optimization_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                    t_total=num_train_optimization_steps)

def train_bert():
    train(train_iter, eval_iter, model, optimizer, scheduler, num_epochs)
    torch.save(model.state_dict(), "models/bert.pt")
    print("starting to test")
    eval(test_iter, model)


if __name__=="__main__":
    train(train_iter, eval_iter, model, optimizer, scheduler, num_epochs)
    torch.save(model.state_dict(), "models/bert.pt")
    print("starting to test")
    eval(test_iter, model)
    
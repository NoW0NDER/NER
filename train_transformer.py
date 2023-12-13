import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from seqeval.metrics import classification_report,f1_score,precision_score,recall_score
from torch.utils.tensorboard import SummaryWriter
from seqProc import *
from evaluate import evaluate
from models import LSTMTagger,TransFormerTagger
from datapreprocess import load_data
from model_config import Transformer_config
batch_size = 64
batched_X,batched_Y,valid_data,test_data,word_to_ix,tag_to_ix,ix_to_tag = load_data(64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# EMBEDDING_DIM = 256
# HIDDEN_DIM = 512
# NUM_LAYERS = 3
# batch_size = 64
# N_HEAD = 4
# DROPOUT = 0.2
# EPOCH = 150

EMBEDDING_DIM = Transformer_config["EMBEDDING_DIM"]
HIDDEN_DIM = Transformer_config["HIDDEN_DIM"]
NUM_LAYERS = Transformer_config["NUM_LAYERS"]
batch_size = 64
N_HEAD = Transformer_config["N_HEAD"]
DROPOUT = Transformer_config["DROPOUT"]
EPOCH = 200


transformer_model = TransFormerTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, N_HEAD, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)

loss_function = nn.CrossEntropyLoss(ignore_index=9)

optimizer = optim.AdamW(transformer_model.parameters(), lr=2e-5)

print(batched_X[1].shape)
print(batched_Y[1].shape)

def train(model,E):
    model_type = model.model_type
    writer = SummaryWriter(f"runs/{model_type}")
    for epoch in tqdm(range(E)):
        model.train()
        for batch_X,batch_y in zip(batched_X,batched_Y):
            model.zero_grad()

            tag_scores = model(batch_X)
            # print(tag_scores.shape)
            loss = loss_function(tag_scores.transpose(1,2), batch_y)
            
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, epoch)
        if epoch%10==0:
            
            writer.add_scalar('f1_score/valid', evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,f1_score), epoch)
            writer.add_scalar('precision_score/valid', evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,precision_score), epoch)
            writer.add_scalar('recall_score/valid', evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,recall_score), epoch)
            model.train()
        if epoch%50==0:
            torch.save(model.state_dict(), f"models/{model_type}_{epoch}.pt")
    print(evaluate(model,test_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,classification_report))
    torch.save(model.state_dict(), f"models/{model_type}_final.pt")
    writer.close()


def train_transformer():
    train(transformer_model,EPOCH)

if __name__ == "__main__":
    train(transformer_model,100)





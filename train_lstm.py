
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
from model_config import LSTM_config
batch_size = 64
batched_X,batched_Y,valid_data,test_data,word_to_ix,tag_to_ix,ix_to_tag = load_data(64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
LSTM
Best trial:
 Value:  0.41928721174004197
 Params: 
    EMBEDDING_DIM: 1002
    HIDDEN_DIM: 350
    NUM_LAYERS: 1
    DROPOUT: 0.042317306776176476
    LR: 0.002324737215436491
    WEIGHT_0: 0.23899740113395876
    WEIGHT_1: 0.6536782767614332
    WEIGHT_2: 0.09417695541755972
    WEIGHT_3: 0.5089931890494696
    WEIGHT_4: 0.7770563167518693
    WEIGHT_5: 0.6771919040060543
    WEIGHT_6: 0.7028309755954322
    WEIGHT_7: 0.2135624003676747
    WEIGHT_8: 0.575830742915188
    optimizer: RMSprop
"""



# EMBEDDING_DIM = 1002
# HIDDEN_DIM = 350
# NUM_LAYERS = 1

# N_HEAD = 16
# DROPOUT = 0

EMBEDDING_DIM = LSTM_config["EMBEDDING_DIM"]
HIDDEN_DIM = LSTM_config["HIDDEN_DIM"]
NUM_LAYERS = LSTM_config["NUM_LAYERS"]
DROPOUT = LSTM_config["DROPOUT"]
EPOCH = 200

# transformer_model = TransFormerTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, N_HEAD, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)

# lstm_model = LSTMTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
lstm_model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),num_layers=NUM_LAYERS,droptout=DROPOUT).to(DEVICE)
loss_function = nn.CrossEntropyLoss(ignore_index=9,weight=torch.tensor([0.23899740113395876,
                                                                        0.6536782767614332,
                                                                        0.09417695541755972,
                                                                        0.5089931890494696,
                                                                        0.7770563167518693,
                                                                        0.6771919040060543,
                                                                        0.7028309755954322,
                                                                        0.2135624003676747,
                                                                        0.575830742915188
                                                                        ]).to(DEVICE))
# loss_function = nn.NLLLoss()
optimizer = optim.AdamW(lstm_model.parameters(), lr=0.002)

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
            # torch.save(model.state_dict(), f"models/{model_type}_{epoch}.pt")
            model.train()
        if epoch%50==0:
            torch.save(model.state_dict(), f"models/{model_type}_{epoch}.pt")
    print(evaluate(model,test_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,classification_report))
    torch.save(model.state_dict(), f"models/{model_type}_final.pt")
    writer.close()

def train_lstm():
    train(lstm_model,EPOCH)


if __name__ == "__main__":
    train(lstm_model,100)





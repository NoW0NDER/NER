# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm import tqdm
# from seqeval.metrics import classification_report,f1_score
# import optuna

# from models import LSTMTagger,TransFormerTagger
# from datapreprocess import load_data
# from seqProc import prepare_sequence
# from evaluate import evaluate
# batched_X,batched_Y,valid_data,test_data,word_to_ix,tag_to_ix,ix_to_tag = load_data(64)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def LSTM_obj(trial):
#     print("LSTM")
#     EMBEDDING_DIM = trial.suggest_int("EMBEDDING_DIM", 32, 1024)
#     HIDDEN_DIM = trial.suggest_int("HIDDEN_DIM", 32, 2048)
#     NUM_LAYERS = trial.suggest_int("NUM_LAYERS", 1, 4)
#     DROPOUT = trial.suggest_float("DROPOUT", 0.1, 0.5)
#     EPOCH = 50
#     LR = trial.suggest_float("LR", 0.0001, 0.01)
#     # O_WEIGHT = trial.suggest_float("O_WEIGHT", 0.001, 0.9)
#     weights = []
#     for i in range(9):
#         param = trial.suggest_float(f"WEIGHT_{i}", 0.0, 1.0)
#         weights.append(param)
#     OPTIMIZER = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
#     model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),num_layers=NUM_LAYERS,droptout=DROPOUT).to(DEVICE)
#     loss_function = nn.CrossEntropyLoss(ignore_index=0,weight=torch.tensor([0]+weights,device=DEVICE))
#     optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
    
#     # optimizer = optim.Adam(model.parameters(), lr=LR)
#     model.train()
#     for epoch in tqdm(range(EPOCH)):
#         for batch_X,batch_y in zip(batched_X[:len(batched_X)//10],batched_Y[:len(batched_X)//10]):
#             model.zero_grad()
#             tag_scores = model(batch_X)
#             loss = loss_function(tag_scores.transpose(1,2), batch_y)
#             loss.backward()
#             optimizer.step()
#         if epoch%30==0:
#             print(trial.params)
#             print(epoch)
#             print(evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,"lstm",classification_report))
#             model.train()
#     print("epoch",epoch)
#     return evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,"lstm",f1_score)





# if __name__=="__main__":
#     torch.manual_seed(42)
#     lstm_study = optuna.create_study(direction="maximize")
#     lstm_study.optimize(LSTM_obj, n_trials=100)
#     print("Best trial:")
#     trial = lstm_study.best_trial
#     print(" Value: ", trial.value)
#     print(" Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
        
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from seqeval.metrics import classification_report,f1_score
import optuna
import pickle
from models import LSTMTagger,TransFormerTagger
from datapreprocess import load_data
from seqProc import prepare_sequence
from evaluate import evaluate
batched_X,batched_Y,valid_data,test_data,word_to_ix,tag_to_ix,ix_to_tag = load_data(64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def LSTM_obj(trial):
    print("LSTM")
    EMBEDDING_DIM = trial.suggest_int("EMBEDDING_DIM", 16, 1024)
    HIDDEN_DIM = trial.suggest_int("HIDDEN_DIM", 16, 2048)
    NUM_LAYERS = trial.suggest_int("NUM_LAYERS", 1, 4)
    DROPOUT = trial.suggest_float("DROPOUT", 0.01, 0.5)
    EPOCH = 50
    LR = trial.suggest_float("LR", 0.0001, 0.01)
    # O_WEIGHT = trial.suggest_float("O_WEIGHT", 0.001, 0.9)
    weights = []
    for i in range(9):
        param = trial.suggest_float(f"WEIGHT_{i}", 0.0, 1.0)
        weights.append(param)
    OPTIMIZER = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),num_layers=NUM_LAYERS,droptout=DROPOUT).to(DEVICE)
    loss_function = nn.CrossEntropyLoss(ignore_index=9,weight=torch.tensor(weights,device=DEVICE))
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
    
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in tqdm(range(EPOCH)):
        for batch_X,batch_y in zip(batched_X[:len(batched_X)//10],batched_Y[:len(batched_X)//10]):
            model.zero_grad()
            tag_scores = model(batch_X)
            loss = loss_function(tag_scores.transpose(1,2), batch_y)
            loss.backward()
            optimizer.step()
        if epoch%49==0:
            print(trial.params)
            print(epoch)
            print(evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,"lstm",classification_report))
            model.train()
    print("epoch",epoch)
    return evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,"lstm",f1_score)





def Transformer_obj(Trial):
    print("Transformer")
    EMBEDDING_DIM = Trial.suggest_categorical("EMBEDDING_DIM", [4*i for i in range(512)])
    HIDDEN_DIM = Trial.suggest_int("HIDDEN_DIM", 1, 2048)
    NUM_LAYERS = Trial.suggest_int("NUM_LAYERS", 1, 5)
    N_HEAD = 4
    DROPOUT = Trial.suggest_float("DROPOUT", 0, 1)
    EPOCH = 20
    LR = Trial.suggest_float("LR", 1e-6, 0.01)
    use_weight = False


    # OPTIMIZER = Trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    model = TransFormerTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, N_HEAD, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    if use_weight:
        weights = []
        for i in range(9):
            param = Trial.suggest_float(f"WEIGHT_{i}", 0.0, 1.0)
            weights.append(param)
        loss_function = nn.CrossEntropyLoss(ignore_index=9,weight=torch.tensor(weights,device=DEVICE))
    else:
        loss_function = nn.CrossEntropyLoss(ignore_index=9)
    # optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for epoch in tqdm(range(EPOCH)):
        for batch_X,batch_y in zip(batched_X[:len(batched_X)//10],batched_Y[:len(batched_X)//10]):
            model.zero_grad()
            tag_scores = model(batch_X)
            loss = loss_function(tag_scores.transpose(1,2), batch_y)
            loss.backward()
            optimizer.step()
        # if epoch%10==0:
        #     print(Trial.params)
        #     print(epoch)
        #     print(evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,classification_report))
        #     model.train()
    print(evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,classification_report))
    return evaluate(model,valid_data,word_to_ix,tag_to_ix,ix_to_tag,model_type,f1_score)
    
    
    
def opt(model_type):
    if model_type == "lstm":
        study = optuna.create_study(direction="maximize")
        study.optimize(LSTM_obj, n_trials=500)
        print("Best trial:")
        trial = study.best_trial
        print(" Value: ", trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    elif model_type == "transformer":
        study = optuna.create_study(direction="maximize")
        study.optimize(Transformer_obj, n_trials=500)
        print("Best trial:")
        trial = study.best_trial
        print(" Value: ", trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    return study, trial.params


if __name__=="__main__":
    model_type = "transformer"
    study, params = opt(model_type)
    with open(f"models/{model_type}_params_test.pkl","wb") as f:
        pickle.dump(params,f)
    with open(f"models/{model_type}_study_test.pkl","wb") as f:
        pickle.dump(study,f)
        
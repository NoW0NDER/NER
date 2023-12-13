from models import LSTMTagger,TransFormerTagger
from datapreprocess import *
import torch
from seqProc import *
from seqeval.metrics import classification_report
from model_config import LSTM_config,Transformer_config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batched_X,batched_Y,valid_data,test_data,word_to_ix,tag_to_ix,ix_to_tag = load_data(64)
def lstm_pred():
    EMBEDDING_DIM = LSTM_config["EMBEDDING_DIM"]
    HIDDEN_DIM = LSTM_config["HIDDEN_DIM"]
    NUM_LAYERS = LSTM_config["NUM_LAYERS"]
    DROPOUT = LSTM_config["DROPOUT"]
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),num_layers=NUM_LAYERS,droptout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load("models/lstm_final.pt"))
    model.eval()
    toeval = []
    true = []
    for pair in test_data:
        out = model(prepare_sequence(pair[0],word_to_ix).to(DEVICE))
        predicted_tags = [torch.argmax(tag_id) for tag_id in out]
        pred_decoded = tagdecoder(predicted_tags,ix_to_tag)
        toeval.append(pred_decoded)
        true.append(pair[1])
    print()
    print("LSTM")
    print(classification_report(true, toeval))
    return true,toeval
def transformer_pred():
    EMBEDDING_DIM = Transformer_config["EMBEDDING_DIM"]
    HIDDEN_DIM = Transformer_config["HIDDEN_DIM"]
    NUM_LAYERS = Transformer_config["NUM_LAYERS"]
    N_HEAD = Transformer_config["N_HEAD"]
    DROPOUT = Transformer_config["DROPOUT"]

    transformer_model = TransFormerTagger(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, N_HEAD, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)

    transformer_model.load_state_dict(torch.load("models/transformer_final.pt"))
    transformer_model.eval()
    toeval = []
    true = []
    true = []
    for pair in test_data:
        out = transformer_model(prepare_sequence(pair[0],word_to_ix).unsqueeze(-1).to(DEVICE))
        predicted_tags = [torch.argmax(tag_id) for tag_id in out]
        pred_decoded = tagdecoder(predicted_tags,ix_to_tag)
        toeval.append(pred_decoded)
        true.append(pair[1])
    print("Transformer")
    print(classification_report(true, toeval))
    return true,toeval
def write_lstm():
    true,pred = lstm_pred()
    # for t,p in zip(true,pred):
    #     print(t)
    #     print(p)
    #     print()
    p_1d = [item for sublist in pred for item in sublist]
    t_1d = [item for sublist in true for item in sublist]
    print(len(p_1d),len(t_1d))
    idx = 0
    with open("conll2003/test.txt","r") as f1:
        with open("out/lstm_text.txt","w") as f2:
            for line in f1:
                if line!="\n":
                    words = line.strip().split()
                    if t_1d[idx] == words[-1]:
                        words[-1] = p_1d[idx]+"\n"
                        idx+=1
                    else:
                        print("error")
                        break
                    f2.write(" ".join(words))
                else:
                    f2.write("\n")
                    
def write_transformer():
    true,pred = transformer_pred()
    # for t,p in zip(true,pred):
    #     print(t)
    #     print(p)
    #     print()
    p_1d = [item for sublist in pred for item in sublist]
    t_1d = [item for sublist in true for item in sublist]
    print(len(p_1d),len(t_1d))
    idx = 0
    with open("conll2003/test.txt","r") as f1:
        with open("out/transformer_text.txt","w") as f2:
            for line in f1:
                if line!="\n":
                    words = line.strip().split()
                    if t_1d[idx] == words[-1]:
                        words[-1] = p_1d[idx]+"\n"
                        idx+=1
                    else:
                        print("error")
                        break
                    f2.write(" ".join(words))
                else:
                    f2.write("\n")



if __name__=="__main__":
    write_lstm()
    write_transformer()
    # t_l,p_l = lstm_pred()
    # t_t,p_t = transformer_pred()
    # print(len(t_l),len(t_t))
    # print(len(p_l),len(p_t))
    # write_bert()
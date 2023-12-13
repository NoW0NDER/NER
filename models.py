from transformers import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertTagger(BertForTokenClassification):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_masks=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
                                   padding_value=-1)
        sequence_output = self.dropout(token_reprs)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss /= mask.float().sum()
            outputs = (loss,) + outputs + (labels,)

        return outputs




class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,num_layers=1,droptout = 0.1):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.model_type = 'lstm'
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True,num_layers=num_layers,dropout=droptout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size-1)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_space = self.hidden2tag(lstm_out)

        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 125):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransFormerTagger(nn.Module):
    def __init__(self, ntoken: int,tagset_size, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super(TransFormerTagger, self).__init__()
        self.model_type = 'transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        # self.linear = nn.Linear(d_model,ntoken)
        # self.totag = nn.Linear(d_model,tagset_size)
        # self.totag = nn.Linear(ntoken,tagset_size)
        self.linear = nn.Linear(d_model,tagset_size-1)
        
        self.init_weights()
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    def forward(self, src, src_mask= None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(DEVICE)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        # output = self.totag(output)
        # tag_scores = F.log_softmax(output, dim=1)
        return output
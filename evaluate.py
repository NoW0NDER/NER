import torch

from seqProc import prepare_sequence,tagdecoder

def evaluate(model,data,word_to_ix,tag_to_ix,ix_to_tag,type,eval_method):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for pair in data:
            sentence = pair[0]
            tags = pair[1]
            if type == "transformer":
                inputs = prepare_sequence(sentence, word_to_ix).unsqueeze(-1)
            else:
                inputs = prepare_sequence(sentence, word_to_ix)
            tag_scores = model(inputs)
            predicted_tags = [torch.argmax(tag_score) for tag_score in tag_scores]

            true = prepare_sequence(tags,tag_to_ix)
            # print(tags)
            # print(tagdecoder(predicted_tags,ix_to_tag))
            y_true.append(tags)
            y_pred.append(tagdecoder(predicted_tags,ix_to_tag))
        return eval_method(y_true, y_pred)
import torch

DEVICE = torch.device("cuda:0")

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(DEVICE)


def tagdecoder(tag_scores,ix_to_tag):
    tags = []
    for tag_score in tag_scores:
        tags.append(ix_to_tag[int(tag_score)])
    return tags
import torch.nn as nn


class WordEmbedding(nn.Module):
    """
    In : (N, sentence_len)
    Out: (N, sentence_len, embd_size)
    """

    def __init__(self, emb_wi, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(emb_wi)

    def forward(self, x):
        return self.embedding(x)

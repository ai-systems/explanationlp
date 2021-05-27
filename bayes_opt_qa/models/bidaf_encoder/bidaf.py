import torch
import torch.nn as nn
import torch.nn.functional as F

# from .char_embedding import CharEmbedding
from .highway import Highway
from .word_embedding import WordEmbedding


class BiDAF_Wemb(nn.Module):
    def __init__(self, w_embd_size, context_embd_size, fact_max_len, hyp_max_len):
        super(BiDAF_Wemb, self).__init__()
        self.embd_size = w_embd_size
        self.d = context_embd_size  # word_embedding + char_embedding
        # self.d = self.embd_size # only word_embedding

        # self.char_embd_net = CharEmbedding(args)

        self.W = nn.Linear(6 * self.d, 1, bias=False)

        self.modeling_layer = nn.GRU(
            8 * self.d,
            self.d,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True,
        )

        self.representation_layer = nn.GRU(
            10 * self.d,
            self.d,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True,
        )

        self.hyp_max_len = hyp_max_len
        self.fact_max_len = fact_max_len

        # self.p2_lstm_layer = nn.GRU(
        # 2 * self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True
        # )
        # self.p2_layer = nn.Linear(10 * self.d, 1)

    def forward(self, embd_context, embd_query, contex_mask, query_mask):
        # 4. Attention Flow Layer
        # Make a similarity matrix
        batch_size = embd_context.size(0)
        T = self.fact_max_len
        J = self.hyp_max_len
        shape = (
            batch_size,
            self.fact_max_len,
            self.hyp_max_len,
            2 * self.d,
        )  # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2)  # (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape)  # (N, T, J, 2d)
        embd_query_ex = embd_query.unsqueeze(1)  # (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)  # (N, T, J, 2d)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex)  # (N, T, J, 2d)
        cat_data = torch.cat(
            (embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3
        )  # (N, T, J, 6d), [h;u;h◦u]
        S = self.W(cat_data).view(batch_size, T, J)  # (N, T, J)

        # Context2Query
        c2q = torch.bmm(
            F.softmax(S, dim=-1), embd_query
        )  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1)  # (N, T)
        q2c = torch.bmm(
            b.unsqueeze(1), embd_context
        )  # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1)  # (N, T, 2d), tiled T times

        # G: query aware representation of each context word
        G = torch.cat(
            (embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2
        )  # (N, T, 8d)

        # 5. Modeling Layer
        M, _h = self.modeling_layer(G)  # M: (N, T, 2d)

        # 6. Output Layer
        G_M = torch.cat((G, M), 2)  # (N, T, 10d)
        _, hn = self.representation_layer(G_M)
        embd = torch.cat([hn[-1], hn[-2]], dim=1)
        return embd

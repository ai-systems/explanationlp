import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn

from .base_clamp import NegClamp, PosClamp
from .bidaf_encoder import BiDAF_Wemb, Highway, WordEmbedding


class DiffOptRelevanceModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        emb_wi,
        hyp_max_len,
        fact_max_len,
        w_embd_size=100,
        num_choices=4,
        num_facts=2,
    ):
        super(DiffOptRelevanceModel, self).__init__()
        self.num_nodes = num_nodes + 1
        self.num_choices = num_choices

        num_nodes = num_nodes + 1
        edges = cp.Variable((num_nodes, num_nodes), nonneg=True)
        edge_weight_param = cp.Parameter((num_nodes, num_nodes))
        adj_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        question_grounding_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        grounding_abstract_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)

        C = cp.reshape(cp.hstack((np.ones((1)), cp.diag(edges).T)), (1, num_nodes + 1))
        C2 = cp.hstack((cp.reshape(cp.diag(edges), (num_nodes, 1)), edges))
        C3 = cp.vstack((C, C2))

        constraints = [
            # edges >= -1,
            edges <= 1,
            C3 >> 0,
            C3.T == C3,
            # cp.diag(edges) == 1,
            # cp.sum(edges) == 2 * num_facts - num_nodes,
            cp.sum(cp.multiply(adj_param, edges)) == 2 * num_facts,
            # cp.sum(cp.diag(edges)) == num_facts,
            # edges[0][0] == 1
            # cp.sum(cp.multiply(question_grounding_param, edges), axis=1)
            # <= cp.sum(cp.multiply(grounding_abstract_param, edges), axis=1),
        ]
        objective = cp.Maximize(cp.sum(cp.multiply(edge_weight_param, edges)))
        prob = cp.Problem(objective, constraints)

        self.cvxpylayer = CvxpyLayer(
            prob,
            parameters=[
                edge_weight_param,
                adj_param,
                # question_grounding_param,
                # grounding_abstract_param,
            ],
            variables=[edges],
        )

        clamp = PosClamp()

        self.grounding_grounding_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.grounding_grounding_param)
        self.abstract_abstract_overlap_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.abstract_abstract_overlap_param)
        self.abstract_abstract_similarity_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.abstract_abstract_similarity_param)
        self.question_grounding_overlap_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.question_grounding_overlap_param)
        self.question_abstract_overlap_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.question_abstract_overlap_param)
        self.question_abstract_similarity_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.question_abstract_similarity_param)
        self.grounding_abstract_overlap_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.grounding_abstract_overlap_param)

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb_wi))
        self.w_embd_size = w_embd_size
        self.hyp_max_len = hyp_max_len
        self.fact_max_len = fact_max_len
        self.context_embd_size = int(w_embd_size / 2)

        self.word_embd_net = WordEmbedding(emb_wi=torch.tensor(emb_wi))
        self.highway_net = Highway(w_embd_size)
        self.ctx_embd_layer = nn.GRU(
            w_embd_size,
            self.context_embd_size,
            bidirectional=True,
            # dropout=0.2,
            batch_first=True,
        )
        self.expl_bidaf = BiDAF_Wemb(
            w_embd_size=w_embd_size,
            context_embd_size=self.context_embd_size,
            hyp_max_len=hyp_max_len,
            fact_max_len=fact_max_len,
        )

        self.answer_bidaf = BiDAF_Wemb(
            w_embd_size=w_embd_size,
            context_embd_size=self.context_embd_size,
            hyp_max_len=hyp_max_len,
            fact_max_len=fact_max_len * (self.num_nodes - 1),
        )
        self.exp_scoring_layer = nn.Sequential(
            nn.Linear(2 * self.context_embd_size, 1), nn.Tanh()
        )
        self.warm_starts = None
        self.loss = nn.CrossEntropyLoss()
        self.answer_scoring_layer = nn.Linear(2 * self.context_embd_size, 1)
        # self.final_score = nn.Sequential(
        # nn.Linear(self.num_choices, self.num_choices, bias=True),
        # nn.Linear(self.num_choices, self.num_choices, bias=False)
        # )

    def build_contextual_embd(self, x_w, x_m):
        # 1. Caracter Embedding Layer
        # char_embd = self.char_embd_net(x_c)  # (N, seq_len, embd_size)
        # 2. Word Embedding Layer
        embd = self.word_embd_net(x_w)  # (N, seq_len, embd_size)
        # Highway Networks for 1. and 2.
        # embd = torch.cat((char_embd, word_embd), 2)  # (N, seq_len, d=embd_size*2)
        embd = self.highway_net(embd)  # (N, seq_len, d=embd_size*2)

        # 3. Contextual  Embedding Layer
        ctx_embd_out, _h = self.ctx_embd_layer(embd)
        # x_m = x_m.unsqueeze(2)
        # x_m = x_m.repeat(1, 1, 2 * self.context_embd_size)
        return ctx_embd_out

    def forward(
        self,
        hypothesis_input_ids,
        fact_input_ids,
        grounding_grounding_overlap,
        abstract_abstract_overlap,
        abstract_abstract_similarity,
        question_grounding_overlap,
        question_abstract_overlap,
        question_abstract_similarity,
        # question_abstract_relevance,
        grounding_abstract_overlap,
        labels,
        hypothesis_attention_masks,
        fact_attention_masks,
        **kwargs,
    ):
        expanded_hypothesis_ids = hypothesis_input_ids.unsqueeze(2)
        hypothesis_attention_masks = hypothesis_attention_masks.unsqueeze(2)
        expanded_hypothesis_ids = expanded_hypothesis_ids.repeat(
            1, 1, self.num_nodes - 1, 1
        )
        hypothesis_attention_masks = hypothesis_attention_masks.repeat(
            1, 1, self.num_nodes - 1, 1
        )
        expanded_hypothesis_ids = expanded_hypothesis_ids.view(-1, self.hyp_max_len)
        hypothesis_attention_masks = hypothesis_attention_masks.view(
            -1, self.hyp_max_len
        )
        fact_input_ids = fact_input_ids.view(-1, self.fact_max_len)
        fact_attention_masks = fact_attention_masks.view(-1, self.fact_max_len)

        embd_facts = self.build_contextual_embd(
            fact_input_ids, fact_attention_masks
        )  # (N, T, 2d)
        embd_hypothesis = self.build_contextual_embd(
            expanded_hypothesis_ids, hypothesis_attention_masks
        )

        bidaf_representation = self.expl_bidaf(
            embd_facts,
            embd_hypothesis,
            fact_attention_masks,
            hypothesis_attention_masks,
        )
        bidaf_representation = bidaf_representation.view(
            -1, self.num_nodes - 1, 2 * self.context_embd_size
        )
        score = self.exp_scoring_layer(bidaf_representation).squeeze()
        # score = torch.clamp(score, min=-1, max=1)
        # score = torch.sigmoid(score)

        grounding_grounding_overlap = grounding_grounding_overlap.view(
            -1, self.num_nodes, self.num_nodes
        )
        abstract_abstract_overlap = abstract_abstract_overlap.view(
            -1, self.num_nodes, self.num_nodes
        )
        abstract_abstract_similarity = abstract_abstract_similarity.view(
            -1, self.num_nodes, self.num_nodes
        )
        question_grounding_overlap = question_grounding_overlap.view(
            -1, self.num_nodes, self.num_nodes
        )
        question_abstract_overlap = abstract_abstract_overlap.view(
            -1, self.num_nodes, self.num_nodes
        )
        question_abstract_similarity = question_abstract_similarity.view(
            -1, self.num_nodes, self.num_nodes
        )
        grounding_abstract_overlap = grounding_abstract_overlap.view(
            -1, self.num_nodes, self.num_nodes
        )

        question_abstract_similarity = torch.zeros_like(grounding_abstract_overlap)
        question_abstract_similarity[:, 0, 1:] = score
        question_abstract_similarity[:, 1:, 0] = score
        # question_abstract_similarity[:, 0, 0] = 10
        # abstract_abstract_overlap[:, 0, 0] = 1
        # abstract_abstract_overlap = 1 - abstract_abstract_overlap

        # edge_weights = (
        #     self.abstract_abstract_overlap_param * abstract_abstract_overlap * -1
        #     # + self.abstract_abstract_similarity_param * abstract_abstract_similarity
        #     # + self.grounding_grounding_param * grounding_grounding_overlap * -1
        #     + self.question_abstract_overlap_param * question_abstract_overlap
        #     # + self.question_grounding_overlap_param * question_grounding_overlap
        #     # + self.grounding_abstract_overlap_param * grounding_abstract_overlap
        #     + self.question_abstract_similarity_param * question_abstract_similarity
        #     # + opts["question_abstract_relevance"] * question_abstract_relevance
        # )
        edge_weights = (
            self.abstract_abstract_overlap_param * abstract_abstract_overlap * -1
            # + self.abstract_abstract_similarity_param * abstract_abstract_similarity
            # + self.grounding_grounding_param * grounding_grounding_overlap * -1
            + self.question_abstract_overlap_param * question_abstract_overlap
            # + self.question_grounding_overlap_param * question_grounding_overlap
            # + self.grounding_abstract_overlap_param * grounding_abstract_overlap
            + self.question_abstract_similarity_param * question_abstract_similarity
            # + opts["question_abstract_relevance"] * question_abstract_relevance
        )
        adj = torch.where(
            (question_abstract_similarity) != 0,
            torch.ones_like(question_abstract_similarity),
            torch.zeros_like(question_abstract_similarity),
        )
        # print(abstract_abstract_overlap[0], "abs overlap")
        # print(abstract_abstract_similarity[0], "abs simi")
        # print(question_abstract_similarity[0], "q abs sim")
        # print(question_abstract_overlap[0], "q abs overlap")
        # print(edge_weights[0], "edge_weights")
        question_grounding_edges = torch.where(
            question_grounding_overlap != 0,
            torch.ones_like(question_grounding_overlap),
            torch.zeros_like(question_grounding_overlap),
        )

        grounding_abstract_edges = torch.where(
            grounding_abstract_overlap != 0,
            torch.ones_like(question_grounding_overlap),
            torch.zeros_like(question_grounding_overlap),
        )
        (edges, xs, ys, ss) = self.cvxpylayer(
            edge_weights,
            adj,
            # question_grounding_edges,
            # grounding_abstract_edges,
            solver_args={
                #"acceleration_lookback": 10,
                "n_jobs_forward":8,
                "n_jobs_backward":8,
                "verbose": False,
                "eps": 1e-7,
                "max_iters": 100000,
                # "mode": "dense",
                "warm_starts": self.warm_starts,
                # "warm_start": self.warm_starts,
            },
        )
        self.warm_starts = (xs, ys, ss)

        # selected = torch.round(torch.diagonal(edges, dim1=1, dim2=2)[:, 1:])
        # # # selected = torch.round(torch.ones((8, self.num_nodes - 1)).cuda())
        # selected = selected.unsqueeze(2)
        # selected = selected.repeat(1, 1, self.fact_max_len)
        # selected = selected.unsqueeze(3)
        # selected = selected.repeat(1, 1, 1, 2 * self.context_embd_size)
        # # selected_representation = torch.sum(bidaf_representation * selected, dim=1)

        # hypothesis_input_ids = hypothesis_input_ids.view(-1, self.hyp_max_len)
        # hypothesis = self.build_contextual_embd(hypothesis_input_ids)
        # embd_facts = embd_facts.reshape(
        #     -1, (self.num_nodes - 1) * self.fact_max_len, (2 * self.context_embd_size)
        # )
        # selected = selected.reshape(
        #     -1, (self.num_nodes - 1) * self.fact_max_len, (2 * self.context_embd_size)
        # )
        # embd_facts = embd_facts * selected
        # selected_representation = self.answer_bidaf(embd_facts, hypothesis)

        # answer_score = self.answer_scoring_layer(selected_representation).squeeze()
        # answer_score = answer_score.view(-1, self.num_choices)

        # edges = edges.view(-1, self.num_choices, self.num_nodes * self.num_nodes)
        # edge_weights = edge_weights.view(
        #     -1, self.num_choices, self.num_nodes * self.num_nodes
        # )
        edge_weights = edge_weights.view(-1, self.num_nodes * self.num_nodes)
        edges = edges.view(-1, self.num_nodes * self.num_nodes)
        score = edge_weights * torch.round(edges)
        answer_score = torch.sum(score, dim=1)
        answer_score = answer_score.view(-1, self.num_choices)
        # print(answer_score)
        # answer_score = self.final_score(answer_score)
        # answer_score = self.final_score(answer_score)

        # print(torch.sigmoid(score * 0.01))
        return self.loss(answer_score, labels), answer_score

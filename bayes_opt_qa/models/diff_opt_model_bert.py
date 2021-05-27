import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn
from transformers import BertModel, BertPreTrainedModel

from .base_clamp import NegClamp, PosClamp


class BaseBertDiffModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BaseBertDiffModel, self).__init__(config=config)
        num_nodes = 40
        self.num_nodes = num_nodes + 1
        self.num_choices = 4
        self.bert = BertModel(config)
        self.exp_len_linear = nn.Linear(config.hidden_size, 1)

        num_nodes = num_nodes + 1
        edges = cp.Variable((num_nodes, num_nodes), symmetric=True)
        edge_weight_param = cp.Parameter((num_nodes, num_nodes))
        adj_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        question_grounding_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        grounding_abstract_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        exp_len_param = cp.Parameter()

        C = cp.reshape(cp.hstack((np.ones((1)), cp.diag(edges).T)), (1, num_nodes + 1))
        C2 = cp.hstack((cp.reshape(cp.diag(edges), (num_nodes, 1)), edges))
        C3 = cp.vstack((C, C2))

        constraints = [
            edges >= 0,
            edges <= 1,
            C3 >> 0,
            C3.T == C3,
            cp.sum(cp.multiply(adj_param, edges)) <= 2 * exp_len_param,
            cp.sum(cp.multiply(question_grounding_param, edges), axis=1)
            <= cp.sum(cp.multiply(grounding_abstract_param, edges), axis=1),
        ]
        objective = cp.Maximize(cp.sum(cp.multiply(edge_weight_param, edges)))
        prob = cp.Problem(objective, constraints)

        self.cvxpylayer = CvxpyLayer(
            prob,
            parameters=[
                edge_weight_param,
                adj_param,
                question_grounding_param,
                grounding_abstract_param,
                exp_len_param,
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
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        hypothesis_input_ids,
        hypothesis_attention_mask,
        grounding_grounding_overlap,
        abstract_abstract_overlap,
        abstract_abstract_similarity,
        question_grounding_overlap,
        question_abstract_overlap,
        question_abstract_similarity,
        # question_abstract_relevance,
        grounding_abstract_overlap,
        labels,
        opts={},
        **kwargs,
    ):
        hypothesis_input_ids = hypothesis_input_ids.view(-1, 64)
        hypothesis_attention_mask = hypothesis_attention_mask.view(-1, 64)
        qa_outputs = self.bert(
            hypothesis_input_ids, attention_mask=hypothesis_attention_mask
        )[1]
        exp_len = torch.clamp(self.exp_len_linear(qa_outputs), min=1, max=5)
        exp_len = exp_len.squeeze()

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

        edge_weights = (
            self.abstract_abstract_overlap_param * abstract_abstract_overlap * -1
            + self.abstract_abstract_similarity_param * abstract_abstract_similarity
            + self.grounding_grounding_param * grounding_grounding_overlap * -1
            + self.question_abstract_overlap_param * question_abstract_overlap
            + self.question_grounding_overlap_param * question_grounding_overlap
            + self.grounding_abstract_overlap_param * grounding_abstract_overlap
            + self.question_abstract_similarity_param * question_abstract_similarity
            # + opts["question_abstract_relevance"] * question_abstract_relevance
        )
        adj = torch.where(
            (edge_weights) != 0,
            torch.ones_like(question_abstract_similarity),
            torch.zeros_like(question_abstract_similarity),
        )
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
        (edges,) = self.cvxpylayer(
            edge_weights,
            adj,
            question_grounding_edges,
            grounding_abstract_edges,
            exp_len,
            solver_args={
                "acceleration_lookback": 0,
                "verbose": False,
                "eps": 1e-7,
                "max_iters": 1000000,
            },
        )
        edges = edges.view(-1, self.num_choices, self.num_nodes * self.num_nodes)
        edge_weights = edge_weights.view(
            -1, self.num_choices, self.num_nodes * self.num_nodes
        )
        score = edge_weights * edges
        score = torch.sum(score, dim=2)
        # print(score)

        # print(torch.sigmoid(score * 0.01))

        return self.loss(score, labels), score

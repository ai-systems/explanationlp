import cvxpy as cp
import numpy as np
import torch
from bayes_opt_qa.models.base_clamp import NegClamp, PosClamp, QuestionClamp
from bayes_opt_qa.models.encoder.transformer_model import TransformerModel
from cvxpylayers.torch import CvxpyLayer
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import AutoModel


class WiTransform(nn.Module):
    def __init__(self, num_nodes):
        super(WiTransform, self).__init__()
        self.num_nodes = num_nodes

    def forward(self, val):
        return torch.softmax(val.view(-1, self.num_nodes * self.num_nodes), dim=1).view(
            -1, self.num_nodes, self.num_nodes
        )


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        hyp_max_len,
        fact_max_len,
        transformer_model,
        combined_len,
        w_embd_size=768,
        num_choices=4,
        num_facts=2,
    ):
        super(ContrastiveModel, self).__init__()
        logger.info("Loading the model")
        self.num_nodes = num_nodes + 4
        self.num_choices = num_choices
        self.combined_len = combined_len

        num_nodes = num_nodes + 4
        edges = cp.Variable((num_nodes, num_nodes), PSD=True)
        edge_weight_param = cp.Parameter((num_nodes, num_nodes))
        adj_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        question_grounding_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        grounding_abstract_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        no_of_questions = cp.Parameter()

        C = cp.reshape(cp.hstack((np.ones((1)), cp.diag(edges).T)), (1, num_nodes + 1))
        C2 = cp.hstack((cp.reshape(cp.diag(edges), (num_nodes, 1)), edges))
        C3 = cp.vstack((C, C2))

        constraints = [
            # edges >= 0,
            # edges <= 1,
            # C3 >> 0,
            # C3.T == C3,
            cp.diag(edges) == 1,
            cp.sum(cp.multiply(adj_param, edges)) <= 2 * 3,
            # cp.sum(cp.diag(edges[:4, :4])) == 1,
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
                # no_of_questions,
                # question_grounding_param,
                # grounding_abstract_param,
            ],
            variables=[edges],
        )

        clamp = PosClamp()
        q_clamp = QuestionClamp()

        self.fact_limit_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        q_clamp.apply(self.fact_limit_param)
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

        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb_wi))
        self.w_embd_size = w_embd_size
        self.hyp_max_len = hyp_max_len
        self.fact_max_len = fact_max_len

        # self.embedding_layer = TransformerModel(
        #     ntoken=emb_wi.shape[0],
        #     ninp=w_embd_size,
        #     nhead=8,
        #     nhid=2048,
        #     nlayers=2,
        # )

        self.model = AutoModel.from_pretrained(transformer_model)
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        print(self.model)

        freeze_layers = "0,1,2,3,4"
        layer_indexes = [int(x) for x in freeze_layers.split(",")]
        for layer_idx in layer_indexes:
            for param in list(self.model.transformer.layer[layer_idx].parameters()):
                param.requires_grad = False

        # self.q_embedding_layer = TransformerModel(
        #     ntoken=emb_wi.shape[0],
        #     ninp=w_embd_size,
        #     nhead=8,
        #     nhid=2048,
        #     nlayers=2,
        # )

        # self.model = SentenceTransformer("paraphrase-distilroberta-base-v1")
        self.dense_size = 256

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.scoring_layer = nn.Linear(self.w_embd_size, 1)
        self.abstract_scoring_layer = nn.Linear(4 * self.w_embd_size, 1)
        # self.scoring_layer = nn.Linear(self.w_embd_size, self.dense_size)
        self.transform_softmax = WiTransform(self.num_nodes)
        self.final_score = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 4))

        self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.MarginRankingLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        hypothesis_fact_ids,
        hypothesis_fact_attention_masks,
        grounding_grounding_overlap,
        abstract_abstract_overlap,
        abstract_abstract_similarity,
        question_grounding_overlap,
        question_abstract_overlap,
        question_abstract_similarity,
        # question_abstract_relevance,
        grounding_abstract_overlap,
        question_question_score,
        labels,
        gold_similarity_scores,
        **kwargs,
    ):

        hypothesis_fact_ids = hypothesis_fact_ids.view(-1, self.combined_len)
        hypothesis_fact_attention_masks = hypothesis_fact_attention_masks.view(
            -1, self.combined_len
        )

        hidden_state = self.model(hypothesis_fact_ids, hypothesis_fact_attention_masks)[
            0
        ]
        pooled_output = hidden_state[:, 0]

        relevance_score = self.scoring_layer(pooled_output).squeeze()

        # print(hypothesis_fact_ids)
        # print(relevance_score)
        similarity_scores = torch.sigmoid(
            relevance_score.view(
                -1, self.num_choices, self.num_nodes - self.num_choices
            )
        )
        # print(similarity_scores)

        question_abstract_similarity = torch.zeros_like(grounding_abstract_overlap)
        question_abstract_similarity[
            :, self.num_choices :, : self.num_choices
        ] = similarity_scores.transpose(1, 2)
        question_abstract_similarity[
            :, : self.num_choices, self.num_choices :
        ] = similarity_scores

        edge_weights = (
            self.abstract_abstract_overlap_param * (abstract_abstract_overlap) * -1
            + self.abstract_abstract_similarity_param * (abstract_abstract_similarity)
            # + self.grounding_grounding_param * (grounding_grounding_overlap) * -1
            + self.question_abstract_overlap_param * (question_abstract_overlap)
            # + self.question_grounding_overlap_param * (question_grounding_overlap)
            # + self.grounding_abstract_overlap_param * (grounding_abstract_overlap)
            + self.question_abstract_similarity_param * (question_abstract_similarity)
            + question_question_score
            # + opts["question_abstract_relevance"] * question_abstract_relevance
        )

        # print(similarity_scores)
        # print(self.question_abstract_similarity_param)

        # # edge_weights = (
        # #     (abstract_abstract_overlap) * -1
        # #     # + self.abstract_abstract_similarity_param * (abstract_abstract_similarity)
        # #     # + self.grounding_grounding_param * (grounding_grounding_overlap) * -1
        # #     + (question_abstract_overlap)
        # #     # + self.question_grounding_overlap_param * (question_grounding_overlap)
        # #     # + self.grounding_abstract_overlap_param * (grounding_abstract_overlap)
        # #     + (question_abstract_similarity)
        # #     + question_question_score
        # #     # + opts["question_abstract_relevance"] * question_abstract_relevance
        # # )

        # # # print((edge_weights[0].transpose(0, 1) == edge_weights[0]).all())
        # # # edge_weights = (
        # # #     (abstract_abstract_overlap) * -1
        # # #     + (abstract_abstract_similarity)
        # # #     + (question_abstract_overlap)
        # # #     + (question_grounding_overlap)
        # # #     + (question_abstract_similarity)
        # # #     + question_question_score
        # # #     # + opts["question_abstract_relevance"] * question_abstract_relevance
        # # # )
        # # # print(question_abstract_similarity[0].nonzero())
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
            # self.fact_limit_param,
            # question_grounding_edges,
            # grounding_abstract_edges,
            solver_args={
                # "acceleration_lookback": 20,
                # "acceleration_lookback": 0,
                "verbose": False,
                # "mode": "dense",
                "eps": 1e-8,
                # "max_iters": 1000000,
                # "max_iters": 1000000,
                "max_iters": 1000000,
                # "max_iters": 100000,
            },
        )
        edges = edges.view(-1, self.num_nodes, self.num_nodes)

        # # print(edges[:, :4])
        # # edge_weights = edge_weights.view(-1, self.num_nodes * self.num_nodes)
        # # score = edge_weights * edges
        # # score = torch.sum(score, dim=2) * 0.01

        # # print(torch.sigmoid(score * 0.01))
        # # for val in edges[0]:
        # # print(val)
        # # print(torch.sum(edges.view(-1, self.num_nodes * self.num_nodes)[0]))

        final_pred = self.final_score(torch.diagonal(edges[:, :4, :4], dim1=1, dim2=2))
        # final_pred = torch.zeros((2, 4))

        return (
            # self.loss(final_pred1.view(-1), final_pred2.view(-1), (a1 - a2).view(-1)),
            self.loss(final_pred, labels),
            # + self.mse_loss(
            #     similarity_scores.view(-1), gold_similarity_scores.view(-1)
            # ),
            final_pred,
        )

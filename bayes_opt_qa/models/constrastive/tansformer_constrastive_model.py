import cvxpy as cp
import numpy as np
import torch
from bayes_opt_qa.models.base_clamp import NegClamp, PosClamp, QuestionClamp
from bayes_opt_qa.models.encoder.transformer_model import TransformerModel
from cvxpylayers.torch import CvxpyLayer
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import AutoModel


class TransformerContrastiveModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        hyp_max_len,
        fact_max_len,
        transformer_model,
        w_embd_size=768,
        num_choices=4,
        num_facts=2,
    ):
        super(TransformerContrastiveModel, self).__init__()
        self.num_nodes = num_nodes + 4
        self.num_choices = num_choices

        num_nodes = num_nodes + 4
        edges = cp.Variable((num_nodes, num_nodes), symmetric=True)
        edge_weight_param = cp.Parameter((num_nodes, num_nodes))
        adj_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        question_grounding_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        grounding_abstract_param = cp.Parameter((num_nodes, num_nodes), nonneg=True)
        no_of_questions = cp.Parameter()

        C = cp.reshape(cp.hstack((np.ones((1)), cp.diag(edges).T)), (1, num_nodes + 1))
        C2 = cp.hstack((cp.reshape(cp.diag(edges), (num_nodes, 1)), edges))
        C3 = cp.vstack((C, C2))

        constraints = [
            edges >= 0,
            edges <= 1,
            C3 >> 0,
            C3.T == C3,
            cp.sum(cp.multiply(adj_param, edges)) <= 2 * no_of_questions,
            cp.sum(cp.diag(edges[:4, :4])) == 1,
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
                no_of_questions,
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

        self.w_embd_size = w_embd_size
        self.hyp_max_len = hyp_max_len
        self.fact_max_len = fact_max_len

        self.model = AutoModel.from_pretrained(transformer_model)
        # self.q_embedding_layer = TransformerModel(
        #     ntoken=emb_wi.shape[0],
        #     ninp=w_embd_size,
        #     nhead=8,
        #     nhid=2048,
        #     nlayers=2,
        # )

        # self.model = SentenceTransformer("paraphrase-distilroberta-base-v1")

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.scoring_layer = nn.Linear(4 * self.w_embd_size, 1)

        self.loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        hypothesis_input_ids,
        hypothesis_attention_mask,
        fact_attention_mask,
        fact_input_ids,
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

        fact_input_ids = fact_input_ids.view(-1, self.fact_max_len)
        fact_attention_mask = fact_attention_mask.view(-1, self.fact_max_len)
        fact_embedding = self.model(fact_input_ids, fact_attention_mask)[0]
        input_mask_expanded = (
            fact_attention_mask.unsqueeze(-1).expand(fact_embedding.size()).float()
        )
        fact_seq_embedding = torch.sum(fact_embedding * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        fact_seq_embedding = fact_seq_embedding / sum_mask

        hypothesis_input_ids = hypothesis_input_ids.view(-1, self.hyp_max_len)
        hypothesis_attention_mask = hypothesis_attention_mask.view(-1, self.hyp_max_len)
        hypothesis_embedding = self.model(
            hypothesis_input_ids, hypothesis_attention_mask
        )[0]
        input_mask_expanded = (
            hypothesis_attention_mask.unsqueeze(-1)
            .expand(hypothesis_embedding.size())
            .float()
        )
        hypothesis_seq_embedding = torch.sum(
            hypothesis_embedding * input_mask_expanded, 1
        )
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        hypothesis_seq_embedding = hypothesis_seq_embedding / sum_mask

        hypothesis_seq_embedding = hypothesis_seq_embedding.view(
            -1, self.num_choices, self.w_embd_size
        )
        fact_seq_embedding = fact_seq_embedding.view(
            -1, self.num_nodes - self.num_choices, self.w_embd_size
        )

        hypothesis_norm = (
            hypothesis_seq_embedding / hypothesis_seq_embedding.norm(dim=2)[:, :, None]
        )
        fact_norm = fact_seq_embedding / fact_seq_embedding.norm(dim=2)[:, :, None]
        similarity_scores = torch.bmm(hypothesis_norm, fact_norm.transpose(1, 2))

        # score = self.cos(hypothesis_seq_embedding, fact_seq_embedding)
        # score = self.scoring_layer(combined_representation).squeeze()
        # score = score.view(-1, self.num_nodes)
        # score = torch.sigmoid(score)

        question_abstract_similarity = torch.zeros_like(grounding_abstract_overlap)
        question_abstract_similarity[
            :, self.num_choices :, : self.num_choices
        ] = similarity_scores.transpose(1, 2)
        question_abstract_similarity[
            :, : self.num_choices, self.num_choices :
        ] = similarity_scores
        # question_abstract_similarity[:, 0, 0] = 10

        edge_weights = (
            self.abstract_abstract_overlap_param * (abstract_abstract_overlap) * -1
            + self.abstract_abstract_similarity_param * (abstract_abstract_similarity)
            + self.grounding_grounding_param * (grounding_grounding_overlap) * -1
            + self.question_abstract_overlap_param * (question_abstract_overlap)
            + self.question_grounding_overlap_param * (question_grounding_overlap)
            + self.grounding_abstract_overlap_param * (grounding_abstract_overlap)
            + self.question_abstract_similarity_param * (question_abstract_similarity)
            + question_question_score
            # + opts["question_abstract_relevance"] * question_abstract_relevance
        )
        # print(question_abstract_similarity[0].nonzero())
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
            self.fact_limit_param,
            # question_grounding_edges,
            # grounding_abstract_edges,
            solver_args={
                "acceleration_lookback": 20,
                # "acceleration_lookback": 0,
                "verbose": False,
                # "mode": "dense",
                "eps": 1e-4,
                # "max_iters": 1000000,
                "max_iters": 5000,
                # "max_iters": 100000,
            },
        )
        edges = edges.view(-1, self.num_nodes, self.num_nodes)

        # print(edges[:, :4])
        # edge_weights = edge_weights.view(-1, self.num_nodes * self.num_nodes)
        # score = edge_weights * edges
        # score = torch.sum(score, dim=2) * 0.01

        # print(torch.sigmoid(score * 0.01))
        # for val in edges[0]:
        # print(val)
        # print(torch.sum(edges.view(-1, self.num_nodes * self.num_nodes)[0]))

        return (
            self.loss(torch.diagonal(edges[:, :4, :4], dim1=1, dim2=2) * 100, labels),
            # + 0.3 * self.mse_loss(similarity_scores, gold_similarity_scores),
            torch.diagonal(edges[:, :4, :4], dim1=1, dim2=2),
        )

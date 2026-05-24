# -*- coding: utf-8 -*-
# @Time    : 2026/05/05
# @Author  : Gemini CLI (4500-2 Pod M3 Agent)

r"""
GMUSASRec
################################################
Reference:
    Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A.
        "Gated Multimodal Units for Information Fusion." arXiv 2017.
    Wang-Cheng Kang et al.
        "Self-Attentive Sequential Recommendation." in ICDM 2018.
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType


class GMUSASRec(SequentialRecommender):
    """Sequential recommender that fuses item ID embeddings and a multimodal
    feature (FLOAT_SEQ, e.g. image vector) via Gated Multimodal Units (GMU).

    Architecture
    ------------
    For each position in the interaction sequence:
        h_item  = tanh(W_item  * item_emb)
        h_image = tanh(W_image * image_feat)
        z       = sigmoid(W_gate * [item_emb; image_feat])
        fused   = z * h_item + (1 - z) * h_image

    The fused representation is then processed by a Transformer encoder
    identical to SASRec.

    Candidate scoring (CE / BPR)
    ----------------------------
    Candidate item scores are computed via inner product between the final
    sequence representation and ``item_embedding.weight`` (ID space only).
    Image features contribute to sequence encoding but not to candidate
    representation. This is consistent with SASRecF and the standard RecBole
    sequential recommender convention. If full multimodal candidate scoring
    is required, a separate item tower that also passes through GMU is needed.

    Configuration requirements
    --------------------------
    - ``selected_features`` must contain exactly one ``FLOAT_SEQ`` field.
      If more than one FLOAT_SEQ field is listed, only the first is used and
      a warning is emitted.
    - The FLOAT_SEQ field must NOT appear in ``discretization`` config.
      Discretization rewrites channel-0 to constant 1, which silently
      destroys multimodal fusion.
    - When the field is listed in ``numerical_features``, RecBole stores the
      feature as a (value, mask) tensor of shape [n_items, D, 2]; this class
      extracts channel-0 (actual values) automatically.
    - ``normalize_field`` or ``normalize_all`` is recommended to prevent
      tanh saturation for high-magnitude image vectors.
    """

    def __init__(self, config, dataset):
        super(GMUSASRec, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.selected_features = config["selected_features"]

        # --- P3-B: FLOAT_SEQ 필드 수집 및 다중 필드 경고 ---
        float_seq_fields = [
            f for f in self.selected_features
            if dataset.field2type[f] == FeatureType.FLOAT_SEQ
        ]
        if not float_seq_fields:
            raise ValueError(
                "GMUSASRec requires at least one FLOAT_SEQ field in "
                "'selected_features'. Check dataset field types."
            )
        if len(float_seq_fields) > 1:
            self.logger.warning(
                f"GMUSASRec uses only the first FLOAT_SEQ feature "
                f"'{float_seq_fields[0]}'. "
                f"Remaining fields {float_seq_fields[1:]} are ignored."
            )
        self.image_feat_field = float_seq_fields[0]

        # --- P1-A: 이산화 설정 충돌 검증 ---
        dis_info = config.get("discretization") or {}
        if self.image_feat_field in dis_info:
            raise ValueError(
                f"GMUSASRec does not support discretization on "
                f"'{self.image_feat_field}'. "
                "Discretization replaces channel-0 with constant 1, which "
                "silently destroys multimodal fusion. "
                "Remove it from 'discretization' config."
            )

        # --- P1-B: numerical_features 경로 감지 ---
        numerical_features = config.get("numerical_features") or []
        use_tuple = self.image_feat_field in numerical_features

        self.image_feat_dim = dataset.field2seqlen[self.image_feat_field]

        # --- P4-A: 정규화 미적용 경고 ---
        normalize_fields = list(config.get("normalize_field") or [])
        normalize_all = bool(config.get("normalize_all"))
        if self.image_feat_field not in normalize_fields and not normalize_all:
            self.logger.warning(
                f"Image feature '{self.image_feat_field}' is not normalized. "
                "High-magnitude vectors may cause tanh saturation in GMU. "
                "Consider setting 'normalize_field' or 'normalize_all'."
            )

        # --- P2-B: image_feat_matrix 를 register_buffer 로 등록 ---
        # register_buffer 를 사용하면 model.to(device) 시 자동으로 이동하며,
        # state_dict 에도 포함되어 체크포인트 저장/복원이 완결된다.
        raw_feat = dataset.get_item_feature()[self.image_feat_field]
        if use_tuple:
            # numerical_features 경로: shape [n_items, D, 2]
            # 채널 0 = 실제 float 벡터, 채널 1 = 패딩 mask (all 1, 이산화 OFF 시)
            raw_feat = raw_feat[..., 0].float()
        else:
            raw_feat = raw_feat.float()
        self.register_buffer("image_feat_matrix", raw_feat)

        # --- 레이어 정의 ---
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.w_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_image = nn.Linear(self.image_feat_dim, self.hidden_size)
        self.w_gate = nn.Linear(self.hidden_size + self.image_feat_dim, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gmu_fusion(self, item_emb, image_feat):
        """Gated Multimodal Unit fusion.

        h_item  = tanh(W_item  * item_emb)          [B, L, H]
        h_image = tanh(W_image * image_feat)         [B, L, H]
        z       = sigmoid(W_gate * [item_emb; image_feat])  [B, L, H]
        output  = z * h_item + (1 - z) * h_image    [B, L, H]
        """
        h_item = self.tanh(self.w_item(item_emb))
        h_image = self.tanh(self.w_image(image_feat))
        gate_input = torch.cat([item_emb, image_feat], dim=-1)
        z = self.sigmoid(self.w_gate(gate_input))
        return z * h_item + (1 - z) * h_image

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)

        # --- P2-B resolved: image_feat_matrix 는 buffer 이므로 item_seq.device 와 동일 ---
        image_feat = self.image_feat_matrix[item_seq]  # [B, L, D]

        # --- P2-A: 패딩 위치(item_id=0)의 이미지를 0 벡터로 마스킹 ---
        # item_embedding 은 padding_idx=0 으로 0 벡터를 반환하지만
        # image_feat_matrix[0] 은 아이템 0의 실제 특징이므로 일관성을 맞춘다.
        padding_mask = (item_seq == 0).unsqueeze(-1)  # [B, L, 1]
        image_feat = image_feat.masked_fill(padding_mask, 0.0)

        input_emb = self.gmu_fusion(item_emb, image_feat)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output  # [B, H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores

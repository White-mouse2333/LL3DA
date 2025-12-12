import math, os  # math: 数学函数；os: 操作系统相关（此文件里可能暂时没用到）
from functools import partial  # partial：固定函数部分参数，方便复用

import numpy as np  # 数值计算库
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块

# PointNet++ 的采样/聚合模块（用于点云下采样和特征提取）
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
# 最远点采样（FPS），常用于从点云里选代表点
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

# 常用工具：huber loss（鲁棒回归损失）
from utils.misc import huber_loss
# 点坐标缩放/平移到统一范围的工具（归一化、反归一化等）
from utils.pc_util import scale_points, shift_scale_points
# 数据集配置（比如类别数、角度bin数等）
from datasets.scannet import BASE
from typing import Dict  # 类型标注

# 模型配置与损失函数构建
from models.detector_Vote2Cap_DETR.config import model_config
from models.detector_Vote2Cap_DETR.criterion import build_criterion
# 通用 MLP（这里常用于 1x1 conv 形式的“MLP头”）
from models.detector_Vote2Cap_DETR.helpers import GenericMLP

# Vote Query：从 encoder 的点特征里生成 DETR 的 query（投票/候选点）
from models.detector_Vote2Cap_DETR.vote_query import VoteQuery

# 位置编码（把 3D 坐标编码成可学习/可用的 embedding）
from models.detector_Vote2Cap_DETR.position_embedding import PositionEmbeddingCoordsSine

# Transformer 编码器/解码器相关实现
from models.detector_Vote2Cap_DETR.transformer import (
    MaskedTransformerEncoder, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer
)

    
class BoxProcessor(object):
    """
    将 3DETR 的 MLP Head 输出（中心偏移、尺寸、角度、类别等）
    转换为可用的 3D bounding box（包括 corners 等）的工具类
    """

    def __init__(self, dataset_config):
        # dataset_config 通常包含：类别数、角度bin数、box corner计算方式等
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        """
        计算预测框中心：
        - center_offset: 网络预测的中心偏移（通常是归一化/小范围偏移）
        - query_xyz: query 对应的 3D 坐标（投票得到的点）
        - point_cloud_dims: [min_xyz, max_xyz]，每个 batch 的点云范围
        """
        # 先把偏移加到 query 的坐标上，得到未归一化坐标下的中心
        center_unnormalized = query_xyz + center_offset
        # 再把中心坐标按点云范围做归一化（例如映射到 [0,1]）
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        """
        计算预测框尺寸：
        - size_normalized: 网络输出的归一化尺寸（通常 0~1）
        - point_cloud_dims: [min_xyz, max_xyz]，用于反归一化回真实尺度
        """
        # 场景尺度 = max - min（每个 batch 都可能不同）
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)  # 防止尺度太小导致数值不稳定
        # 把归一化尺寸放大回真实尺度
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        """
        计算预测框朝向角（yaw）：
        - angle_logits: 角度分类 logits（num_angle_bin 类）
        - angle_residual: 每个角度 bin 的残差回归
        """
        if angle_logits.shape[-1] == 1:
            # 特殊情况：某些数据集不考虑旋转角
            # 这里仍然让输出参与计算图，避免 DDP 报 unused variable
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            # 每个角度 bin 对应的角度跨度
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            # 预测最可能的角度类别（bin）
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            # 该 bin 的中心角
            angle_center = angle_per_cls * pred_angle_class
            # 加上该 bin 的残差，得到连续角度
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            # 把角度范围调整到 (-pi, pi]
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        """
        将语义分类 logits 转为：
        - semcls_prob: 每个语义类别概率（不含背景）
        - objectness_prob: 是“物体”的概率（1 - 背景概率）
        """
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1  # +1 是背景类
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]  # 最后一类当作背景/非物体
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        """
        把 (center, size, angle) 参数化形式转换为 8 个角点坐标 corners
        具体实现由 dataset_config 提供（不同数据集坐标系可能不同）
        """
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model_Vote2Cap_DETR(nn.Module):
    """
    Vote2Cap-DETR 检测器主体：
    - 输入：点云
    - tokenizer(pre-encoder)：PointNet++ 采样聚合生成 token
    - encoder：Transformer 编码点特征
    - vote_query_generator：从 encoder 特征生成 queries
    - decoder：Transformer 解码得到每个 query 的 box 特征
    - mlp heads：把 box 特征映射为类别、中心、尺寸、角度等
    """
    
    def __init__(
        self,
        tokenizer,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        criterion=None
    ):
        super().__init__()
        
        self.tokenizer = tokenizer  # 点云 token 化模块（PointNet++）
        self.encoder = encoder      # Transformer encoder
        
        # 如果 encoder 有 masking_radius，说明是 masked encoder（结构不同，投影层深度略不同）
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        
        # 把 encoder 输出特征投影到 decoder 维度（通常 encoder_dim == decoder_dim，但也可能不同）
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,              # 用 1x1 conv 实现 MLP（对点/序列更方便）
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        # 位置编码：把 (x,y,z) 转为可输入 Transformer 的 embedding
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        
        # 从 encoder 特征生成 num_queries 个 query（投票生成）
        self.vote_query_generator = VoteQuery(decoder_dim, num_queries)
        
        # 对 query 的位置编码再过一层投影（得到 query_pos）
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        
        self.decoder = decoder  # Transformer decoder
        # 构建检测头（类别、中心、尺寸、角度等）
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.box_processor = BoxProcessor(dataset_config)  # 把 head 输出转换为 box
        self.criterion = criterion  # 损失函数（训练时用）
        


    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        """
        构建多个 MLP Head，用于把 decoder 的 box_features 变成最终预测：
        - 语义类别（含背景）
        - 中心偏移
        - 尺寸
        - 角度分类 + 角度残差
        """
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # 语义类别：num_semcls + 1（最后一类当背景/非物体）
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # box 几何：中心、尺寸、角度
        center_head = mlp_func(output_dim=3)  # x,y,z 偏移
        size_head = mlp_func(output_dim=3)    # dx,dy,dz
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)  # 角度bin分类
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)  # 对每个bin的残差回归

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)


    def _break_up_pc(self, pc):
        """
        把点云拆分为：
        - xyz: (B, N, 3)
        - features: (B, C, N)  （如果有颜色/法向等额外通道）
        """
        # pc 可能包含 RGB/normal 等额外信息：[..., 0:3] 是 xyz
        xyz = pc[..., 0:3].contiguous()
        # 若 pc 的最后一维 >3，则把后面的当 feature，并转成 (B, C, N)
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features


    def run_encoder(self, point_clouds):
        """
        运行 tokenizer + encoder，得到编码后的点坐标与特征
        返回：
        - enc_xyz: (B, Nenc, 3)
        - enc_features: (Nenc, B, C)   （注意这里是 Transformer 常用的序列优先格式）
        - enc_inds: (B, Nenc)          （enc_xyz/enc_features 对应原始点的索引）
        """
        xyz, features = self._break_up_pc(point_clouds)
        
        # -------------------------
        # 1) 点云 tokenization（PointNet++ SA）
        # xyz: (B, N, 3)
        # features: (B, C, N)
        # pre_enc_inds: (B, Npre) （token 对应原始点索引）
        # -------------------------
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.tokenizer(xyz, features)

        # nn.MultiHeadAttention 通常用 (S, B, C)，所以把特征转成 (Npre, B, C)
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # -------------------------
        # 2) Transformer encoder
        # enc_xyz: (B, Nenc, 3)
        # enc_features: (Nenc, B, C)
        # enc_inds: (B, Nenc) 或 None（取决于 encoder 是否下采样）
        # -------------------------
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder 没有下采样：索引沿用 tokenizer 的索引
            enc_inds = pre_enc_inds
        else:
            # encoder 做了下采样：把 pre_enc_inds 根据 enc_inds gather 出真正索引
            # 用 gather 兼容 FPS 和随机采样等不同策略
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long())
        return enc_xyz, enc_features, enc_inds

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        将 decoder 的输出 box_features 转成最终 box 预测结果（含 corners 等）

        参数：
            query_xyz: (B, Nq, 3)  每个 query 的 3D 坐标
            point_cloud_dims: [min_xyz, max_xyz]
                - min_xyz: (B, 3)
                - max_xyz: (B, 3)
            box_features: (L, Nq, B, C)
                - L: decoder 层数（return_intermediate=True 时会返回每层输出）
        """
        # box_features 转成 (L, B, C, Nq) 方便用 1x1 conv 形式的 MLP head
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        # 合并 (L, B) 维度：变为 (L*B, C, Nq)
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # -------------------------
        # MLP head 输出：
        # 这里输出形状先是 (L*B, out_dim, Nq)，再 transpose 成 (L*B, Nq, out_dim)
        # -------------------------
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)

        # center_offset：用 sigmoid 限制到 (0,1)，再减 0.5 -> (-0.5, 0.5)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape 回 (L, B, Nq, out_dim)
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )

        # 把 normalized residual 放缩到角度范围（常见做法：乘一个比例系数）
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # -------------------------
            # BoxProcessor：把 head 输出转换为可解释的 3D box
            # -------------------------
            center_normalized, center_unnormalized = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # 下面这些（类别概率、objectness 概率）主要用于匹配/评估，不用于 loss
            # 用 no_grad 避免 DDP 因“变量未使用”而报错
            with torch.no_grad():
                semcls_prob, objectness_prob = self.box_processor.compute_objectness_and_cls_prob(
                    cls_logits[l]
                )

            box_prediction = {
                "sem_cls_logits": cls_logits[l],  # (B, Nq, num_semcls+1)
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,       # (B, Nq, 8, 3) 之类的角点格式（取决于实现）
            }
            outputs.append(box_prediction)

        # 训练时可用中间层输出作为辅助监督（aux loss）
        aux_outputs = outputs[:-1]  # 前 L-1 层
        outputs = outputs[-1]       # 最后一层作为最终输出

        return {
            "outputs": outputs,         # decoder 最后一层输出
            "aux_outputs": aux_outputs, # decoder 中间层输出
        }

    def forward(self, inputs, is_eval: bool=False):
        """
        前向过程：
        1) encoder 编码点云
        2) vote query 生成 queries
        3) decoder 解码得到每个 query 的 box_features
        4) box head 输出最终预测
        5) 如果训练且 criterion 不为空，计算匹配与 loss
        """
        point_clouds = inputs["point_clouds"]
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        
        # -------------------------
        # 1) feature encoding（编码点云特征）
        # enc_features: (Nenc, B, C) -> 转成 (B, C, Nenc) 方便后续卷积形式处理
        # -------------------------
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = enc_features.permute(1, 2, 0)  # (B, C, Nenc)
        
        # -------------------------
        # 2) vote query generation（生成 query）
        # query_xyz: (B, Nq, 3)
        # query_features: (B, C, Nq) 或类似（取决于 VoteQuery 实现）
        # -------------------------
        query_outputs = self.vote_query_generator(enc_xyz, enc_features)
        query_outputs['seed_inds'] = enc_inds  # 记录 query 对应的原始点索引
        query_xyz = query_outputs['query_xyz']
        query_features = query_outputs["query_features"]
        
        # -------------------------
        # 3) decoding（Transformer 解码）
        # - query 的位置编码：pos_embedding(query_xyz)
        # - encoder 的位置编码：pos_embedding(enc_xyz)
        # -------------------------
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        
        # 把 encoder 特征投影到 decoder 维度： (B, Cenc, Nenc) -> (B, Cdec, Nenc)
        enc_features = self.encoder_to_decoder_projection(enc_features)
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder 常用输入格式： (S, B, C)
        enc_features = enc_features.permute(2, 0, 1)  # (Nenc, B, C)
        enc_pos = enc_pos.permute(2, 0, 1)            # (Nenc, B, C)
        query_embed = query_embed.permute(2, 0, 1)    # (Nq, B, C)
        tgt = query_features.permute(2, 0, 1)         # (Nq, B, C) 作为 decoder 的初始 query 特征
        
        # decoder 输出：
        # [0] 是中间层堆叠输出，形状通常为 (L, Nq, B, C)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]    # nlayers x nqueries x batch x channel

        # -------------------------
        # 4) 将 box_features -> box predictions
        # -------------------------
        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        
        # -------------------------
        # 5) 训练阶段：计算匹配与 loss
        # -------------------------
        if self.criterion is not None and is_eval is False:
            (
                box_predictions['outputs']['assignments'],  # 匹配结果（预测与GT如何对应）
                box_predictions['outputs']['loss'],         # 各项 loss
                _
            ) = self.criterion(query_outputs, box_predictions, inputs)
            
        # 补充一些中间特征到输出，方便后续模块或可视化调试
        box_predictions['outputs'].update({
            'prop_features': box_features.permute(0, 2, 1, 3),  # (L, B, Nq, C)
            'enc_features': enc_features.permute(1, 0, 2),      # (B, Nenc, C)
            'enc_xyz': enc_xyz,                                 # (B, Nenc, 3)
            'query_xyz': query_xyz,                             # (B, Nq, 3)
        })
        
        return box_predictions['outputs']



def build_preencoder(cfg):
    """
    构建 tokenizer / pre-encoder（PointNet++ Set Abstraction）
    用于从原始点云采样 + 聚合邻域特征，得到较少数量的“token 点”
    """
    mlp_dims = [cfg.in_channel, 64, 128, cfg.enc_dim]  # MLP 通道逐层变化
    preencoder = PointnetSAModuleVotes(
        radius=0.2,                 # 邻域球半径
        nsample=64,                 # 邻域采样点数
        npoint=cfg.preenc_npoints,  # 下采样后的点数
        mlp=mlp_dims,               # PointNet MLP
        normalize_xyz=True,         # 是否对 xyz 做归一化处理
    )
    return preencoder


def build_encoder(cfg):
    """
    构建 Transformer Encoder：
    - vanilla：普通 TransformerEncoder
    - masked：带中间下采样 + masking 机制的 Encoder
    """
    if cfg.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.enc_dim,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_ffn_dim,
            dropout=cfg.enc_dropout,
            activation=cfg.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=cfg.enc_nlayers
        )
    elif cfg.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.enc_dim,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_ffn_dim,
            dropout=cfg.enc_dropout,
            activation=cfg.enc_activation,
        )
        # interim_downsampling：encoder 中间做一次 PointNet++ 下采样
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=cfg.preenc_npoints // 2,
            mlp=[cfg.enc_dim, 256, 256, cfg.enc_dim],
            normalize_xyz=True,
        )
        
        # masking_radius：不同层的 mask 半径（这里用平方形式存储）
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {cfg.enc_type}")
    return encoder


def build_decoder(cfg):
    """
    构建 Transformer Decoder（DETR 风格）
    return_intermediate=True：返回每一层 decoder 的输出（用于 aux loss）
    """
    decoder_layer = TransformerDecoderLayer(
        d_model=cfg.dec_dim,
        nhead=cfg.dec_nhead,
        dim_feedforward=cfg.dec_ffn_dim,
        dropout=cfg.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=cfg.dec_nlayers, return_intermediate=True
    )
    return decoder


def detector(args, dataset_config):
    """
    组装整个检测模型：
    1) 读取 cfg
    2) 构建 tokenizer/encoder/decoder
    3) 构建 criterion（匹配 + loss）
    4) 返回 Model_Vote2Cap_DETR
    """
    cfg = model_config(args, dataset_config)
    
    tokenizer = build_preencoder(cfg)
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    
    criterion = build_criterion(cfg, dataset_config)
    
    model = Model_Vote2Cap_DETR(
        tokenizer,
        encoder,
        decoder,
        cfg.dataset_config,
        encoder_dim=cfg.enc_dim,
        decoder_dim=cfg.dec_dim,
        mlp_dropout=cfg.mlp_dropout,
        num_queries=cfg.nqueries,
        criterion=criterion
    )
    return model

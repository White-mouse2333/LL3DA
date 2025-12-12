# -*- coding: utf-8 -*-
# 这个文件主要做的事：
# 1) 从 3D detector 的输出中拿到 proposal/box 等信息
# 2) 把「box 提示」和「click 提示」编码成 Q-Former 可用的视觉 prompt tokens
# 3) 用 Q-Former 生成一段 prefix feature，再喂给大语言模型（LLM）做 caption / QA / chat 的生成或训练

import copy, math, importlib
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict

from collections import OrderedDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig
)
from models.ll3da.generation_utils import generation
from models.ll3da.position_embedding import PositionEmbeddingCoordsSine

from utils.box_util import box3d_iou_batch_tensor


def proposal_dimension_select(features: Tensor, indices: Tensor) -> Tensor:
    """
    在 proposal 维度上做 gather（按给定 index 从 features 中取出对应 proposal 的特征）

    参数
    ----------
    features : Tensor，形状 [batch x nsrc x ...]
        “数据池/候选池”，比如 nsrc=proposal 数量
    indices : Tensor，形状 [batch x ntgt]
        需要从 nsrc 里挑选的索引（每个 target 一个 index）

    返回
    -------
    Tensor，形状 [batch x ntgt x ...]
        在 dim=1（proposal 维度）上按 indices 收集后的结果
    """
    return torch.gather(
        features, 1,
        indices.reshape(
            *(indices.shape + tuple(1 for _ in features.shape[2:]))
        ).repeat(
            *((1, 1) + features.shape[2:])
        )
    )


def select_proposal_feature(
    prop_features: Tensor, prop_box_corners: Tensor, prop_sem_mask: Tensor, box_query: Tensor
) -> Tensor:
    """
    用 3D IoU 把每个 query box 匹配到最相似的 proposal，然后取出对应 proposal 的特征

    参数
    ----------
    prop_features : Tensor，[batch x nproposal x n_embd]
        proposal 的特征
    prop_box_corners : Tensor，[batch x nproposal x 8 x 3]
        proposal 的 3D box 八个角点坐标
    prop_sem_mask : Tensor，[batch x nproposal]
        语义有效 mask：背景为 0，非背景为 1
    box_query : Tensor，[batch x nquery x 8 x 3]
        输入的查询 box（比如用户给的 box prompt）

    返回
    -------
    Tensor，[batch x nquery x n_embd]
        对每个 query box，取 IoU 最大的 proposal 的特征
    """
    batch_size, nproposal, _, _ = prop_box_corners.shape
    nquery = box_query.shape[1]

    # 计算每个 query box 与每个 proposal box 的 IoU
    matched_box_iou = box3d_iou_batch_tensor(
        prop_box_corners.unsqueeze(1).repeat(1, nquery, 1, 1, 1).reshape(-1, 8, 3),
        box_query.unsqueeze(2).repeat(1, 1, nproposal, 1, 1).reshape(-1, 8, 3)
    )
    matched_box_iou = matched_box_iou.reshape(batch_size, nquery, nproposal)

    # 把背景 proposal 的 IoU 清零，避免被选中
    matched_box_iou = matched_box_iou * prop_sem_mask.unsqueeze(1)

    # 对每个 query，选 IoU 最大的 proposal index
    matched_indices = matched_box_iou.argmax(-1)  # [batch x nquery]

    # 在 proposal 维度把对应 proposal 的特征 gather 出来
    return proposal_dimension_select(prop_features, matched_indices)


class PromptEncoder(nn.Module):
    """
    把来自 detector 的信息（box/click 等 prompt）编码成 Q-Former 可用的 prompt tokens。
    """

    def __init__(self, encoder_hidden_size, visual_nquery, qformer_hidden_size, n_embd):
        super(PromptEncoder, self).__init__()
        self.n_embd = n_embd
        self.visual_nquery = visual_nquery            # 每个 prompt 展开成多少个 token
        self.qformer_hidden_size = qformer_hidden_size
        self.encoder_hidden_size = encoder_hidden_size

        # box prompt 的 MLP：把 encoder 特征映射到 qformer hidden，并展开成 visual_nquery 个 token
        self.box_prompt_projector = nn.Sequential(
            nn.Linear(encoder_hidden_size, qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_hidden_size, visual_nquery * qformer_hidden_size),
        )

        # click prompt 的 MLP：先做 3D 位置编码，再映射+展开
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(encoder_hidden_size, qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_hidden_size, visual_nquery * qformer_hidden_size),
        )

        # 3D 坐标位置编码（Fourier/Sine-Cosine）
        self.pos_emb3d = PositionEmbeddingCoordsSine(
            d_pos=encoder_hidden_size,
            pos_type='fourier',
            normalize=True
        )

    def expand_prompt_representation(self, prompt_feature: Tensor, prompt_mask: Tensor = None):
        """
        把每个 prompt 的 feature 展开成 visual_nquery 个 token

        输入
        ----
        prompt_feature: [batch x nprompt x (visual_nquery*qformer_hidden_size)] 或可reshape的等价形状
        prompt_mask:    [batch x nprompt]，1 表示该 prompt 有效

        输出
        ----
        prompt_feature: [batch x (nprompt*visual_nquery) x qformer_hidden_size]
        prompt_mask:    [batch x (nprompt*visual_nquery)]
        """
        batch_size, nprompt = prompt_feature.shape[:2]

        # 没提供 mask 就默认全有效
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_feature[..., 0])

        # mask 也要展开到 visual_nquery 个 token
        prompt_mask = prompt_mask.unsqueeze(-1).repeat(1, 1, self.visual_nquery)
        prompt_mask = prompt_mask.reshape(batch_size, nprompt * self.visual_nquery)

        # feature reshape 成 [batch x nprompt x visual_nquery x hidden] 再拉平成 token 序列
        prompt_feature = prompt_feature.reshape(batch_size, nprompt, self.visual_nquery, self.qformer_hidden_size)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt * self.visual_nquery, self.qformer_hidden_size)

        return prompt_feature, prompt_mask

    def forward(
        self,
        detector_output,
        point_cloud_dims,
        box_query=None,
        box_qmask=None,
        click_query=None,
        click_qmask=None
    ):
        """
        把 box_query / click_query 编码成 prompt tokens，并拼接返回

        detector_output 里需要用到：
        - sem_cls_logits: [batch x nproposal x nclass]（用于区分背景）
        - prop_features[-1]: [batch x nproposal x encoder_hidden_size]（proposal 特征）
        - box_corners: [batch x nproposal x 8 x 3]（proposal box）

        返回：
        - prompt_feature: [batch x ntoken_total x qformer_hidden_size]
        - prompt_mask:    [batch x ntoken_total]
        """
        sem_cls_logits = detector_output['sem_cls_logits']

        # 背景类别通常在最后一类，这里把非背景设为 1
        prop_sem_mask = (sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)).float()

        net_device = sem_cls_logits.device
        batch_size = sem_cls_logits.shape[0]

        # 先放一个空的起始张量，便于后面 cat
        visual_prompt = [torch.zeros(batch_size, 0, self.qformer_hidden_size).to(net_device)]
        visual_mask = [torch.zeros(batch_size, 0).to(net_device)]

        # -------- box prompt 编码：通过 IoU 匹配 proposal，然后 MLP 展开成 tokens
        if box_query is not None:
            box_prompt = select_proposal_feature(
                detector_output['prop_features'][-1],
                detector_output['box_corners'],
                prop_sem_mask,
                box_query
            )
            box_prompt = self.box_prompt_projector(box_prompt)
            box_prompt, box_qmask = self.expand_prompt_representation(box_prompt, box_qmask)
            visual_prompt.append(box_prompt)
            visual_mask.append(box_qmask)

        # -------- click prompt 编码：对点击点做 3D 位置编码，再 MLP 展开成 tokens
        if click_query is not None:
            click_xyz = click_query  # [batch x nquery x 3]
            click_prompt = self.pos_emb3d(click_xyz, input_range=point_cloud_dims)  # -> [batch x C x nquery] 或等价
            click_prompt = self.click_prompt_projector(click_prompt.permute(0, 2, 1))
            click_prompt, click_qmask = self.expand_prompt_representation(click_prompt, click_qmask)
            visual_prompt.append(click_prompt)
            visual_mask.append(click_qmask)

        # 拼接所有 prompt tokens 和 mask
        prompt_feature = torch.cat(visual_prompt, dim=1)  # [batch x ntoken_total x hidden]
        prompt_mask = torch.cat(visual_mask, dim=1)       # [batch x ntoken_total]

        return prompt_feature, prompt_mask


class captioner(nn.Module):
    """
    核心模块：
    - detector_output 提供 3D proposals / encoder features
    - PromptEncoder 把 box/click prompt 编码成 tokens
    - Q-Former 把视觉 tokens + latent queries 汇聚成 prefix feature
    - LLM 进行 caption/QA/chat 的生成与训练
    """

    def train(self, mode: bool = True):
        """
        重写 train()：如果 freeze_llm=True，则把 LLM 固定在 eval + requires_grad=False
        """
        super().train(mode)
        if self.freeze_llm is True:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        return self

    def __init__(self, args, train_dataset):
        super(captioner, self).__init__()

        # detector encoder 输出的 hidden size（通常来自 3D backbone）
        self.encoder_hidden_size = 256
        self.dtype = torch.float16

        # 每个 prompt（box/click）展开成多少个 qformer token
        self.visual_nquery = 8

        # Q-Former 的 latent query 数量（可学习）
        self.nlatent_query = 32

        self.freeze_llm = args.freeze_llm

        # -------- tokenizer（用于解码/配置 eos）
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        self.nvocabs = len(self.tokenizer)

        # -------- LLM（用于生成与训练）
        self.transformer = AutoModelForCausalLM.from_pretrained(
            args.vocab,
            torch_dtype=self.dtype
        )
        self.n_embd = self.transformer.config.hidden_size

        # -------- Q-Former（多模态汇聚器）
        qformer_config = InstructBlipQFormerConfig(
            num_hidden_layers=6,
            encoder_hidden_size=self.encoder_hidden_size
        )
        self.qformer = InstructBlipQFormerModel.from_pretrained(
            args.qformer_vocab,
            config=qformer_config
        )
        self.qformer_hidden_size = qformer_config.hidden_size

        # -------- 把 detector encoder 特征映射到 Q-Former 期望的 encoder hidden size
        self.encoder_to_qformer_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_config.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
        )

        # prompt encoder：box/click -> qformer tokens
        self.prompt_encoder = PromptEncoder(
            self.encoder_hidden_size,
            self.visual_nquery,
            self.qformer_hidden_size,
            self.n_embd
        )

        # 可学习 latent queries（[nlatent_query x qformer_hidden_size]）
        self.latent_query = nn.Embedding(self.nlatent_query, self.qformer_hidden_size)

        # 把 Q-Former 输出映射到 LLM 的 embedding 维度
        self.qformer_to_language_projection = nn.Linear(self.qformer_hidden_size, self.n_embd)

        # 一次最多生成多少条（避免显存爆炸）
        self.max_gen_per_iter = 8

        # -------- generation 配置（beam search / eos 等）
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 4 if args.use_beam_search is True else None,
        }

        self.train()

    def _get_instruction_response(
        self,
        detector_output: dict,
        inputs: dict,
        box_query: Tensor = None,
        box_qmask: Tensor = None,
        click_query: Tensor = None,
        click_qmask: Tensor = None
    ) -> dict:
        """
        生成给 LLM 的 prefix feature：
        detector_output + (box/click prompts) -> PromptEncoder -> Q-Former -> prefix_feature

        返回：
        prefix_feature: [batch x nlatent_query x n_embd]
        """
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        net_device = inputs["point_clouds"].device
        batch_size = inputs["point_clouds"].shape[0]
        encoder_hidden_states = detector_output['enc_features']

        # 1) prompt encoding（box/click）
        prompt_feature, prompt_mask = self.prompt_encoder(
            detector_output,
            point_cloud_dims,
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask
        )

        # 2) 构造 Q-Former 的 query tokens：latent queries + prompt tokens
        query_tokens = self.latent_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_tokens = torch.cat((query_tokens, prompt_feature), dim=1)

        # attention mask：latent 全 1 + prompt_mask
        query_attention_mask = torch.cat(
            (torch.ones(batch_size, self.nlatent_query).to(net_device), prompt_mask), dim=1
        )

        # 3) 拼上文本侧（qformer_input_ids）的 attention mask
        query_attention_mask = torch.cat((query_attention_mask, inputs['qformer_attention_mask']), dim=1)

        # 4) 运行 Q-Former
        query_outputs = self.qformer(
            input_ids=inputs['qformer_input_ids'],
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=self.encoder_to_qformer_projection(encoder_hidden_states),
        )

        # 只取前 nlatent_query（latent 部分）的输出作为 prefix
        query_outputs = query_outputs[0][:, : self.nlatent_query, :]
        prefix_feature = self.qformer_to_language_projection(query_outputs)

        return prefix_feature

    def forward(self, detector_output: dict, inputs: dict, is_eval: bool = False, task_name: str = 'qa') -> dict:
        """
        统一入口：
        - 训练：forward_training
        - 推理：根据 task_name 走 densecap / qa / chat
        """
        if is_eval is False:
            return self.forward_training(detector_output, inputs)

        response_config = {
            'ov-det': 64,
            'dense-cap': 48,
            'qa': 16,
            'chat': 512,
        }
        max_gen_length = response_config[task_name]

        if task_name in {'ov-det', 'dense-cap'}:
            return self.predict_densecap(detector_output, inputs, task_name, max_gen_length=max_gen_length)
        elif task_name == 'qa':
            return self.predict_answer(detector_output, inputs, max_gen_length=max_gen_length)
        else:
            return self.predict_chat(detector_output, inputs, max_gen_length=max_gen_length)

    def forward_training(self, detector_output: Dict, inputs: Dict) -> Dict:
        """
        训练阶段：
        - 先用 Q-Former 得到 prefix_tokens（视觉+prompt 汇聚）
        - 再把 prefix_tokens + 文本 token embeddings 拼起来喂给 LLM
        - 用 cross entropy 计算 loss（按 gradient_mask 做有效位置加权）
        """
        input_ids = inputs['input_ids']           # [batch x ntokens]
        input_mask = inputs['attention_mask']     # [batch x ntokens]
        gradient_mask = inputs['gradient_mask']   # [batch x ntokens]（哪些 token 参与 loss）

        box_query = inputs.get('box_query', None)       # [batch x nquery x 8 x 3]
        box_qmask = inputs.get('box_mask', None)        # [batch x nquery]
        click_query = inputs.get('click_query', None)   # [batch x nquery x 3]
        click_qmask = inputs.get('click_mask', None)    # [batch x nquery]

        embedding_layer = self.transformer.get_input_embeddings()

        # 视觉 prefix： [batch x nprefix x n_embd]
        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask
        )
        prefix_mask = torch.ones_like(prefix_tokens[..., 0])  # prefix 全有效

        # 拼接：prefix embeddings + 文本 embeddings
        inputs_embeds = torch.cat((prefix_tokens, embedding_layer(input_ids)), dim=1)
        attention_mask = torch.cat((prefix_mask, input_mask), dim=1)

        # 前向得到 logits
        outputs = self.transformer(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask.to(self.dtype),
        )

        # logits 对齐：从 prefix_tokens.shape[1]-1 开始，预测后续文本 token
        detector_output['loss'] += self.loss_caption(
            logits=outputs.logits[:, prefix_tokens.shape[1] - 1: -1],
            target=input_ids,
            mask=gradient_mask.to(self.dtype),
        )
        return detector_output

    def loss_caption(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        计算 token-level cross entropy，并用 mask 做加权平均

        logits: [batch x ntoken x vocab]
        target: [batch x ntoken]
        mask:   [batch x ntoken]（1 表示参与 loss）
        """
        loss_per_word = nnf.cross_entropy(
            logits.permute(0, 2, 1).contiguous(),
            target,
            reduction='none',
        )

        final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)

        # 多卡训练时，有时需要确保所有参数都“参与”计算图（这里用 0*sum(...) 的方式挂上去）
        for param in self.parameters():
            if param.requires_grad:
                final_loss += 0 * torch.sum(param.to(final_loss.dtype) ** 2)
        return final_loss

    def predict_densecap(self, detector_output: Dict, inputs: Dict, task_name: str, max_gen_length: int = 64) -> Dict:
        """
        dense caption / open-vocabulary detection：
        - 对每个 proposal 生成一条描述
        - ov-det 时额外加入 click_query（可能是每个 proposal 的 query 点）
        """
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        batch_size, nproposals, _, _ = detector_output['box_corners'].shape

        # 输出 token ids：初始化为 eos
        output_ids = torch.ones(batch_size, nproposals, max_gen_length).long().to(net_device)
        output_ids = output_ids * self.tokenizer.eos_token_id

        # ---- 取 instruction（注意这里用了 [0]，意味着它可能是“共享指令模板”）
        instruction = inputs['instruction'][0]              # [ntoken]
        instruction_mask = inputs['instruction_mask'][0]    # [ntoken]
        instruction_id = instruction[instruction_mask == 1] # 过滤掉 padding
        instruction_id = instruction_id[None, :].repeat(batch_size, 1)
        instruction_embedding = embedding_layer(instruction_id)  # [batch x ntoken x n_embd]

        # 对每个 proposal 构造 prefix tokens
        prefix_tokens = []
        for proposal_id in range(nproposals):
            box_query = detector_output['box_corners'][:, [proposal_id]]  # [batch x 1 x 8 x 3]

            click_query = None
            if task_name == 'ov-det':
                click_query = detector_output['query_xyz'][:, [proposal_id]]  # [batch x 1 x 3]

            instruct_prefix_feature = self._get_instruction_response(
                detector_output=detector_output,
                inputs=inputs,
                box_query=box_query,
                click_query=click_query,
            )  # [batch x nprefix x n_embd]

            # prefix 后面拼上 instruction 的 embedding
            instruct_prefix_feature = torch.cat((instruct_prefix_feature, instruction_embedding), dim=1)
            prefix_tokens.append(instruct_prefix_feature.unsqueeze(1))

        # [batch x nproposal x 1 x n_embd] -> 实际上第 3 维是 token 序列维
        prefix_tokens = torch.cat(prefix_tokens, dim=1).to(self.dtype)

        # 用 detector 的语义分类过滤背景 proposal
        sem_cls_logits = detector_output["sem_cls_logits"]
        objectness_mask = sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)

        # 只对非背景 proposals 做生成
        candidate_prefix = prefix_tokens[objectness_mask].to(self.dtype)

        # 分 batch 生成，避免一次性太大
        gather_output_ids = []
        for start_idx in range(0, candidate_prefix.shape[0], self.max_gen_per_iter):
            prefix = candidate_prefix[start_idx: start_idx + self.max_gen_per_iter]
            scene_cap_output = generation(
                self.transformer,
                inputs_embeds=prefix,
                max_length=max_gen_length,
                **self.caption_config
            )
            gather_output_ids.append(scene_cap_output['output_ids'])
        gather_output_ids = torch.cat(gather_output_ids, dim=0)

        # 把生成结果写回对应 proposal
        output_ids[objectness_mask] = gather_output_ids
        detector_output['output_ids'] = output_ids

        return detector_output

    def predict_answer(self, detector_output: Dict, inputs: Dict, max_gen_length: int = 8) -> Dict:
        """
        QA 推理：
        - 每个样本：prefix + instruction -> generation()
        """
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device

        output_ids = []

        instruction = inputs['instruction']            # [batch x ntoken]
        instruction_mask = inputs['instruction_mask']  # [batch x ntoken]

        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        ).to(self.dtype)

        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]
            sample_mask = instruction_mask[batch_id]

            output = generation(
                self.transformer,
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),  # [1 x nprefix x n_embd]
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_length=max_gen_length,
                **self.caption_config
            )
            output_ids.append(output['output_ids'])

        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids

        return detector_output

    def predict_chat(self, detector_output: Dict, inputs: Dict, max_gen_length: int = 512) -> Dict:
        """
        Chat 推理：
        - prefix + instruction -> transformer.generate（采样式生成）
        - 再把长度 pad/截断到 max_gen_length
        """
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device

        output_ids = []

        instruction = inputs['instruction']
        instruction_mask = inputs['instruction_mask']

        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        ).to(self.dtype)

        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]
            sample_mask = instruction_mask[batch_id]

            output = self.transformer.generate(
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_new_tokens=max_gen_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=4,
                num_return_sequences=1,
            )  # [1 x generated_len]

            output = output.squeeze(0)

            # pad 到固定长度（不足补 eos，超出截断）
            placeholder = torch.ones(max_gen_length).to(net_device) * self.tokenizer.eos_token_id
            output = output[:min(max_gen_length, output.shape[0])]
            placeholder[:output.shape[0]] = output

            output_ids.append(placeholder.unsqueeze(0).long())

        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids

        return detector_output

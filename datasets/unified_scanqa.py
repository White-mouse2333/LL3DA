import os, json
import torch
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from datasets.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_qa import evaluate
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT


class Dataset(ScanNetBaseDataset):
    
    def __init__(
        self,
        args,
        dataset_config,
        split_set="train",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
    ):
        """
        初始化数据集对象（构造函数）
        - args: 训练/配置参数（里面包含 vocab、最大长度等）
        - dataset_config: 数据集配置
        - split_set: "train" 或 "val"（训练集/验证集）
        - num_points: 每个点云采样点数
        - use_color/use_normal/use_multiview/use_height: 是否使用额外特征
        - augment: 是否数据增强
        """
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,        # 是否随机裁剪立方体（这里关闭）
            random_cuboid_min_points=None,  # 裁剪最少点数（关闭后无效）
        )
        
        # 当前任务名（ScanQA：点云问答任务）
        self.task_name = 'scanqa'
        # 将 3D box 坐标离散化时用的网格大小
        self.grid_size_3d = args.grid_size_3d
        # 最大提示数量（用于 box_query/click_query 的预留槽位）
        self.max_prompts = args.max_prompts
        # 当前数据属于 train 还是 val
        self.split = split_set
        self.dataset_config = dataset_config
        # 文本最大长度（token 数）
        self.max_des_len = args.max_des_len
        # 评估函数（用于 QA）
        self.eval_func = evaluate
        
        # ========== 初始化 tokenizer ==========
        # LLM 的 tokenizer：用于把文字转成 token id
        # add_bos_token=False 表示不自动加句首 token
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, add_bos_token=False)
        # 把 padding token 设成 eos token（很多 LLM 这么做）
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # padding 在右边（常见设置）
        self.tokenizer.padding_side = 'right'

        # Q-Former 的 tokenizer（另一个模型/模块用的 tokenizer）
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        # ========== 读取标注文件 ==========
        # split_set 只能是 train 或 val
        assert split_set in ["train", "val"]
        
        # 标注文件路径，例如：.../data/ScanQA/ScanQA_v1.0_train.json
        annotation_file = os.path.join(BASE, 'data', 'ScanQA', f'ScanQA_v1.0_{split_set}.json')
        self.annotations = json.load(open(annotation_file, 'r'))  # 读取 json 标注列表
        self._tag_dataset(self.annotations, 'qa')                 # 给每条标注加上 task_name='qa'
        
        # ========== tokenizer 的通用配置 ==========
        # max_length: 固定长度
        # padding='max_length': 不够就补 pad
        # truncation='longest_first': 太长就截断
        # return_tensors='np': 返回 numpy 格式（不是 pytorch tensor）
        self.tokenizer_config = dict(
            max_length=self.max_des_len, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")

    
    def _tag_dataset(self, corpus, task_name): 
        """
        给每条 annotation 加字段 task_name
        """
        for anno in corpus:
            anno['task_name'] = task_name
        return 
    
    def _encode_box_coords(self, annotation_mask, ret_dict):
        """
        将目标物体对应的 3D box（中心+尺寸）编码成字符串，作为 prompt 中的“位置提示”
        annotation_mask: 哪些 gt box 是目标物体（1表示是目标）
        ret_dict: _get_scan_data(scan_name) 返回的数据字典，包含 gt box 信息等
        """
        # gt_box_centers_normalized: 归一化后的 box 中心坐标 (N, 3)
        center_normalized = ret_dict['gt_box_centers_normalized']
        # gt_box_sizes_normalized: 归一化后的 box 尺寸 (N, 3)
        size_normalized = ret_dict['gt_box_sizes_normalized']

        # 拼成 (N, 6): <cx, cy, cz, w, h, l>
        box_normalized = np.hstack((center_normalized, size_normalized))    # (N, 6)

        # 只保留 mask==1 的目标 box
        box_normalized = box_normalized[annotation_mask == 1]

        # 将连续值离散到网格坐标（乘 grid_size 再取整）
        box_normalized = (box_normalized * self.grid_size_3d).astype(np.int64)

        # 按 BOX_FORMAT 格式拼成字符串，多个 box 用空格连接
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)
    
    def __len__(self):
        """
        数据集长度 = 标注条数
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        根据 idx 返回一条训练/验证样本
        这个函数会被 DataLoader 调用
        """
        # 获取场景名（scan id）
        scan_name = self.annotations[idx]['scene_id']
        # 获取任务名（这里是 qa）
        task_name = self.annotations[idx]['task_name']

        # 从父类读取该场景点云、gt box、instance 标签等信息
        ret_dict = self._get_scan_data(scan_name)
        
        # ========== 读取问题和答案 ==========
        # question 统一转小写
        question = self.annotations[idx]['question'].lower()
        # 可能有多个答案，训练时随机选一个
        answer = random.choice(self.annotations[idx]['answers'])

        # ========== 找到“问题所指的目标物体”的 box ==========
        # annotation 里给了 object_ids（可能多个）
        target_obj_id = np.asarray(self.annotations[idx]['object_ids'])

        # ret_dict["gt_object_ids"] 是每个 gt box 对应的物体 id（NUM_MAX_OBJ）
        # 这里做匹配：NUM_MAX_OBJ x nobj
        match_mask = ret_dict["gt_object_ids"][:, None] == target_obj_id[None, :]
        # 如果某个 gt box 匹配任意一个 target_obj_id，就置为 1
        match_mask = (match_mask.sum(-1) > 0).astype(np.float32)  # (NUM_MAX_OBJ,)

        # 乘上 gt_box_present，过滤掉无效 box
        match_mask = match_mask * ret_dict["gt_box_present"]

        # 把这些目标 box 编码成字符串，放到 prompt 里
        boxes = self._encode_box_coords(match_mask, ret_dict)
        
        # ========== 构建 prompt（提示词） ==========
        if self.split == 'train':
            # 训练时：从模板列表里随机选一种 prompt 风格
            prompt = deepcopy(random.choice(TASK_PROPMT[task_name]))
        else:
            # 验证时：用固定模板（第一个），并且不提供 boxes（防止泄漏）
            prompt = deepcopy(TASK_PROPMT[task_name][0])
            boxes = ''

        # 将模板里的 {locations} 和 {question} 替换成真实内容
        prompt['instruction'] = prompt['instruction'].format(locations=boxes, question=question)
        
        # instruction 编码（给 LLM 用）
        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        # instruction 编码（给 Q-Former 用）
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        
        # ========== 构建 ground truth response ==========
        # prompt['answer'] 也是模板，把 {answer} 替换成真实答案
        response = prompt['answer'].format(locations=boxes, answer=answer)

        # LLM 的训练输入一般是：instruction + response + eos
        llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((prompt['instruction'], response, self.tokenizer.eos_token))],
            **self.tokenizer_config
        )
        
        # ========== 初始化交互查询（box/click）相关数组 ==========
        # box_query: (max_prompts, 8, 3) 可能是预留给某种 box 提示/采样点
        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))

        # click_query: (max_prompts, 3) 可能表示某个点击点 (x,y,z)
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))
        
        
        # ========== 训练时 25% 概率加入 click 提示 ==========
        if self.split == 'train' and random.random() < 0.25:

            # 从目标物体里随机选一个物体 id
            target_obj_id = random.choice(self.annotations[idx]['object_ids'])
            try:
                # 从点云里取 xyz 坐标
                point_clouds = ret_dict["point_clouds"][:, :3]  # x, y, z
                # 取该物体实例对应的点（instance_labels 里通常是 id+1）
                object_points = point_clouds[ret_dict["instance_labels"] == (target_obj_id + 1)]  # (npt, 3)
                # 随机选一个点作为 click
                click_query[0] = random.choice(object_points)
            except:
                # 如果取点失败，就用该物体 box 的中心点作为 click
                match_mask = (ret_dict["gt_object_ids"] == target_obj_id).astype(np.float32)
                match_mask = match_mask * ret_dict["gt_box_present"]
                click_query[0] = ret_dict["gt_box_centers"][match_mask == 1].reshape(3,).astype(np.float32)

            # 标记第 0 个 click 是有效的
            click_mask[0] = 1
        
        # 把这些 query/mask 放进 ret_dict，供模型使用
        ret_dict['box_query'] = box_query.astype(np.float32)
        ret_dict['box_mask'] = box_mask.astype(np.float32)
        ret_dict['click_query'] = click_query.astype(np.float32)
        ret_dict['click_mask'] = click_mask.astype(np.float32)
        
        # ========== 下面这些主要用于训练 ==========
        # 完整输入序列的 token ids
        ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
        # 完整输入序列的 attention mask
        ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)

        # gradient_mask：区分 instruction 与 response
        # 通常希望只对 response 部分计算 loss/梯度
        ret_dict['gradient_mask'] = \
            (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        
        # ========== 下面这些用于训练 & 评估 ==========
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)

        # instruction 的 token ids（不包含答案）
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)

        # qformer 输入
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)
        
        return ret_dict

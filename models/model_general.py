import torch
import importlib
from torch import nn


class CaptionNet(nn.Module):
    def train(self, mode: bool = True):
        """
        重写 nn.Module 的 train()：
        - 正常情况下会切换整个模型到 train/eval 模式
        - 如果 freeze_detector=True，则强制 detector 一直处于 eval 模式，并冻结其参数不参与训练
        """
        super().train(mode)  # 调用父类的 train(mode)，切换整个模型的训练/评估模式

        # 如果选择冻结 detector（检测器）
        if self.freeze_detector is True:
            self.detector.eval()  # 强制 detector 进入 eval 模式（例如 BN/Dropout 行为固定）
            for param in self.detector.parameters():
                param.requires_grad = False  # 冻结 detector 参数，不计算梯度
        return self

    def pretrained_parameters(self):
        """
        返回需要使用预训练权重的参数列表（如果 captioner 支持的话）。
        用途常见：给预训练部分设置不同的学习率，或单独加载/优化。
        """
        if hasattr(self.captioner, 'pretrained_parameters'):
            return self.captioner.pretrained_parameters()  # 若 captioner 自己实现了该方法，则直接调用
        else:
            return []  # 否则返回空列表

    def __init__(self, args, dataset_config, train_dataset):
        """
        初始化网络：
        - 根据 args.detector 动态导入 detector 模块并实例化
        - 根据 args.captioner 动态导入 captioner 模块并实例化
        - args.freeze_detector 用于决定是否冻结 detector
        """
        super(CaptionNet, self).__init__()

        self.freeze_detector = args.freeze_detector  # 是否冻结 detector
        self.detector = None  # 检测器（例如输出候选框 proposals 等）
        self.captioner = None  # 生成描述的模块

        # 如果配置了 detector，就动态导入对应模块：models.{detector_name}.detector
        if args.detector is not None:
            detector_module = importlib.import_module(
                f'models.{args.detector}.detector'
            )
            # 约定 detector_module 内部暴露 detector() 构造函数
            self.detector = detector_module.detector(args, dataset_config)

        # 如果配置了 captioner，就动态导入对应模块：models.{captioner_name}.captioner
        if args.captioner is not None:
            captioner_module = importlib.import_module(
                f'models.{args.captioner}.captioner'
            )
            # 约定 captioner_module 内部暴露 captioner() 构造函数
            self.captioner = captioner_module.captioner(args, train_dataset)

        # 初始化后立刻调用一次 train()：
        # - 让整个网络进入 train 模式
        # - 如果 freeze_detector=True，会把 detector 强制 eval 并冻结参数
        self.train()

    def forward(self, batch_data_label: dict, is_eval: bool = False, task_name: str = None) -> dict:
        """
        前向传播：
        1) 先走 detector（如果存在），得到检测输出（比如 box_corners 等）
        2) 再走 captioner（如果存在），基于检测输出 + 输入数据生成语言描述
        3) 如果没有 captioner，就生成一个占位的默认句子
        """
        # outputs 用字典承载中间结果与 loss，先给一个默认 loss（放到 GPU 上）
        outputs = {'loss': torch.zeros(1)[0].cuda()}

        # ---------- 1) Detector 前向 ----------
        if self.detector is not None:
            if self.freeze_detector is True:
                # 冻结 detector 时：即使整体在训练，也让 detector 用 eval 的方式跑（不更新 BN/Dropout）
                outputs = self.detector(batch_data_label, is_eval=True)
            else:
                # 不冻结时：外部传入 is_eval 决定 detector 训练/评估行为
                outputs = self.detector(batch_data_label, is_eval=is_eval)

        # 如果冻结 detector，则不让 detector 的 loss 参与训练（直接清零）
        if self.freeze_detector is True:
            outputs['loss'] = torch.zeros(1)[0].cuda()

        # ---------- 2) Captioner 前向 ----------
        if self.captioner is not None:
            # captioner 通常会读取 outputs（检测结果）并结合 batch_data_label 生成 caption
            outputs = self.captioner(
                outputs,
                batch_data_label,
                is_eval=is_eval,
                task_name=task_name
            )
        else:
            # 没有 captioner 时，生成一个默认的占位 caption
            batch, nproposals, _, _ = outputs['box_corners'].shape  # box_corners 形状一般为 [B, N, ?, ?]
            outputs['lang_cap'] = [["this is a valid match!"] * nproposals] * batch  # 每个 proposal 给一句默认文本

        return outputs

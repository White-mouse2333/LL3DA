# =========================
# 训练/测试入口脚本（带分布式）
# =========================

import os
import argparse
import importlib

import numpy as np
import torch
from collections import OrderedDict  # 这里虽然导入了，但本文件里没用到（可能其他版本会用）

# 训练主循环（你项目里的 engine.py）
from engine import do_train

# 模型（你项目里的 models/model_general.py）
from models.model_general import CaptionNet

# 数据集配置（你项目里的 datasets/scannet_base_dataset.py）
from datasets.scannet_base_dataset import DatasetConfig

# 多进程启动方式（DataLoader/分布式常用）
from torch.multiprocessing import set_start_method

# 断点续训（读取 checkpoint）
from utils.io import resume_if_possible

# DataLoader 的 worker 初始化函数（常用于给每个 worker 设置随机种子等）
from utils.misc import my_worker_init_fn

# 分布式相关工具函数
from utils.dist import (
    init_distributed,
    is_distributed,
    is_primary,
    get_rank,
    barrier
)


def make_args_parser():
    """
    解析命令行参数（比如你运行 python xxx.py --batchsize_per_gpu 8）
    """
    parser = argparse.ArgumentParser(
        "LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning",
        add_help=False
    )

    # ======================
    # Optimizer / 学习率相关
    # ======================
    parser.add_argument("--base_lr", default=5e-4, type=float)       # 初始学习率
    parser.add_argument("--final_lr", default=1e-6, type=float)      # 最终学习率（配合 scheduler 用）
    parser.add_argument("--lr_scheduler", default="cosine", type=str)  # 学习率策略（例如 cosine）
    parser.add_argument("--weight_decay", default=0.1, type=float)   # 权重衰减（正则化）
    parser.add_argument("--optimizer", default="AdamW", type=str)    # 优化器类型：AdamW / SGD

    parser.add_argument(
        "--clip_gradient",
        default=0.1,
        type=float,
        help="Max L2 norm of the gradient"  # 梯度裁剪阈值，防止梯度爆炸
    )

    # dense caption 训练阶段可能用到 warmup（你注释里写了 DISABLE/ACTIVATE）
    parser.add_argument("--warm_lr", default=1e-6, type=float)       # warmup 起始学习率
    parser.add_argument("--warm_lr_epochs", default=9, type=int)     # warmup 的 epoch 数

    # 只有 dense caption training 才会启用：给预训练参数设置单独学习率
    parser.add_argument("--pretrained_params_lr", default=None, type=float)

    # 预训练权重路径（如果要加载已有模型参数）
    parser.add_argument("--pretrained_weights", default=None, type=str)

    # ==============
    # Model / 模型相关
    # ==============
    # 输入特征相关：是否使用颜色、法向量、高度、多视角等
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")   # 如果加了这个参数，就不使用高度
    parser.add_argument("--use_multiview", default=False, action="store_true")

    # detector 模块所在文件夹名（项目里通常是一个模块/目录名）
    parser.add_argument(
        "--detector",
        default="detector_Vote2Cap_DETR",
        help="folder of the detector"
    )

    # captioner 模块所在文件夹名（可选）
    parser.add_argument(
        "--captioner",
        default=None,
        type=str,
        help="folder of the captioner"
    )

    # 训练策略：冻结 detector 或冻结 LLM
    parser.add_argument(
        "--freeze_detector",
        default=False,
        action='store_true',
        help="freeze all parameters other than the caption head"
    )
    parser.add_argument(
        "--freeze_llm",
        default=False,
        action='store_true',
        help="freeze the llm for caption generation"
    )

    # 生成 caption 的解码策略相关
    parser.add_argument(
        "--use_beam_search",
        default=False,
        action='store_true',
        help='whether use beam search during caption generation.'
    )
    parser.add_argument("--max_des_len", default=128, type=int, help="maximum length of object descriptions.")
    parser.add_argument("--max_gen_len", default=32, type=int, help="maximum length of object descriptions.")

    # =================
    # Dataset / 数据集相关
    # =================
    parser.add_argument("--max_prompts", default=16, type=int, help="number of visual interactions")
    parser.add_argument("--dataset", default='scannet', help="dataset list split by ','")  # 支持多个数据集用逗号分隔
    parser.add_argument("--grid_size_3d", default=255, type=int, help="grid size of the 3D scene")

    # LLM 和 QFormer 的 tokenizer/backbone
    parser.add_argument('--vocab', default="llama-hf/7B", type=str, help="The LLM backend")
    parser.add_argument('--qformer_vocab', default="bert-base-uncased", type=str, help="The QFormer backend")

    parser.add_argument("--dataset_num_workers", default=4, type=int)   # DataLoader worker 数
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)     # 每张 GPU 的 batch size

    # ==================
    # Training / 训练流程相关
    # ==================
    parser.add_argument("--start_epoch", default=-1, type=int)          # 从哪个 epoch 开始（断点续训时会改）
    parser.add_argument("--max_epoch", default=1080, type=int)          # 最大 epoch
    parser.add_argument("--start_eval_after", default=-1, type=int)     # 多少 epoch 后开始 eval
    parser.add_argument("--eval_every_iteration", default=4000, type=int)  # 每多少 iteration 评估一次
    parser.add_argument("--seed", default=0, type=int)                  # 随机种子

    # ==================
    # Testing / 测试相关
    # ==================
    parser.add_argument("--test_only", default=False, action="store_true")  # 只测试，不训练
    parser.add_argument(
        "--test_min_iou",
        default=0.50,
        type=float,
        help='minimum iou for evaluating dense caption performance'
    )
    parser.add_argument(
        "--criterion",
        default='CiDEr',
        type=str,
        help='metrics for saving the best model'  # 保存最优模型用什么指标
    )
    parser.add_argument("--test_ckpt", default="", type=str)            # 测试用 checkpoint 路径

    # ==========
    # I/O / 保存与日志
    # ==========
    parser.add_argument("--checkpoint_dir", default=None, type=str)     # 保存 checkpoint 的目录
    parser.add_argument("--save_every", default=4000, type=int)         # 每多少 iteration 保存一次
    parser.add_argument("--log_every", default=10, type=int)            # 每多少 iteration 打印日志
    parser.add_argument("--filter_name", default='captioner.transformer.', type=str)

    # =================
    # Distributed / 分布式
    # =================
    parser.add_argument("--ngpus", default=1, type=int, help='number of gpus')  # 使用几张 GPU
    parser.add_argument("--dist_url", default='tcp://localhost:12345', type=str)  # 分布式通信地址

    args = parser.parse_args()

    # 是否使用高度特征：use_height = not no_height
    args.use_height = not args.no_height
    return args


def build_dataloader_func(args, dataset, split):
    """
    根据 split(train/test) 构建 sampler 和 dataloader
    分布式时要用 DistributedSampler，保证每个进程拿到不同的数据
    """
    if is_distributed():
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=(split == 'train')
        )
    else:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(dataset)         # 训练随机采样
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)     # 测试顺序采样

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,  # worker 初始化（一般用于固定随机种子等）
    )
    return sampler, dataloader


def build_dataset(args):
    """
    1) 构建 dataset_config（描述数据集类别/标签等配置）
    2) 根据 args.dataset 动态 import 对应数据集模块（datasets.xxx）
    3) 构建 train/val 数据集，并拼接 train（ConcatDataset）
    4) 构建 dataloaders
    """
    dataset_config = DatasetConfig()

    datasets = {'train': None, 'test': []}
    train_datasets = []

    # args.dataset 可以是 "scannet" 或 "scannet,xxx" 这种
    for dataset in args.dataset.split(','):
        # 动态导入 datasets/{dataset}.py
        dataset_module = importlib.import_module(f'datasets.{dataset}')

        # 训练集：augment=True
        train_datasets.append(
            dataset_module.Dataset(
                args,
                dataset_config,
                split_set="train",
                use_color=args.use_color,
                use_normal=args.use_normal,
                use_multiview=args.use_multiview,
                use_height=args.use_height,
                augment=True
            )
        )

        # 验证/测试集：这里用 split_set="val"，augment=False
        datasets['test'].append(
            dataset_module.Dataset(
                args,
                dataset_config,
                split_set="val",
                use_color=args.use_color,
                use_normal=args.use_normal,
                use_multiview=args.use_multiview,
                use_height=args.use_height,
                augment=False
            )
        )

    # 多个训练数据集拼起来当总训练集
    datasets['train'] = torch.utils.data.ConcatDataset(train_datasets)

    # 构建训练 dataloader
    train_sampler, train_loader = build_dataloader_func(args, datasets['train'], split='train')

    # dataloaders 字典：train / test / train_sampler
    dataloaders = {
        'train': train_loader,
        'test': [],
        'train_sampler': train_sampler,
    }

    # 构建每个测试集的 dataloader（可能有多个 test dataset）
    for dataset in datasets['test']:
        _, test_loader = build_dataloader_func(args, dataset, split='test')
        dataloaders['test'].append(test_loader)

    return dataset_config, datasets, dataloaders


def main(local_rank, args):
    """
    每个进程都会跑到这里（单卡时 local_rank=0）
    local_rank：当前进程使用的 GPU id
    """
    # ---------- 初始化分布式 ----------
    if args.ngpus > 1:
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )
        torch.cuda.set_device(local_rank)  # 每个进程绑定一张 GPU

    # ---------- 设置随机种子 ----------
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed + get_rank())  # 不同 rank 的 seed 略微偏移，避免完全一致

    # ---------- 处理 checkpoint_dir ----------
    if args.checkpoint_dir is not None:
        pass
    elif args.test_ckpt is not None:
        # 如果只给了 test_ckpt，就用它所在目录当 checkpoint_dir
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
        print(f'testing directory: {args.checkpoint_dir}')
    else:
        raise AssertionError('Either checkpoint_dir or test_ckpt should be presented!')

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------- 构建数据集和 dataloader ----------
    dataset_config, datasets, dataloaders = build_dataset(args)

    # ---------- 构建模型 ----------
    model = CaptionNet(args, dataset_config, datasets['train'])

    # ======================
    # 测试阶段（test_only）
    # ======================
    if args.test_only:
        try:
            # 加载测试 checkpoint
            checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model"], strict=False)
        except Exception:
            # 如果加载失败就用随机初始化的模型测试（一般意义不大，但代码允许）
            print('test the model from scratch...')

        # model_no_ddp：不包 DDP 的版本（方便拿参数/保存/打印等）
        model_no_ddp = model.cuda()

        # 把模型放到对应 GPU
        model = model.cuda(local_rank)

        # 如果是分布式：SyncBN + DDP
        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )

        # 对每个 test loader 做 eval
        for test_loader in dataloaders['test']:
            test_loader.dataset.eval_func(
                args, -1, model, dataset_config, test_loader
            )

    # ======================
    # 训练阶段（非 test_only）
    # ======================
    else:
        assert (args.checkpoint_dir is not None), "Please specify a checkpoint dir using --checkpoint_dir"
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # ---------- 是否加载预训练权重 ----------
        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
            model.load_state_dict(checkpoint['model'], strict=False)

            print('==== ====')            
            print('==== loading following pre-trained parameters ====')            
            print('==== ====')
            for name, param in checkpoint['model'].items():
                print('\t', name, param.shape)

        # 放到 GPU
        model_no_ddp = model.cuda()
        model = model.cuda(local_rank)

        # 分布式封装
        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )

        # ---------- 构建优化器 ----------
        if args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                filter(lambda params: params.requires_grad, model_no_ddp.parameters()),
                lr=args.base_lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                filter(lambda params: params.requires_grad, model_no_ddp.parameters()),
                lr=args.base_lr,
                weight_decay=args.weight_decay
            )
        else:
            raise NotImplementedError

        # 打印哪些参数会被训练（requires_grad=True）
        print('==== ====')
        print('==== Only training the following parameters ====')
        print('==== ====')
        for name, param in model_no_ddp.named_parameters():
            if param.requires_grad is True:
                print('\t', name, param.shape)

        # ---------- 断点续训 ----------
        # loaded_epoch：上次保存到了哪个 epoch
        # best_val_metrics：目前最好的验证指标（用于保存best模型）
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )
        args.start_epoch = loaded_epoch + 1

        # ---------- 开始训练 ----------
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(args):
    """
    根据 args.ngpus 决定单卡运行还是多卡 spawn
    """
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        # 多进程启动，每个进程跑 main(local_rank, args)
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    args = make_args_parser()

    # 屏蔽某些 multiprocessing semaphore 的 warning（不影响训练）
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    # 设置多进程启动方式为 spawn（跨平台更稳定，pytorch 推荐）
    try:
        set_start_method("spawn")
    except RuntimeError:
        # 如果已经设置过 start method，会抛 RuntimeError，忽略即可
        pass

    launch_distributed(args)

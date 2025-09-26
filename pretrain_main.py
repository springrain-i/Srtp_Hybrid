import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pretraining_dataset import PretrainingDataset
from datasets import shu_dataset
from HybridMamba import HybridMamba
from pretrain_trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=3, help='cuda number (default: 1)')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')
    parser.add_argument('--use_lr_scheduler', type=bool, default=False, help='whether to use learning rate scheduler')
    
    # 分类任务参数
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes (2 for binary, 5 for multiclass)')
    parser.add_argument('--task_type', type=str, default='multiclass', choices=['binary', 'multiclass'], 
                        help='task type: binary or multiclass')

    # parser.add_argument('--project_mode', type=str, default='cnn', help='project_mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')

    parser.add_argument('--dataset_dir', type=str, default='data/BCIC2020_datasets/processed',
                        help='dataset_dir')
    parser.add_argument('--model_dir',   type=str,   default='model_dir', help='model_dir')
    
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--multi_lr', type=bool, default=False, help='multi_lr')
    
    # 混合模式参数
    parser.add_argument('--hybrid_mode', type=str, default='mamba_first', 
                        choices=['alternate', 'mamba_first', 'attention_first', 'all_mamba', 'all_attention'],
                        help='hybrid mode for mixing Mamba and Attention layers')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs for mamba layers')
    parser.add_argument('--custom_depths', type=str, default=None, 
                        help='custom layer depths in format "3,3,4,3" (comma-separated)')
    parser.add_argument('--stage_types',type=str, default='mamba,mamba,atten,atten',help='stage types for each stage, comma-separated, e.g., "mamba,mamba,atten,atten"')
    
    # 后边调整
    parser.add_argument('--in_chans', type=int, default=32, help='in_chans')
    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    print(len(pretrained_dataset))
    load_dataset = shu_dataset.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    # 处理自定义层数和stage类型
    custom_depths = None
    stage_types = None
    
    if params.custom_depths is not None:
        custom_depths = [int(x.strip()) for x in params.custom_depths.split(',')]
        print(f"Using custom depths: {custom_depths}")
    
    if params.stage_types is not None:
        stage_types = [x.strip() for x in params.stage_types.split(',')]
        print(f"Using stage types: {stage_types}")
    
    model = HybridMamba(
        in_chans=params.in_chans,
        patch_size = params.in_dim,
        out_dim = params.out_dim,
        d_model = params.d_model,
        drop_rate = params.dropout,
        hybrid_mode = params.hybrid_mode,
        custom_depths = custom_depths,
        stage_types = stage_types,
        num_classes = params.num_classes,
    )
    trainer = Trainer(params, data_loader, model)
    if params.task_type == 'binary':
        trainer.train_for_binaryclass()
    else:
        trainer.train_for_multiclass()
    pretrained_dataset.db.close()


if __name__ == '__main__':
    main()

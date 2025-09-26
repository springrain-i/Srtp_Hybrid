import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        
        # 根据模型类别数选择损失函数
        if hasattr(model, 'num_classes') and model.num_classes > 2:
            self.criterion = CrossEntropyLoss().cuda()
            self.is_multiclass = True
        else:
            self.criterion = BCEWithLogitsLoss().cuda()
            self.is_multiclass = False

        self.best_model_states = None

        # 为不同类型的层设置不同的学习率
        mamba_params = []
        attention_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                if hasattr(params, 'frozen') and params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            
            # 根据参数名称分组
            if 'mixer' in name and 'MambaVisionMixer' in str(type(self.model)):
                mamba_params.append(param)
            elif 'mixer' in name and 'Attention' in str(type(self.model)):
                attention_params.append(param)
            elif 'blocks' in name:
                # 检查是否是mamba相关参数
                if any(mamba_key in name for mamba_key in ['ssm', 'conv1d', 'dt_proj', 'x_proj', 'A_log', 'D']):
                    mamba_params.append(param)
                elif any(attn_key in name for attn_key in ['qkv', 'proj', 'attn']):
                    attention_params.append(param)
                else:
                    other_params.append(param)
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            # 为不同类型的层设置更保守的学习率
            param_groups = []
            if mamba_params:
                param_groups.append({'params': mamba_params, 'lr': self.params.lr * 1.0})  # Mamba层使用标准学习率
            if attention_params:
                param_groups.append({'params': attention_params, 'lr': self.params.lr * 1.0})  # Attention层使用标准学习率
            if other_params:
                param_groups.append({'params': other_params, 'lr': self.params.lr * 1.2})  # 其他层稍微提高学习率
            
            if param_groups:
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            # SGD优化器的类似处理
            param_groups = []
            if mamba_params:
                param_groups.append({'params': mamba_params, 'lr': self.params.lr * 1.0})
            if attention_params:
                param_groups.append({'params': attention_params, 'lr': self.params.lr * 1.0})
            if other_params:
                param_groups.append({'params': other_params, 'lr': self.params.lr * 1.2})
            
            if param_groups:
                self.optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        if hasattr(self.params, 'use_lr_scheduler') and self.params.use_lr_scheduler:
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
            )
        else:
            self.optimizer_scheduler = None
        print(self.model)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                if self.optimizer_scheduler is not None:
                    self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_multiclass(self):
        """多分类训练方法"""
        acc_best = 0
        f1_best = 0
        cm_best = None
        
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda().long()  # 多分类需要long类型标签
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                if self.optimizer_scheduler is not None:
                    self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1_weighted, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1_weighted: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1_weighted,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                
                if acc > acc_best:
                    print("Accuracy increasing....saving weights !!")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1_weighted: {:.5f}".format(
                        acc, kappa, f1_weighted
                    ))
                    best_epoch = epoch + 1
                    acc_best = acc
                    f1_best = f1_weighted
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        
        # 加载最佳模型进行测试
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1_weighted, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1_weighted: {:.5f}".format(
                    acc, kappa, f1_weighted
                )
            )
            print(cm)
            
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_f1_{:.5f}.pth".format(best_epoch, acc, f1_weighted)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

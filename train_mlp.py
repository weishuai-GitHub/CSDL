import argparse
import sys,os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader,Sampler,WeightedRandomSampler
from tqdm import tqdm
from typing import Iterator,Sequence
import numpy as np

from util.cluster_and_log_utils import log_accs_from_preds

#from sklearn.metrics.cluster import _supervised as supervised
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.cluster import split_cluster_nmi, split_cluster_ari,split_cluster_acc

from util.general_utils import AverageMeter, init_experiment
from config import exp_root
from models.loss import  DistillLoss, ContrastiveLearningViewGenerator, SupConLoss,get_params_groups, info_nce_logits,renyi_loss,sup_renyi_loss
from models.model import SENet,DINOHead,MLP

class Mysampler(Sampler[int]):
    weights: torch.Tensor
    num_samples: int
    replacement: bool
    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator).tolist()
    def __iter__(self) -> Iterator[int]:
        #rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(self.rand_tensor)
    def __len__(self) -> int:
        return self.num_samples
    def set_randTensor(self, tensor: torch.Tensor):
        self.rand_tensor = tensor.tolist()

def train(student:SENet, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    # inductive
    best_test_acc_ubl = 0
    # transductive
    best_train_acc_lab = 0
    best_train_acc_ubl = 0 
    best_train_acc_all = 0

    iters_per_epoch = student.base_N//args.chunk_size
    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                        beta=args.beta,
                    )
    
    for epoch in range(args.begin,args.epochs):
        loss_record = AverageMeter()
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            images = torch.cat(images, dim=0).cuda(non_blocking=True)
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                feaX,coff_c,student_proj,logits = student(images)
                logits_out = logits.detach()
                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (logits / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                sup_logits = sup_logits.softmax(dim=1)
                sup_labels = torch.nn.functional.one_hot(sup_labels, args.num_subspaces).float()
                cls_loss = renyi_loss(sup_labels, sup_logits) + args.beta*renyi_loss(1-sup_labels, 1-sup_logits)
                # cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(logits, logits_out, epoch)
                avg_probs = (logits / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # self_express_loss
                base = student.base
                coff_X = coff_c.mm(base)
                self_express_loss = torch.sum(torch.pow((coff_X - feaX)/(2*feaX.shape[0]), 2))

                pstr = ''
                pstr +=f'self_express_loss: {self_express_loss.item():.4f} '
                pstr +=f'cls_loss: {cls_loss.item():.4f} '
                pstr +=f'cluster_loss: {cluster_loss.item():.4f} '

                loss = 0
                loss += (1 - args.sup_weight) * cls_loss + args.sup_weight * cluster_loss
                loss += self_express_loss
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(student.parameters(),10,2)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(student.parameters(), 10,2)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr)) 
            if epoch<2 and batch_idx % 100 == 0:
                with torch.no_grad():
                    for i in range(iters_per_epoch):
                        chunk_data = samples[i*200:(i+1)*200]
                        chunk_data = chunk_data.to(device)
                        chunk_data = model.backbone(chunk_data)
                        model.base[i*200:(i+1)*200,:] = chunk_data   
        # Step schedule
        exp_lr_scheduler.step()
        torch.cuda.empty_cache()
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        if ( epoch + 1) % 2==0:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_eva, old_eva, new_eva = test(model=student, test_loader=unlabelled_train_loader,epoch=epoch, 
                                            save_name='Train ACC Unlabelled',args = args)
            args.logger.info('Train all: acc {:.4f} | mni {:.4f} | ari {:.4f}'.format(all_eva[0], all_eva[1], all_eva[2]))
            args.logger.info('Train old: acc {:.4f} | mni {:.4f} | ari {:.4f}'.format(old_eva[0], old_eva[1], old_eva[2]))
            args.logger.info('Train new: acc {:.4f} | mni {:.4f} | ari {:.4f}'.format(new_eva[0], new_eva[1], new_eva[2]))
            save_dict = {
                'model': student.state_dict(),
                'base':student.base,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(save_dict, args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))

            if new_eva[0] > best_test_acc_ubl:

                # args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_eva_test[0]:.4f}...')
                args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_eva[0], old_eva[0], new_eva[0]))

                torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                # inductive
                best_test_acc_ubl = new_eva[0]
                # transductive            
                best_train_acc_lab = old_eva[0]
                best_train_acc_ubl = new_eva[0]
                best_train_acc_all = all_eva[0]

            args.logger.info(f'Exp Name: {args.exp_name}')
            args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')
        with torch.no_grad():
            for i in range(iters_per_epoch):
                chunk_data = samples[i*args.chunk_size:(i+1)*args.chunk_size]
                chunk_data = chunk_data.to(device)
                chunk_data = student.backbone(chunk_data)
                student.base[i*args.chunk_size:(i+1)*args.chunk_size,:] = chunk_data
                student.base.requires_grad=False 

def test(model, test_loader,epoch, save_name, args):
    model.eval()
    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(images,is_train=False)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mask = np.bool8(mask)
    all_acc,old_acc,new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    all_nmi,old_nmi,new_nmi = split_cluster_nmi(targets, preds,mask)
    all_ari,old_ari,new_ari = split_cluster_ari(targets, preds,mask)
    all_eva = [all_acc,all_nmi,all_ari]
    old_eva = [old_acc,old_nmi,old_ari]
    new_eva = [new_acc,new_nmi,new_ari]
    return all_eva,old_eva,new_eva

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--mlp_out_dim',default=2000,type=int)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--alpha',type=float,default=2.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default='./logs_mlp')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--self_express_weight', type=float, default=1.0)
    parser.add_argument('--constrative_subspace_weight', type=float, default=5.0)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--beta',default=1.0,type=float)

    parser.add_argument('--temperature', default=0.1, type=float, help='Number of warmup epochs for the self-express temperature.')
    parser.add_argument('--is_head', default=False, type=bool, help='Whether to use head or not.')
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default="_simgcd", type=str)

    parser.add_argument('--non_zeros', type=int, default=30)
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--n_neighbors', type=int, default=20)
    parser.add_argument('--spectral_dim', type=int, default=26)
    parser.add_argument('--affinity', type=str, default="nearest_neighbor")

    # ----------------------
    # INIT
    # ----------------------
    
    device = torch.device('cuda:0')
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    args = parser.parse_args()
    args.exp_name = f'{args.dataset_name}'+args.exp_name
    
    args.device = device
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    args.logger.info(f'the base size is {args.mlp_out_dim}')
    args.logger.info(f'the alpha is {args.alpha}')
    args.logger.info(f'the beta is {args.beta}')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.bottleneck_dim = 256

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',map_location='cpu')

    # if args.warmup_model_dir is not None:
    #     args.logger.info(f'Loading weights from {args.warmup_model_dir}')
    #     backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=512, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=512, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    query_embedding = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, 
                               nlayers=args.num_mlp_layers,bottleneck_dim=args.bottleneck_dim)
    key_embedding  = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, 
                              nlayers=args.num_mlp_layers,bottleneck_dim=args.bottleneck_dim)
    if args.is_head:
        cls_head = MLP(in_dim=args.bottleneck_dim, out_dim=args.num_subspaces, nlayers=args.num_mlp_layers)
    else:
        cls_head = MLP(in_dim=args.feat_dim, out_dim=args.num_subspaces, nlayers=args.num_mlp_layers)
    sampled_idx = np.random.choice(len(train_dataset), args.mlp_out_dim, replace=False)
    samples=[]
    samples_labels = []
    model = SENet(backbone,query_embedding,key_embedding,cls_head,args)
    for idx in sampled_idx:
        sample,samples_label,_ = datasets['test_train'][idx]
        samples_labels.append(samples_label)
        samples.append(sample)
    samples = torch.stack(samples)
    model.samples = samples
    if args.warmup_model_dir is not None:
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict['model'])
        model.to(device)
        model.base = state_dict['base'].to(device)
        args.begin = state_dict['epoch']
        
    else:
        model.to(device)
        args.begin  =0
        if(model.base_N <= args.chunk_size):
            args.chunk_size = model.base_N
        if (model.base_N % args.chunk_size != 0):
            raise Exception("chunk_size should be a factor of base size.")
        iters_per_epoch = model.base_N // args.chunk_size
        with torch.no_grad():
            for i in range(iters_per_epoch):
                chunk_data = samples[i*args.chunk_size:(i+1)*args.chunk_size]
                chunk_data = chunk_data.to(device)
                chunk_data = model.backbone(chunk_data)
                model.base[i*args.chunk_size:(i+1)*args.chunk_size,:] = chunk_data
    # model.base.requires_grad=True
    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, test_loader_unlabelled,args)

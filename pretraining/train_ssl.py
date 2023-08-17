# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Created By    : Subhajit Maity
# Created Date  : 01 Feb 2023
# version       : 3.6
# ---------------------------------------------------------------------------

import copy
import os
import argparse
import time
import re
import pandas as pd
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import lightly
from lightly.models.modules import heads
from lightly.models import utils
from lightly.utils import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.optimizer import Optimizer, required

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SelfDocSeg')

    # parser.add_argument('--base_dir', type=str, default=os.getcwd(), help='the root directory to the experiments')

    parser.add_argument('--num_eval_classes', type=int, default=16, help='number of classes in linear evaluation train/test dataset')
    parser.add_argument('--dataset_name', type=str, default='DocLayNet', help='name of pre-training dataset')
    parser.add_argument('--knn_train_root', type=str, help='path to the training dataset for linear evaluation')
    parser.add_argument('--knn_eval_root', type=str, help='path to the evaluation dataset for linear evaluation')
    parser.add_argument('--dataset_root', type=str, help='path to the training dataset image directory for self-supervised training')
    parser.add_argument('--logs_root', type=str, default=os.path.join(os.getcwd(), 'benchmark_logs'), help='path to save logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
    
    parser.add_argument('--num_workers', type=int, default=0, help='number of threads in dataloader')
    parser.add_argument('--max_epochs', type=int, default=800, help='number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize to use in dataloader')
    parser.add_argument('--n_runs', type=int, default=1, help='number of times to run training for reporting mean linear evaluation performance')

    parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate to use with optimizer LARS')
    parser.add_argument('--lr_decay', type=float, default=5e-4, help='learning rate decay to use in the scheduler')
    parser.add_argument('--wt_momentum', type=float, default=0.99, help='momentum to use for EMA updates')

    parser.add_argument('--bin_threshold', type=int, default=239, help='threshold value for binarization')
    parser.add_argument('--kernel_shape', type=str, default='rect', help='shape of kernel for erosion')
    parser.add_argument('--kernel_size', type=int, default=3, help='size of kernel for erosion')
    parser.add_argument('--kernel_iter', type=int, default=2, help='number of times to apply kernel for erosion')

    parser.add_argument('--eeta', type=float, default=0.001, help='small constant to avoid unfavourable calculations')
    parser.add_argument('--alpha', type=float, default=1.0, help='weightage of the mask loss')
    parser.add_argument('--beta', type=float, default=1.0, help='weightage of the similarity loss')

    parser.add_argument('--distributed', action='store_true', default=False, help='use more than one GPU')
    parser.add_argument('--sync_batchnorm', action='store_true', default=False, help='synchronize batchnorm across GPUs')
    parser.add_argument('--gather_distributed', action='store_true', default=False, help='accumulate features from GPUs before loss calculation')

    args = parser.parse_args()

    logs_root_dir = args.logs_root

    num_workers = args.num_workers                                                      # number of threads for dataloaders
    max_epochs = args.max_epochs                                                        # maximum epochs to train

    # The dataset structure should be like this:
    path_to_train = args.knn_train_root                                                 # path to the root directory of training dataset of RVL-CDIP    '/path/to/train/'
    path_to_test = args.knn_eval_root                                                   # path to the root directory of evaluation dataset of RVL-CDIP  '/path/to/test/'
    classes = args.num_eval_classes                                                     # 16 classes for RVL-CDIP dataset for linear evaluation
    ssl_dataset_path = args.dataset_root                                                # path to the self-supervised dataset image directory           '/path/to/dataset/images/'

    ssl_dataset_name = args.dataset_name

    resume_ckpt = args.resume                                                           # path to the checkpoint for resuming training

    EETA_DEFAULT = args.eeta                                                            # small constant required for avoiding unfavourable calculations
    alpha = args.alpha                                                                  # weightage for the mask loss (focal loss)
    beta = args.beta                                                                    # weightage for the similatrity loss (nehgative cosine similarity loss)

    bin_threshold = args.bin_threshold                                                  # threshold for binarization
    kernel_shape = args.kernel_shape                                                    # kernel shape for erosion
    kernel_size = (args.kernel_size,args.kernel_size)                                   # kernel size for erosion
    kernel_iter = args.kernel_iter                                                      # iterations for erosion

    # benchmark
    n_runs = args.n_runs                                                                # optional, increase to create multiple runs and report mean + std
    batch_size = args.batchsize                                                         # batch size, 8 is max for 24GB gpu using approximately 16GB

    learning_rate = args.learning_rate                                                  # learning rate
    lr_decay = args.lr_decay                                                            # weight decay for the learning rate
    wt_momentum = args.wt_momentum                                                      # momentum for updating the weights in momentum branch

    # use a GPU if available
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Set to True to enable Distributed Data Parallel training.
    distributed = args.distributed                                                      # change to True to use more than one GPU

    # Set to True to enable Synchronized Batch Norm (requires distributed=True). 
    # If enabled the batch norm is calculated over all gpus, otherwise the batch
    # norm is only calculated from samples on the same gpu.
    sync_batchnorm = args.sync_batchnorm                                                # change to True to syncronize the batchnorm across GPUs

    # Set to True to gather features from all gpus before calculating 
    # the loss (requires distributed=True).
    # If enabled then the loss on every gpu is calculated with features from all 
    # gpus, otherwise only features from the same gpu are used.
    gather_distributed = args.gather_distributed                                        # change to True to accumulate features from all GPUs before loss calculation

    if distributed:
        distributed_backend = 'ddp'
        # reduce batch size for distributed training
        batch_size = batch_size // gpus
    else:
        distributed_backend = None
        # limit to single gpu if not using distributed training
        gpus = min(gpus, 1)

    class LARS(Optimizer):
        """
        Layer-wise Adaptive Rate Scaling for large batch training.
        Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
        I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
        """

        def __init__(
            self,
            params,
            lr=required,
            momentum=0.9,
            use_nesterov=False,
            weight_decay=0.0,
            exclude_from_weight_decay=None,
            exclude_from_layer_adaptation=None,
            classic_momentum=True,
            eeta=EETA_DEFAULT
        ):
            """Constructs a LARSOptimizer.
            Args:
            param_names: names of parameters of model obtained by
                [name for name, p in model.named_parameters() if p.requires_grad]
            lr: A `float` for learning rate.
            momentum: A `float` for momentum.
            use_nesterov: A 'Boolean' for whether to use nesterov momentum.
            weight_decay: A `float` for weight decay.
            exclude_from_weight_decay: A list of `string` for variable screening, if
                any of the string appears in a variable's name, the variable will be
                excluded for computing weight decay. For example, one could specify
                the list like ['bn', 'bias'] to exclude BN and bias
                from weight decay.
            exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
                for layer adaptation. If it is None, it will be defaulted the same as
                exclude_from_weight_decay.
            classic_momentum: A `boolean` for whether to use classic (or popular)
                momentum. The learning rate is applied during momeuntum update in
                classic momentum, but after momentum for popular momentum.
            eeta: A `float` for scaling of learning rate when computing trust ratio.
            name: The name for the scope.
            """

            self.epoch = 0
            defaults = dict(
                lr=lr,
                momentum=momentum,
                use_nesterov=use_nesterov,
                weight_decay=weight_decay,
                exclude_from_weight_decay=exclude_from_weight_decay,
                exclude_from_layer_adaptation=exclude_from_layer_adaptation,
                classic_momentum=classic_momentum,
                eeta=eeta
            )

            super(LARS, self).__init__(params, defaults)
            self.lr = lr
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.use_nesterov = use_nesterov
            self.classic_momentum = classic_momentum
            self.eeta = eeta
            self.exclude_from_weight_decay = exclude_from_weight_decay
            # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
            # arg is None.
            if exclude_from_layer_adaptation:
                self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
            else:
                self.exclude_from_layer_adaptation = exclude_from_weight_decay

            self.param_name_map = {'batch_normalization':'bn','bias':'bias'}

        def step(self, epoch=None, closure=None):
            loss = None
            if closure is not None:
                loss = closure()

            if epoch is None:
                epoch = self.epoch
                self.epoch += 1

            for group in self.param_groups:
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                eeta = group["eeta"]
                lr = group["lr"]
                #print(lr)
                #param_names = group["param_names"]

                for p_name, p in zip(group["param_names"],group["params"]):
                    if p.grad is None:
                        continue

                    param = p.data
                    grad = p.grad.data

                    param_state = self.state[p]

                    # TODO: get param names
                    if self._use_weight_decay(p_name):
                        grad += self.weight_decay * param
                    #else:
                    #    print(p_name)

                    if self.classic_momentum:
                        trust_ratio = 1.0

                        # TODO: get param names
                        if self._do_layer_adaptation(p_name):
                            w_norm = torch.norm(param)
                            g_norm = torch.norm(grad)

                            device = g_norm.get_device()
                            trust_ratio = torch.where(
                                w_norm.gt(0),
                                torch.where(
                                    g_norm.gt(0),
                                    (self.eeta * w_norm / g_norm),
                                    torch.Tensor([1.0]).to(device),
                                ),
                                torch.Tensor([1.0]).to(device),
                            ).item()

                        scaled_lr = lr * trust_ratio
                        if "momentum_buffer" not in param_state:
                            next_v = param_state["momentum_buffer"] = torch.zeros_like(
                                p.data
                            )
                        else:
                            next_v = param_state["momentum_buffer"]

                        next_v.mul_(momentum).add_(grad, alpha = scaled_lr)
                        if self.use_nesterov:
                            update = (self.momentum * next_v) + (scaled_lr * grad)
                        else:
                            update = next_v

                        p.data.add_(-update)
                    else:
                        trust_ratio = 1.0

                        if "momentum_buffer" not in param_state:
                            next_v = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        else:
                            next_v = param_state["momentum_buffer"]

                        next_v.mul_(momentum).add_(grad)
                        if self.use_nesterov:
                            update = (self.momentum * next_v) + grad
                        else:
                            update = next_v

                        if self._do_layer_adaptation(p_name):
                            w_norm = torch.norm(param)
                            g_norm = torch.norm(grad)

                            device = g_norm.get_device()
                            trust_ratio = torch.where(
                                w_norm.gt(0),
                                torch.where(
                                    g_norm.gt(0),
                                    (self.eeta * w_norm / g_norm),
                                    torch.Tensor([1.0]).to(device),
                                ),
                                torch.Tensor([1.0]).to(device),
                            ).item()

                        scaled_lr = lr * trust_ratio

                        p.data.add_(-update, alpha = scaled_lr)

            return loss

        def _use_weight_decay(self, param_name):
            """Whether to use L2 weight decay for `param_name`."""
            if not self.weight_decay:
                return False
            if self.exclude_from_weight_decay:
                for r in self.exclude_from_weight_decay:
                    if re.search(self.param_name_map[r], param_name) is not None:
                        return False
            return True

        def _do_layer_adaptation(self, param_name):
            """Whether to do layer-wise learning rate adaptation for `param_name`."""
            if self.exclude_from_layer_adaptation:
                for r in self.exclude_from_layer_adaptation:
                    if re.search(self.param_name_map[r], param_name) is not None:
                        return False
            return True

    class BYOLTransform:
        def __init__(self, h, w):
            self.h = h
            self.w = w
            l = h if h>= w else w
            self.l = l
            self.transform = transforms.Compose([
                # transforms.RandomResizedCrop(self.l, (0.2,1.0)),                      # disabling Random Crop
                transforms.Resize((self.h, self.w)),
                # transforms.RandomHorizontalFlip(p=0.5),                               # disabling Horizontal Flip
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(l//10) if int(l//10)%2!=0 else int(l//10)+1,
                                                                sigma=(0.8,2.0))],
                                    p=1.0 if self.l > 32 else 0.0),
                transforms.RandomSolarize(threshold = 0.5, p=0.0)
            ])
            self.transform_prime = transforms.Compose([
                # transforms.RandomResizedCrop(self.l, (0.8,1.0)),                      # disabling Random Crop
                transforms.Resize((self.h, self.w)),
                # transforms.RandomHorizontalFlip(p=0.5),                               # disabling Horizontal Flip
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(l//10) if int(l//10)%2!=0 else int(l//10)+1,
                                                                sigma=(0.8,2.0))],
                                    p=0.1),
                transforms.RandomSolarize(threshold = 0.5, p=0.2)
            ])
        def __call__(self, x):
            x1 = self.transform(x)
            x2 = self.transform_prime(x)
            return x1, x2

    class LayNet_SSL_dataset(torch.utils.data.Dataset):                                 # SSL Dataset class
        def __init__(self, dataset_path : str) -> None:
            self.dataset_path = dataset_path
            self.img_path = self.dataset_path
            self.imglist = os.listdir(self.img_path)
            print('Total Images : ', len(self.imglist))
            self.transform = BYOLTransform(640, 512)                                    # 320x256 coverts to 10x8 in resnet50 (32x shrink) and 640x512 coverts to 20x16 in resnet50 (32x shrink)
        
        def __getitem__(self, index):
            imgname = self.imglist[index]                                               # getting image file name
            img_fp = os.path.join(self.img_path, imgname)

            img = cv2.imread(img_fp)                                                    # reading image
            m = torch.from_numpy(self.process(img=img, size=(64,80)))                   # mask generation
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                  # colour channel conversion

            img = F.to_tensor(img)                                                      # converting to PyTorch tensors
            p1, p2 = self.transform(img)                                                # getting two augmented versions
            x1 = F.normalize(p1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            x2 = F.normalize(p2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
            x = (x1, x2, m)
            y = -1                                                                      # assigning label as ignore index, y shall not be used during pre-training

            return x, y, imgname
        
        def __len__(self):
            return len(self.imglist)

        def process(self, img, size=(64,80), thresh=bin_threshold, kernel_size=kernel_size, it=kernel_iter, kernel_type=kernel_shape):
            if kernel_type == 'rect':
                kernel_morph = cv2.MORPH_RECT
            elif kernel_type == 'ellipse':
                kernel_morph = cv2.MORPH_ELLIPSE
            elif kernel_type == 'cross':
                kernel_morph = cv2.MORPH_CROSS
            else:
                print('Invalid kernel type. Using Rectangular Kernel...')
                kernel_morph = cv2.MORPH_RECT

            img = cv2.resize(img, (512,640))                                            # resizing image
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                            # grayscale conversion
            _, img_bin = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)        # binary thresholding

            kernel = cv2.getStructuringElement(kernel_morph, kernel_size)               # building kernel, default rectangular
            mod_img = cv2.erode(img_bin, kernel, iterations=it)                         # performing erosion

            return cv2.resize((255.0 - mod_img), size, interpolation=cv2.INTER_NEAREST) # performing inversion

    normalize_transform = torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )

    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(640, 512)),
        normalize_transform,
    ])

    dataset_train_ssl = LayNet_SSL_dataset(dataset_path=ssl_dataset_path)

    # we use test transformations for getting the feature for kNN on train data
    dataset_train_kNN = lightly.data.LightlyDataset(
        input_dir=path_to_train,
        transform=test_transforms
    )

    dataset_test = lightly.data.LightlyDataset(
        input_dir=path_to_test,
        transform=test_transforms
    )

    def get_data_loaders(batch_size: int, model):
        """
        Helper method to create dataloaders for ssl, kNN train and kNN test
        Args:
            batch_size: Desired batch size for all dataloaders
        """
        dataloader_train_ssl = torch.utils.data.DataLoader(
            dataset_train_ssl,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers
        )

        dataloader_train_kNN = torch.utils.data.DataLoader(
            dataset_train_kNN,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        return dataloader_train_ssl, dataloader_train_kNN, dataloader_test

    class SelfDocSeg(BenchmarkModule):
        def __init__(self, dataloader_kNN, num_classes):
            super().__init__(dataloader_kNN, num_classes)

            # create a ResNet backbone and remove the classification head
            resnet = torchvision.models.resnet50(weights=None)

            feature_dim = list(resnet.children())[-1].in_features
            self.feature_dim = feature_dim

            # backbone feature map extractor
            self.bb_feat = nn.Sequential(
                *list(resnet.children())[:-2],
                # nn.AdaptiveAvgPool2d(1)
                nn.Upsample(scale_factor=4, mode='nearest')
            )

            # backbone to pooled encoding extractor
            self.bb_vec = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.AdaptiveAvgPool2d(1)
            )

            # full backbone encoder, required for linear evaluation
            self.backbone = nn.Sequential(self.bb_feat, self.bb_vec)

            # projector and predictor
            self.projection_head = heads.BYOLProjectionHead(feature_dim, 4096, 256)
            self.prediction_head = heads.BYOLPredictionHead(256, 4096, 256)

            # mask predictor module
            self.mask_localizer = nn.Sequential(nn.Conv2d(in_channels=2*feature_dim, out_channels=1, kernel_size=1), nn.Sigmoid())

            # momentum branch
            self.bb_feat_momentum = copy.deepcopy(self.bb_feat)
            self.bb_vec_momentum = copy.deepcopy(self.bb_vec)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)

            # deactivating gradients in moemntum branch
            utils.deactivate_requires_grad(self.bb_feat_momentum)
            utils.deactivate_requires_grad(self.bb_vec_momentum)
            utils.deactivate_requires_grad(self.projection_head_momentum)

            # setting up loss functions
            self.criterion = lightly.loss.NegativeCosineSimilarity()
            self.mask_criterion = torchvision.ops.sigmoid_focal_loss

        def forward(self, x, train=False):
            if train:
                x_feat = self.bb_feat(x)
                return x_feat
            else:
                x_feat = self.bb_feat(x)
                y = self.bb_vec(x_feat).flatten(start_dim=1)
                z = self.projection_head(y)
                p = self.prediction_head(z)
                return p

        def forward_momentum(self, x, train=False):
            if train:
                x_feat = self.bb_feat_momentum(x)
                return x_feat.detach()
            else:
                x_feat = self.bb_feat_momentum(x)
                y = self.bb_vec_momentum(x_feat).flatten(start_dim=1)
                z = self.projection_head_momentum(y)
                z = z.detach()
                return z

        def training_step(self, batch, batch_idx):
            # update weights in momentum branch
            utils.update_momentum(self.bb_feat, self.bb_feat_momentum, m=wt_momentum)
            utils.update_momentum(self.bb_vec, self.bb_vec_momentum, m=wt_momentum)
            utils.update_momentum(self.projection_head, self.projection_head_momentum, m=wt_momentum)
            
            (x0, x1, m), _, _ = batch

            # extract feature maps
            x0_feats = self.forward(x0, train=True)
            x0_feats_m = self.forward_momentum(x0, train=True)
            x1_feats = self.forward(x1, train=True)
            x1_feats_m = self.forward_momentum(x1, train=True)

            # predict masks from layout prediction module
            m_pred = self.mask_localizer(torch.cat((x0_feats, x1_feats), dim=1))
            mask_loss = self.mask_criterion(m_pred, m.unsqueeze(1)/((torch.max(m)).to(torch.float)), alpha=0.25, gamma=2, reduction='mean')

            # initialize empty features for each layout object
            y0 = torch.empty(size=(0, self.feature_dim), requires_grad=True).to(device=torch.device('cuda'))
            y1 = torch.empty(size=(0, self.feature_dim), requires_grad=True).to(device=torch.device('cuda'))
            y0_m = torch.empty(size=(0, self.feature_dim), requires_grad=True).to(device=torch.device('cuda'))
            y1_m = torch.empty(size=(0, self.feature_dim), requires_grad=True).to(device=torch.device('cuda'))

            # iterate across batch
            for idx, ymask in enumerate(copy.deepcopy(m).detach().cpu().numpy().astype('uint8')):
                
                # iterate across every layout object mask
                cmasks, _ = cv2.findContours(ymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cmask in cmasks:
                    mask = np.zeros(ymask.shape, np.uint8)
                    cv2.fillPoly(mask, pts=[cmask], color=(255))
                    
                    # creating binary mask of layout objects
                    mask = (torch.from_numpy(mask).to(torch.float).to(device=torch.device('cuda')))/255.0

                    # performing mask pooling operation

                    y0_feat = torch.mul(x0_feats[idx], mask).sum(dim=(-1, -2))/mask.sum()
                    y0 = torch.cat((y0, y0_feat.unsqueeze(0)))

                    y1_feat = torch.mul(x1_feats[idx], mask).sum(dim=(-1, -2))/mask.sum()
                    y1 = torch.cat((y1, y1_feat.unsqueeze(0)))

                    y0_feat_m = torch.mul(x0_feats_m[idx], mask).sum(dim=(-1, -2))/mask.sum()
                    y0_m = torch.cat((y0_m, y0_feat_m.unsqueeze(0)))

                    y1_feat_m = torch.mul(x1_feats_m[idx], mask).sum(dim=(-1, -2))/mask.sum()
                    y1_m = torch.cat((y1_m, y1_feat_m.unsqueeze(0)))

            # passing through projectors and predictors in online and momentum branches
            z0 = self.projection_head(y0)
            p0 = self.prediction_head(z0)

            z0_m = self.projection_head_momentum(y0_m)

            z1 = self.projection_head(y1)
            p1 = self.prediction_head(z1)

            z1_m = self.projection_head_momentum(y1_m)

            # calculating loss
            loss = alpha*mask_loss + beta*0.5*(self.criterion(p0, z1_m) + self.criterion(p1, z0_m))

            # loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
            self.log('train_loss_ssl', loss)
            return loss

        def configure_optimizers(self):
            params_dict = {**dict(self.bb_feat.named_parameters()),
                **dict(self.bb_vec.named_parameters()),
                **dict(self.mask_localizer.named_parameters()),
                **dict(self.projection_head.named_parameters()),
                **dict(self.prediction_head.named_parameters())}

            params = [{'params' : list(params_dict.values()), 'param_names' : list(params_dict.keys())}]
            optim = LARS(
                params, 
                lr=learning_rate,
                # exclude_from_weight_decay=["batch_normalization", "bias"], 
                weight_decay=lr_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
            return [optim], [scheduler]


    models = [
        SelfDocSeg,
    ]
    bench_results = dict()

    experiment_version = None
    # loop through configurations and train models
    for BenchmarkModel in models:
        runs = []
        model_name = BenchmarkModel.__name__.replace('Model', '')
        for seed in range(n_runs):
            pl.seed_everything(seed)
            dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
                batch_size=batch_size, 
                model=BenchmarkModel,
            )
            benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

            if resume_ckpt is not None:                                                 # Resuming ckpt option
                benchmark_model.load_state_dict(torch.load(resume_ckpt, map_location=benchmark_model.dummy_param.device)['state_dict'])
                print('Resuming from ', resume_ckpt)

            # Save logs to: {CWD}/benchmark_logs/{ssl_dataset_name}/{experiment_version}/{model_name}/
            # If multiple runs are specified a subdirectory for each run is created.
            sub_dir = model_name if n_runs <= 1 else f'{model_name}/run{seed}'
            logger = TensorBoardLogger(
                save_dir=os.path.join(logs_root_dir, ssl_dataset_name),
                name='',
                sub_dir=sub_dir,
                version=experiment_version,
            )
            if experiment_version is None:
                # Save results of all models under same version directory
                experiment_version = logger.version
            
            # checkpoint is saved for the best linear evaluation accuracy and the last epoch
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(logger.log_dir, 'checkpoints'),
                save_on_train_epoch_end=True
            )
            
            # building PyTorch Lightning trainer
            trainer = pl.Trainer(
                max_epochs=max_epochs, 
                gpus=gpus,
                default_root_dir=logs_root_dir,
                strategy=distributed_backend,
                sync_batchnorm=sync_batchnorm,
                logger=logger,
                callbacks=[checkpoint_callback]
            )
            start = time.time()
            
            # starting the training
            trainer.fit(
                benchmark_model,
                train_dataloaders=dataloader_train_ssl,
                val_dataloaders=dataloader_test
            )
            end = time.time()
            run = {
                'model': model_name,
                'batch_size': batch_size,
                'epochs': max_epochs,
                'max_accuracy': benchmark_model.max_accuracy,
                'runtime': end - start,
                'gpu_memory_usage': torch.cuda.max_memory_allocated(),
                'seed': seed,
            }
            runs.append(run)
            print(run)

            # delete model and trainer + free up cuda memory
            del benchmark_model
            del trainer
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        bench_results[model_name] = runs

    # print results table
    header = (
        f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
        f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
    )
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for model, results in bench_results.items():
        runtime = np.array([result['runtime'] for result in results])
        runtime = runtime.mean() / 60 # convert to min
        accuracy = np.array([result['max_accuracy'] for result in results])
        gpu_memory_usage = np.array([result['gpu_memory_usage'] for result in results])
        gpu_memory_usage = gpu_memory_usage.max() / (1024**3) # convert to gbyte

        if len(accuracy) > 1:
            accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
        else:
            accuracy_msg = f"{accuracy.mean():>18.3f}"

        print(
            f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
            f"| {accuracy_msg} | {runtime:>6.1f} Min "
            f"| {gpu_memory_usage:>8.1f} GByte |",
            flush=True
        )
    print('-' * len(header))

import torch
import torch.nn as nn
from torchmetrics import Accuracy
import os
import math
from scipy.stats import entropy
from copy import deepcopy
from torch.nn.utils.weight_norm import WeightNorm

from networks import BaseModule, FeatureExtractor
from utils import SupConLoss, HScore
from augmentation import get_tta_transforms


class COMET(BaseModule):
    def __init__(self, datamodule, rejection_threshold=0.5, feature_dim=256, lr=1e-4, lower_confidence_threshold=0.25,
                 upper_confidence_threshold=0.75, ckpt_dir='', cl_projection_dim=128, cl_temperature=0.1,
                 m_teacher_momentum=0.999, lbd=0.1, use_source_prototypes=True):
        super(COMET, self).__init__(datamodule, feature_dim, lr, rejection_threshold, ckpt_dir)

        self.lower_confidence_threshold = lower_confidence_threshold
        self.upper_confidence_threshold = upper_confidence_threshold

        self.backbone_teacher = self.copy_model(self.backbone)
        self.feature_extractor_teacher = self.copy_model(self.feature_extractor)
        self.classifier_teacher = self.copy_model(self.classifier)

        self.ckpt_dir = ckpt_dir
        self.class_prototypes = None
        self.prototype_sum = torch.zeros(self.known_classes_num, feature_dim)
        self.prototype_sample_counter = torch.zeros(self.known_classes_num, 1)

        self.total_online_tta_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        self.total_online_tta_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)

        cl_projector = nn.Sequential(nn.Linear(self.feature_extractor.feature_dim, cl_projection_dim),
                                     nn.ReLU(), nn.Linear(cl_projection_dim, cl_projection_dim)).to(self.device)
        self.contrastive_loss = SupConLoss(projector=cl_projector, temperature=cl_temperature)
        self.tta_transform = get_tta_transforms()
        self.m_teacher_momentum = m_teacher_momentum
        self.lbd = lbd

        self.use_source_prototypes = use_source_prototypes
        self.automatic_optimization = False

    def configure_optimizers(self):
        # define different learning rates for different subnetworks
        params_group = []

        for k, v in self.backbone.named_parameters():
            params_group += [{'params': v, 'lr': self.lr * 0.1}]
        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.contrastive_loss.projector.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]

        optimizer = torch.optim.SGD(params_group, momentum=0.9, nesterov=True)
        return optimizer

    def on_train_start(self):
        if torch.cuda.is_available():
            self.class_prototypes = torch.load(self.ckpt_dir, map_location=torch.device('cuda'))['class_prototypes']
        else:
            self.class_prototypes = torch.load(self.ckpt_dir, map_location=torch.device('cpu'))['class_prototypes']

    def generate_pseudo_labels(self, y_hat):
        y_hat_entropy = torch.tensor(entropy(y_hat.detach().cpu(), axis=1) / math.log(self.known_classes_num))
        confident_idx = torch.where(torch.logical_or(y_hat_entropy <= self.lower_confidence_threshold,
                                                     y_hat_entropy >= self.upper_confidence_threshold))[0]
        pseudo_labels = torch.where(y_hat_entropy[confident_idx] >= self.upper_confidence_threshold,
                                    self.known_classes_num, torch.argmax(y_hat.cpu(), dim=1)[confident_idx])

        return confident_idx, pseudo_labels

    def forward_teacher(self, x, apply_softmax=True):
        x = self.backbone_teacher(x)
        feature_embed = self.feature_extractor_teacher(x)
        x = self.classifier_teacher(feature_embed)
        if apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x, feature_embed

    def copy_model(self, model):
        if not isinstance(model, FeatureExtractor):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            coppied_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            coppied_model = deepcopy(model)
        return coppied_model

    def update_ema_variables(self, ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model

    def on_train_epoch_end(self):
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.total_online_tta_hscore.compute()
            print(f"H-Score: {h_score}")
            print(f"Known Accuracy: {known_acc}")
            print(f"Unknown Accuracy: {unknown_acc}")
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('UnknownAcc', unknown_acc)

    def on_train_end(self):
        os.makedirs(os.path.join(self.trainer.log_dir, 'checkpoints'))
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
        }, self.trainer.log_dir + '/checkpoints/adapted_ckpt.pt')

    def training_step(self, train_batch):
        opt = self.optimizers()
        self.on_test_model_eval()
        self.class_prototypes = self.class_prototypes.to(self.device)
        self.prototype_sum = self.prototype_sum.to(self.device)
        self.prototype_sample_counter = self.prototype_sample_counter.to(self.device)

        x, y = train_batch
        y = torch.where(y >= self.known_classes_num, self.known_classes_num, y)

        opt.zero_grad()

        # FORWARD
        y_hat, features = self.forward(x, apply_softmax=True)
        y_hat_aug, features_aug = self.forward(self.tta_transform(x))
        y_hat_teacher, _ = self.forward_teacher(x, apply_softmax=True)

        # ADAPTATION
        with torch.no_grad():
            pseudo_label_idx, pseudo_label = self.generate_pseudo_labels(y_hat_teacher)
            pseudo_label = pseudo_label.to(self.device)
            pseudo_label_idx = pseudo_label_idx.to(self.device)
        known_idx = torch.where(pseudo_label != self.known_classes_num)[0].to(self.device)

        if not self.use_source_prototypes:
            with torch.no_grad():
                self.prototype_sum[pseudo_label[known_idx]] += features[pseudo_label_idx[known_idx]]
                for label in pseudo_label[known_idx]:
                    self.prototype_sample_counter[label] += 1

                self.class_prototypes = self.prototype_sum / (self.prototype_sample_counter + 1e-5)
                self.class_prototypes = self.class_prototypes.to(self.device)

        cl_known_features = torch.cat([torch.unsqueeze(self.class_prototypes[pseudo_label[known_idx]], dim=1),
                                       torch.unsqueeze(features[pseudo_label_idx[known_idx]], dim=1),
                                       torch.unsqueeze(features_aug[pseudo_label_idx[known_idx]], dim=1)],
                                      dim=1)
        unknown_idx = torch.where(pseudo_label == self.known_classes_num)[0]
        cl_unknown_features = torch.cat([torch.unsqueeze(features[pseudo_label_idx[unknown_idx]], dim=1),
                                         torch.unsqueeze(features_aug[pseudo_label_idx[unknown_idx]], dim=1)], dim=1)

        y_hat_entropy = -torch.matmul(y_hat, torch.log(y_hat.T)) / torch.log(torch.tensor(self.known_classes_num))
        y_hat_entropy = torch.diagonal(y_hat_entropy)

        if len(known_idx) != 0:
            con_loss = self.contrastive_loss(cl_known_features, labels=pseudo_label[known_idx],
                                             confident_unknown_features=cl_unknown_features)
            entropy_loss = y_hat_entropy[pseudo_label_idx[known_idx]].mean() -\
                           y_hat_entropy[pseudo_label_idx[unknown_idx]].mean()
            loss = con_loss + self.lbd * entropy_loss
            self.manual_backward(loss)
            self.log('tta_loss', loss, on_epoch=True, prog_bar=True)
        else:
            loss = None
        opt.step()

        self.backbone_teacher = self.update_ema_variables(ema_model=self.backbone_teacher, model=self.backbone,
                                                          alpha_teacher=self.m_teacher_momentum)
        self.feature_extractor_teacher = self.update_ema_variables(ema_model=self.feature_extractor_teacher,
                                                                   model=self.feature_extractor,
                                                                   alpha_teacher=self.m_teacher_momentum)
        self.classifier_teacher = self.update_ema_variables(ema_model=self.classifier_teacher,
                                                            model=self.classifier,
                                                            alpha_teacher=self.m_teacher_momentum)

        # PREDICTION
        with torch.no_grad():
            pred = torch.where(y_hat_entropy.detach() <= self.rejection_threshold, torch.argmax(y_hat.detach(), dim=1),
                               self.known_classes_num).to(self.device)
            self.total_online_tta_acc(pred, y)
            self.log('tta_acc', self.total_online_tta_acc, on_epoch=True, prog_bar=True)
            if self.open_flag:
                self.total_online_tta_hscore.update(pred, y)

from padim_utils import embedding_concat, plot_fig, process_mask
from mvtec import TRANSFORM
import cv2 as cv

import random
from random import sample
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
from mvtec import MVTecSingleClassDataset

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class Padim():
    transform = TRANSFORM

    def __init__(self, arch: str, classname: str, save_path: str = 'results', threshold: float = 0.5):
        self.arch = arch
        self.save_path = save_path
        self.class_name = classname

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.init_model()

        self.threshold = threshold

        self.train_weights = None

    def init_model(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        # load model
        if self.arch == 'resnet18':
            self.model = resnet18(weights='DEFAULT', progress=True)
            t_d = 448
            d = 100
        elif self.arch == 'wide_resnet50_2':
            self.model = wide_resnet50_2(weights='DEFAULT', progress=True)
            t_d = 1792
            d = 550

        # randomly reducing feature dimension to 100 or 550 for resnet18 and wide_resnet50_2, respectively
        self.random_idx = torch.tensor(sample(range(0, t_d), d))

        self.model.to(self.device)
        self.model.eval()
        random.seed(1024)
        torch.manual_seed(1024)

        if use_cuda:
            torch.cuda.manual_seed_all(1024)

        # set model's intermediate outputs
        self.hook_outputs = []

        def hook(module, input, output):
            self.hook_outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def save_train_weights(self):
        if self.train_weights is not None:
            print(f"Saving train features for {self.class_name}")
            mean,cov_inv = self.train_weights
            train_weights_filepath = os.path.join(self.save_path, f'train_{self.class_name}.npz')
            np.savez_compressed(train_weights_filepath, mean=mean, cov_inv=cov_inv, random_idx=self.random_idx.numpy())

    def get_train_weights(self):
        if self.train_weights is None:
            train_weights_filepath = os.path.join(self.save_path, f'train_{self.class_name}.npz')
            if os.path.exists(train_weights_filepath):
                print(f"Loading train features for {self.class_name}")
                data = np.load(train_weights_filepath)
                self.train_weights = [data['mean'], data['cov_inv']]
                self.random_idx = torch.tensor(data['random_idx'])
                del data
            else:
                raise FileNotFoundError(f"Train features for {self.class_name} not found")

    def save_eval_data(self):
        if self.normalization_data is not None and self.threshold is not None:
            print(f"Saving evaluation data for {self.class_name}")
            eval_data_filepath = os.path.join(self.save_path, f'eval_{self.class_name}.npz')
            np.savez_compressed(eval_data_filepath, normalization_data=self.normalization_data, threshold=self.threshold)

    def get_eval_data(self):
        eval_data_filepath = os.path.join(self.save_path, f'eval_{self.class_name}.npz')
        if os.path.exists(eval_data_filepath):
            print(f"Loading evaluation data for {self.class_name}")
            data = np.load(eval_data_filepath)
            self.normalization_data = data['normalization_data']
            self.threshold = data['threshold']
            del data
        else:
            raise FileNotFoundError(f"Evaluation data for {self.class_name} not found")


    def extract_features(self, dataset: MVTecSingleClassDataset, shuffle: bool = False):

        assert self.class_name == dataset.class_name, "Class names must be the same"
        
        dataloader = DataLoader(dataset, batch_size=32, pin_memory=True, shuffle=shuffle)

        embedding_vect = None
        anomaly_types = []
        for (x, _, _, anomaly_type) in tqdm(dataloader, f'| feature extraction | {dataset.subset} | {self.class_name} |'):
            anomaly_types.extend(anomaly_type)
            
            outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(outputs.keys(), self.hook_outputs):
                outputs[k].append(v.cpu().detach())
            self.hook_outputs = []

            for k, v in outputs.items():
                outputs[k] = torch.cat(v, 0)

            # Embedding concat
            batch_embedding_vect = outputs['layer1']
            del outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                batch_embedding_vect = embedding_concat(batch_embedding_vect, outputs[layer_name])
                del outputs[layer_name]

            batch_embedding_vect = torch.index_select(batch_embedding_vect, 1, self.random_idx)
            
            embedding_vect = batch_embedding_vect if embedding_vect is None else torch.cat([embedding_vect, batch_embedding_vect], 0)
            
            del batch_embedding_vect

        return embedding_vect, anomaly_types
    
    def train(self, train_dataset: MVTecSingleClassDataset):

        assert self.class_name == train_dataset.class_name, "Class names must be the same"
        
        # extract train set features
        train_feature_filepath = os.path.join(self.save_path, f'train_{self.class_name}.npz')

        if os.path.exists(train_feature_filepath):
            print(f"Train features for {self.class_name} already exist")
            self.get_train_weights()
            return

        embedding_vectors, _ = self.extract_features(train_dataset, shuffle=True)

        # calculate multivariate Gaussian distribution
        print("Calculating mean and covariance matrix")
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        for i in range(H * W):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        self.train_weights = [
            torch.mean(embedding_vectors, dim=0).numpy(),
            np.linalg.inv(cov.T).T
        ]
        del cov,embedding_vectors
        self.save_train_weights()
        return
    
    def validate(self, test_dataset: MVTecSingleClassDataset):       

        assert self.class_name == test_dataset.class_name, "Class names must be the same"
        
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True) 
        self.get_train_weights()
        
        gt_list = []
        gt_mask_list = []
        test_imgs = []

        total_roc_auc = []
        total_pixel_roc_auc = []

        embedding_vect = None
        # extract test set features
        for (x, y, mask, anomaly_type) in tqdm(test_dataloader, f'| feature extraction | {test_dataset.subset} | {self.class_name} |'):
            test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), self.hook_outputs):
                test_outputs[k].append(v.cpu().detach())
            self.hook_outputs = []
            
            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)
            self.hook_outputs = []
                
            # Embedding concat
            batch_embedding_vect = test_outputs['layer1']
            del test_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                batch_embedding_vect = embedding_concat(batch_embedding_vect, test_outputs[layer_name])
                del test_outputs[layer_name]

            # randomly select d dimension
            batch_embedding_vect = torch.index_select(batch_embedding_vect, 1, self.random_idx)
            
            embedding_vect = batch_embedding_vect if embedding_vect is None else torch.cat([embedding_vect, batch_embedding_vect], 0)

            del batch_embedding_vect
            
        # calculate distance matrix
        print("Calculating distance matrix")
        B, C, H, W = embedding_vect.size()
        embedding_vect = embedding_vect.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            dist_list.append([
                mahalanobis(sample[:, i], self.train_weights[0][:, i], self.train_weights[1][:, :, i])
                for sample in embedding_vect
            ])

        del embedding_vect

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        
        print("Generating score map")
        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()
            
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
            
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        self.normalization_data = min_score, max_score

        print("Calculating metrics")

        save_dir = os.path.join(self.save_path, f'_{self.arch}_{self.class_name}')
        os.makedirs(save_dir, exist_ok=True)

        fig1, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig1_img_rocauc = ax[0]
        fig1_pixel_rocauc = ax[1]

        fig2, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig2_pixel_f1 = ax[0]
        fig2_pixel_p_r = ax[1]

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print(f'image ROCAUC: {img_roc_auc:.3f}')
        fig1_img_rocauc.plot(fpr, tpr, label=f'{self.class_name} img_ROCAUC: {img_roc_auc:.3f}')
            
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        precision = np.maximum.accumulate(precision)
        recall = np.minimum.accumulate(recall)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        self.threshold = thresholds[np.argmax(f1)]
        
        print(f'Best threshold: {self.threshold:.3f} (F1={f1.max():.3f} at {self.threshold:.3f})')
        fig2_pixel_f1.plot(thresholds, f1[:-1], label=f'{self.class_name} F1: {f1.max():.3f}')
        fig2_pixel_f1.axvline(self.threshold, color='r', linestyle='--')
        fig2_pixel_f1.title.set_text(f'Pixel F1: {f1.max():.3f} at {self.threshold:.3f}')
        fig2_pixel_f1.legend(loc="lower right")

        fig2_pixel_p_r.plot(thresholds, precision[:-1], label=f'{self.class_name} Precision')
        fig2_pixel_p_r.plot(thresholds, recall[:-1], label=f'{self.class_name} Recall')
        fig2_pixel_p_r.title.set_text(f'Pixel Precision and Recall curves')
        fig2_pixel_p_r.legend(loc="lower right")
        
        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print(f'pixel ROCAUC: {per_pixel_rocauc:.3f}')

        fig1_pixel_rocauc.plot(fpr, tpr, label=f'{self.class_name} ROCAUC: {per_pixel_rocauc:.3f}')
        plot_fig(test_imgs, scores, gt_mask_list, self.threshold, save_dir, self.class_name)

        print(f'Average ROCAUC: {np.mean(total_roc_auc):.3f}')
        fig1_img_rocauc.title.set_text(f'Average image ROCAUC: {np.mean(total_roc_auc):.3f}')
        fig1_img_rocauc.legend(loc="lower right")

        print(f'Average pixel ROCUAC: {np.mean(total_pixel_roc_auc):.3f}')
        fig1_pixel_rocauc.title.set_text(f'Average pixel ROCAUC: {np.mean(total_pixel_roc_auc):.3f}')
        fig1_pixel_rocauc.legend(loc="lower right")

        fig1.tight_layout()
        fig1.savefig(os.path.join(self.save_path, f'roc_curve_{self.arch}_{self.class_name}.png'), dpi=300)

        fig2.tight_layout()
        fig2.savefig(os.path.join(self.save_path, f'pixel_f1_{self.arch}_{self.class_name}.png'), dpi=300)

        fig1.show()
        fig2.show()

        self.save_eval_data()

    def preview(self, image: np.ndarray):

        self.get_train_weights()
        self.get_eval_data()

        preview_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        
        print("Extracting features")
        # extract test set features        
        x = self.transform(Image.fromarray(cv.cvtColor(image,cv.COLOR_BGR2RGB))).unsqueeze(0) # unsqueeze to add dummy batch dimension
        # model prediction
        with torch.no_grad():
            _ = self.model(x.to(self.device))
        # get intermediate layer outputs
        for k, v in zip(preview_outputs.keys(), self.hook_outputs):
            preview_outputs[k].append(v.cpu().detach())
        self.hook_outputs = []
        for k, v in preview_outputs.items():
            preview_outputs[k] = torch.cat(v, 0)
            
        # Embedding concat
        embedding_vect = preview_outputs['layer1']
        del preview_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vect = embedding_concat(embedding_vect, preview_outputs[layer_name])
            del preview_outputs[layer_name]

        # randomly select d dimension
        embedding_vect = torch.index_select(embedding_vect, 1, self.random_idx)
            
        # calculate distance matrix
        print("Calculating distance matrix")
        B, C, H, W = embedding_vect.size()
        embedding_vect = embedding_vect.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            dist_list.append([
                mahalanobis(sample[:, i], self.train_weights[0][:, i], self.train_weights[1][:, :, i]) 
                for sample in embedding_vect
            ])

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample        
        print("Generating score map")
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()
            
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
            
        # Normalization
        min_score, max_score = self.normalization_data
        score = ((score_map - min_score) / (max_score - min_score))

        mask, heat_map = process_mask(score, self.threshold)
        return mask, heat_map
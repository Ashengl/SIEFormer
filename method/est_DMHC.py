import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from data.get_datasets import get_datasets, get_class_splits

from config import feature_extract_dir
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment as linear_assignment

from scipy.optimize import minimize_scalar
from functools import partial

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def test_DMHC_for_scipy(merge_test_loader, args=None, verbose=False, small_k=0):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """


    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (feats, label, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):

        feats = feats.to(device)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                 else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------

    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)
    #
    #
    all_feats = np.concatenate(all_feats)

    linked = linkage(all_feats, method="ward")

    gt_dist = linked[:, 2][-args.num_labeled_classes - args.num_unlabeled_classes]
    preds = fcluster(linked, t=gt_dist, criterion='distance')

    dist = linked[:, 2][:-(small_k)]
    tolerance = 0
    best_acc = 0
    diss = 0
    for d in reversed(dist):
        preds = fcluster(linked, t=d, criterion='distance')
        preds = preds-1
        k = max(preds)


        feat1 = []
        feat2 = []
        for i in range(preds.astype(int).max()+1):
            feat2.append([all_feats[preds.astype(int) == i].mean(0)])
        for i in range(targets.astype(int)[mask_lab].max()):
            feat1.append([all_feats[mask_lab][targets.astype(int)[mask_lab] == i].mean(0)])
        feat1 = np.concatenate(feat1, 0)
        feat2 = np.concatenate(feat2, 0)
        w = np.matmul(feat1, feat2.T)
        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T

        y_true = targets.astype(int)[mask_lab]
        y_pred = preds.astype(int)[mask_lab]
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w_ = np.zeros((D, D), dtype=int)
        for i in range(y_pred.size):
            w_[y_true[i], y_pred[i]] += 1
        ind = linear_assignment(w_.max() - w_)
        ind = np.vstack(ind).T

        disss = sum([w[i, j] * w_[i, j] for i, j in ind if i < w.shape[0]])

        if disss > diss:
            diss = disss
            best_acc_k = k
            tolerance = 0
            print('Best K: ', best_acc_k, diss)
        else:
            tolerance += 1

        if tolerance == 50:
            break

    mask = mask_lab

    return best_acc_k



def scipy_optimise(merge_test_loader, args):

    small_k = args.num_labeled_classes

    K = test_DMHC_for_scipy(merge_test_loader=merge_test_loader, args=args, verbose=True, small_k=small_k)
    print(f'Optimal K is {K}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_classes', default=1000, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_exp_id', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--search_mode', type=str, default='brent', help='Mode for black box optimisation')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cluster_accs = {}

    args.filter = 1

    args.save_dir = os.path.join(f'./sie/{args.model_name}_{args.dataset_name}') if args.filter else os.path.join(f'./gcd/{args.model_name}_{args.dataset_name}')

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(args)

    if args.warmup_model_exp_id is not None:
        args.save_dir += '_' + args.warmup_model_exp_id
        print(f'Using features from experiment: {args.warmup_model_exp_id}')
    else:
        print(f'Using pretrained {args.model_name} features...')

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    train_transform, test_transform = None, None
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform, test_transform, args)

    # Convert to feature vector dataset
    test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=os.path.join(args.save_dir, 'test'))
    unlabelled_train_examples_test = FeatureVectorDataset(base_dataset=unlabelled_train_examples_test,
                                                          feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset = FeatureVectorDataset(base_dataset=train_dataset, feature_root=os.path.join(args.save_dir, 'train'))

    # --------------------
    # DATALOADERS
    # --------------------
    unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                         batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False)

    print('Testing on all in the training data...')
    if args.search_mode == 'brent':
        print('Optimising with Brents algorithm')
        scipy_optimise(merge_test_loader=train_loader, args=args)
    else:
        binary_search(train_loader, args)
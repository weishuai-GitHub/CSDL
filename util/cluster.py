from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import _supervised as supervised
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_symmetric
from tqdm import tqdm
from sklearn import cluster
import scipy.sparse as sparse
import numpy as np
import torch

def clustering_test(labels_true, labels_pred):
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    # value = supervised.contingency_matrix(labels_true, labels_pred, sparse=False)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_assignment(-value)
    acc = value[r, c].sum() / len(labels_true)
    nmi = supervised.normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
    ari = supervised.adjusted_rand_score(labels_true, labels_pred)
    label_map = dict()
    for i,j in zip(r,c):
        label_map[i] = j
    return acc,label_map

def get_class_inner_boundary(datas,center_ids,labels=None):
    class_num = center_ids.shape[0]
    avg_dises = np.zeros((class_num,))
    for idx in range(class_num):
        indexs = np.where(labels==idx)
        data = datas[indexs]
        avg_dis =np.sqrt(np.sum(np.sum((data-center_ids[idx])**2))/data.shape[0])
        avg_dises[idx] = avg_dis
    return 2*avg_dises

def get_class_outer_boundary(center_ids):
    class_num = center_ids.shape[0]
    class_outer_dis = np.zeros((class_num,class_num))
    minValue = 1e9
    for i in range(class_num):
        for j in range(i+1,class_num):
            class_outer_dis[i][j] = np.sum((center_ids[i]-center_ids[j])**2)
            minValue = min(minValue,class_outer_dis[i][j])
    return minValue**(0.5)

def find_novelty(datas,feature,inner_boundary):
    length = datas.shape[0]
    is_novelty = np.zeros((length,),dtype=np.bool8)
    for idx in range(length):
        data_dis =np.sum((feature-datas[idx])**2,axis=1)**(0.5)
        indexs = np.where(data_dis<inner_boundary)
        if(indexs[0].shape[0]==0):
            is_novelty[idx] = True
    return is_novelty

def get_coff_index(features,labels,CenterId,ave_dises,mask):
    size = labels.shape[0]
    select_label = np.zeros((size,),dtype=np.bool8)
    for i in range(size):
        dis = np.sum((features[i]-CenterId[labels[i]])**2)
        if not mask[i] and dis<ave_dises[labels[i]]*ave_dises[labels[i]]:
            select_label[i] = True
    return select_label

def adjust_centerId(lastCenterIds,CenterIds):
    adjustCenterId = np.zeros_like(lastCenterIds)
    i = 0
    for centerid in CenterIds:
        min_dis = float('inf')
        for lastcenterid in lastCenterIds:
            dis = np.sum((centerid-lastcenterid)**2)
            if dis<min_dis:
                adjustCenterId[i] = lastcenterid
                min_dis = dis
        i = i+1
    return adjustCenterId

def acc(labels_true, labels_pred,label_map):
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    tol= 0.0
    for tlables,plabels in zip(labels_true,labels_pred):
        if label_map[tlables]==plabels:
            tol = tol+1
    return tol/len(labels_true)

def test_kmeans(model,loader,args,is_train=False):
    all_feats = []
    all_mask = []
    all_targets = np.array([])
    
    if not is_train:
        args.logger.info('Collating features...')
    else:
        args.logger.info('Get predicted labels...')
    # First extract all features
    for batch_idx, (images, label,mask) in enumerate(tqdm(loader)):

        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats,_ = model(images)

        #feats = torch.nn.functional.normalize(feats, dim=-1)
        feats = feats.detach().cpu().numpy()
        all_feats.append(feats)
        if is_train:
            label = label.cpu().numpy()
            mask = np.bool8(mask.numpy()[:,0])
        else:
            label = np.array([True if x in range(len(args.train_classes)) else False for x in label])
            mask = np.bool8(mask.numpy())
        all_targets = np.append(all_targets, label)
        
        all_mask.append(mask)

    # -----------------------
    # K-MEANS
    # -----------------------
    args.logger.info('Fitting K-Means...')
    all_mask = np.concatenate(all_mask)
    all_feats = np.concatenate(all_feats)
    all_kmeans = cluster.k_means(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes).fit(all_feats)
    all_preds = all_kmeans.labels_
    args.logger.info('Done!')
    all_acc,label_map = clustering_test(all_targets,all_preds)

    old_target = all_targets[all_mask]
    old_preds = all_preds[all_mask]
    old_acc = acc(old_target,old_preds,label_map)

    new_target = all_targets[~all_mask]
    new_preds = all_preds[~all_mask]
    new_acc = acc(new_target,new_preds,label_map)
    
    if not is_train:
        args.logger.info("testData [ all_acc:{:.4f},old_acc:{:.4f},new_acc:{:.4f} ]".format(all_acc,old_acc,new_acc))
        return all_acc,old_acc,new_acc
    else:
        CenterId = all_kmeans.cluster_centers_
        ave_dises = get_class_inner_boundary(all_feats,CenterId,all_preds)
        select_labels = get_coff_index(all_feats,all_preds,CenterId,ave_dises,all_mask)
        args.logger.info("trainData [ all_acc:{:.4f},old_acc:{:.4f},new_acc:{:.4f} ]".format(all_acc,old_acc,new_acc))
        # if last_CenterId is not None:
        #     last_CenterId = adjust_centerId(last_CenterId,CenterId)
        #     CenterId = (1-args.alph)*last_CenterId + args.alph*CenterId
        return select_labels,all_kmeans

def split_cluster_nmi(labels, preds,mask):
    old_labels = labels[mask]
    new_labels = labels[~mask]
    old_preds = preds[mask]
    new_preds = preds[~mask]
    old_nmi = normalized_mutual_info_score(old_labels, old_preds)
    if new_labels.shape[0]==0:
        new_nmi = old_nmi
    else:
        new_nmi = normalized_mutual_info_score(new_labels, new_preds)
    all_nmi = normalized_mutual_info_score(labels, preds)
    return all_nmi,old_nmi,new_nmi

def split_cluster_ari(labels, preds,mask):
    old_labels = labels[mask]
    new_labels = labels[~mask]
    old_preds = preds[mask]
    new_preds = preds[~mask]
    old_ari = adjusted_rand_score(old_labels, old_preds)
    if new_labels.shape[0]==0:
        new_ari = old_ari
    else:
        new_ari = adjusted_rand_score(new_labels, new_preds)
    all_ari = adjusted_rand_score(labels, preds)
    return all_ari,old_ari,new_ari

def split_cluster_acc(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc

def spectral_clustering(affinity_matrix_, n_clusters, k, seed=1, n_init=20,is_get_embedding=False):
    affinity_matrix_ = check_symmetric(affinity_matrix_)
    random_state = check_random_state(seed)

    laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, 
                                 k=k, sigma=None, which='LA')
    #embedding = normalize(affinity_matrix_.dot(vec))
    vec = normalize(vec)
    _, labels_, _ = cluster.k_means(vec, n_clusters, 
                                    random_state=random_state, n_init=n_init)
    if is_get_embedding:
        return labels_, vec
    else:
        return labels_

import torch
import torch.distributed as dist
from enum import Enum
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import copy

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    bag_score = 0
    aucs = [] 
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        _, _, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        # thresholds_optimal.append(threshold_optimal)

        class_prediction_bag = copy.deepcopy(predictions[:, c])
        class_prediction_bag[predictions[:, c]>=threshold_optimal] = 1
        class_prediction_bag[predictions[:, c]<threshold_optimal] = 0
        predictions[:, c] = class_prediction_bag

    for i in range(0, len(labels)):
        bag_score = np.array_equal(labels[i], predictions[i]) + bag_score         
    acc = bag_score / len(labels)

    return np.array(aucs), acc

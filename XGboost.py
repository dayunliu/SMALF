# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
from get_data import get_train_data
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

plt.style.use('ggplot')

plt.figure(figsize=(8, 5))


def train(X, Y):
    X, Y = shuffle(X, Y, random_state=1)

    clf = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=1)
    kf = KFold(n_splits=5)
    print("开始训练!")
    AUC_list = []
    p_list = []
    r_list = []
    f1_list = []
    AUPR_list = []
    for train_index, test_index in kf.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]
        clf.fit(X_train, Y_train)
        predict_value = clf.predict_proba(X_test)[:, 1]
        AUC = metrics.roc_auc_score(Y_test, predict_value)
        precision, recall, _ = precision_recall_curve(Y_test, predict_value)
        AUCPR = auc(recall, precision)
        AUPR_list.append(AUCPR)
        p = precision_score(Y_test, predict_value.round())
        p_list.append(p)
        r = recall_score(Y_test, predict_value.round())
        r_list.append(r)
        f1 = f1_score(Y_test, predict_value.round())
        f1_list.append(f1)
        AUC_list.append(AUC)
    print("the average of the AUC is ", sum(AUC_list) / len(AUC_list))
    print("the average of the AUPR is ", sum(AUPR_list) / len(AUPR_list))
    print("the average of p is ", sum(p_list) / len(p_list))
    print("the average of r is ", sum(r_list) / len(r_list))
    print("the average of f1 is ", sum(f1_list) / len(f1_list))


if __name__ == "__main__":
    sample_data, label = get_train_data()
    sample_data = np.array(sample_data)
    label = np.array(label)
    train(sample_data, label)

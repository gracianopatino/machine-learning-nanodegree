import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

def plot_roc_auc(y_true, y_pred):
    """
    This function plots the ROC curve and provides the score.
    """

    # initialize dictionaries and array
    fpr = dict()
    tpr = dict()
    
    # prepare for figure
    plt.figure()

    # obtain ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    # obtain ROC AUC
    roc_auc = auc(fpr, tpr)
    # plot ROC curve
    plt.plot(fpr, tpr, color='aqua', lw=2, label='ROC curve(area = {f:.2f})'.format(f=roc_auc))
    
    # format figure
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()
    
    # print AUC score
    print('Score: {f:.3f}'. format(f=roc_auc))
    
def confusion_matrix_com(y_true, y_prob, thresh=0.5, target_names = ['class 0', 'class 1']):
    """
    This function prints the confusion matrix considering a given threshold.
    """
    
    y_pred = y_prob
    
    # obtain class predictions from probabilities
    y_pred = (y_pred>=thresh)*1

    # obtain (unnormalized) confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:\n', cm, '\n')
    
    # print the classification report
    print(classification_report(y_true, y_pred, target_names=target_names))
    
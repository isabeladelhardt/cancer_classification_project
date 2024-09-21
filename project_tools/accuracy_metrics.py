import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scores(predicted, actual):
   ## true positives
    TP = sum(predicted[predicted == 0] == actual[predicted == 0])
    ## true negatives
    TN = sum(predicted[predicted == 1] == actual[predicted == 1])
    ## false positives
    FP = sum(predicted[predicted == 0] != actual[predicted == 0])
    ## false negatives
    FN = sum(predicted[predicted == 1] != actual[predicted == 1])
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)
    
    print(f"{TP=}, {TN=}, {FP=}, {FN=}")
    print(f"{precision=:.4f}, {recall=:.4f}, {fpr=:.4f}, {fnr=:.4f}")

    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"{f1_score=:.4f}")
    


def roc_curve(model, x, y):
    thresholds = np.linspace(0, 1, 100)
    fpr = []
    tpr = []
    for t in thresholds:
        y_hat = model.predict_proba(x)[:, 1] > t
        y_hat = y_hat.astype(int)
        fpr_t = np.sum((y_hat == 1) & (y == 0)) / np.sum(y == 0)
        fpr.append(fpr_t)
    
        tpr_t = np.sum((y_hat == 1) & (y == 1)) / np.sum(y ==1)
        tpr.append(tpr_t)
        
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('ROC Curve')
    plt.show()

    return fpr, tpr
    


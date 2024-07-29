import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred > threshold, dtype=float)

    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }

def my_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    inter = 0
    union = 0
    for i in range(43):
        if pred[i] and target[i]:
            inter += 1
        if pred[i] or target[i]:
            union += 1
    return inter/union
    
def batch_metrics(pred, target):
    batch_size = len(pred)
    a = np.arange(batch_size, dtype = 'float')
    for i in range(batch_size):
        a[i] = my_metrics(pred[i], target[i])
    return np.mean(a)
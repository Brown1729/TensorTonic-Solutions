import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    sort_indices = np.lexsort((-y_true, -y_score))
    y_true_sorted = y_true[sort_indices]
    y_score_sorted = y_score[sort_indices]
    tpr_tmp = y_true_sorted.cumsum() / y_true.sum()
    fpr_tmp = (1 - y_true_sorted).cumsum() / (1 - y_true).sum()
    unique_indices = np.concatenate([np.asarray(np.where(np.diff(y_score_sorted) != 0))[0], [len(y_score_sorted) - 1]])
    tpr = np.concatenate([np.array([0]), tpr_tmp[unique_indices]])
    fpr = np.concatenate([np.array([0]), fpr_tmp[unique_indices]])
    threshold = np.concatenate([np.array([float('inf')]), y_score_sorted[unique_indices]])
    return fpr, tpr, threshold
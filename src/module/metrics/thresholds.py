# module/metrics/thresholds.py
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

@torch.no_grad()
def best_threshold_youden(pos_scores: torch.Tensor, y_true: torch.Tensor):
    scores = pos_scores.detach().flatten().cpu().to(torch.float64)
    y = y_true.detach().flatten().cpu().to(torch.int64)
    scores, idx = torch.sort(scores, descending=True)
    y = y[idx]
    P = int((y == 1).sum().item())
    N = int((y == 0).sum().item())
    if P == 0 or N == 0:
        return 0.5, 0.0, 0.0, 0.0
    tp_cum = torch.cumsum((y == 1).to(torch.int64), dim=0)
    fp_cum = torch.cumsum((y == 0).to(torch.int64), dim=0)
    tpr = tp_cum.to(torch.float64) / P
    fpr = fp_cum.to(torch.float64) / N
    mask = torch.ones_like(scores, dtype=torch.bool)
    if scores.numel() > 1:
        mask[1:] = scores[1:] != scores[:-1]
    tpr_u = tpr[mask]; fpr_u = fpr[mask]; scores_u = scores[mask]
    J_u = tpr_u - fpr_u
    best_i = int(torch.argmax(J_u).item())
    return float(scores_u[best_i].item()), float(J_u[best_i].item()), float(tpr_u[best_i].item()), float(fpr_u[best_i].item())

def best_threshold_fbeta(pos_scores: torch.Tensor, y_true: torch.Tensor, beta: float = 2.0):
    y = y_true.detach().flatten().cpu().to(torch.int64).numpy()
    s = pos_scores.detach().flatten().cpu().to(torch.float64).numpy()
    if np.unique(y).size < 2:
        return 0.5, 0.0, 0.0, 0.0
    p, r, thr = precision_recall_curve(y, s)
    if thr.size == 0:
        return 0.5, 0.0, 0.0, 0.0
    beta2 = float(beta) * float(beta)
    num = (1 + beta2) * (p[1:] * r[1:])
    den = (beta2 * p[1:] + r[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(den > 0, num / den, 0.0)
    k = int(np.nanargmax(f))
    return float(thr[k]), float(f[k]), float(p[k+1]), float(r[k+1])

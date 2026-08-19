import numpy as np
import torch
import math

EPS = 1e-6

def get_cosine_scheduler_with_warmup(optimizer, num_epoch, warmup_epoch):

    def lr_lambda(step):
        if step < warmup_epoch:
            return max(EPS, float(step) / float(max(1, warmup_epoch)))
        progress = float(step - warmup_epoch) / float(max(1, num_epoch - warmup_epoch))
        return max(0.0, (1.0 + math.cos(math.pi * progress)) / 2.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_rank(score_pos, score_neg):
    rank = np.sum(score_neg > score_pos) + 1
    if np.sum(score_neg == score_pos) > 0:
        rank += np.random.randint(np.sum(score_neg == score_pos))

    return rank

def compute_metric(ranks):
    ranks = np.array(ranks)

    mr = np.mean(ranks)
    mrr = np.mean(1 / ranks)
    hit10 = np.mean(ranks < 11)
    hit5 = np.mean(ranks < 6)
    hit1 = np.mean(ranks < 2)

    return mr, mrr, hit10, hit5, hit1
from scipy.sparse import csr_matrix, diags
import numpy as np
import torch

def inv_diag(E):
    res = E.sum(axis=1).A1
    res[res != 0] = 1 / (res[res != 0])
    return res

def create_relation_graph(triplets, num_ent, num_rel):
    triplets = np.array(triplets)
    idx_h = triplets[:, 0]
    idx_r = triplets[:, 1]
    idx_t = triplets[:, 2]

    E_h = csr_matrix((np.ones(len(triplets)), (idx_h, idx_r)), shape=(num_ent, num_rel))
    E_t = csr_matrix((np.ones(len(triplets)), (idx_t, idx_r)), shape=(num_ent, num_rel))

    D_h_inv = csr_matrix((inv_diag(E_h), (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
    D_t_inv = csr_matrix((inv_diag(E_t), (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))

    E_h = D_h_inv @ E_h
    E_t = D_t_inv @ E_t

    A_hh = E_h.transpose() @ E_h
    A_ht = E_h.transpose() @ E_t
    A_th = E_t.transpose() @ E_h
    A_tt = E_t.transpose() @ E_t

    E_hh = [[i, 0, j] for i, j in zip(*A_hh.nonzero())]
    E_ht = [[i, 1, j] for i, j in zip(*A_ht.nonzero())]
    E_th = [[i, 2, j] for i, j in zip(*A_th.nonzero())]
    E_tt = [[i, 3, j] for i, j in zip(*A_tt.nonzero())]
    E_inv = [[2*i, 4, 2*i+1] for i in range(num_rel // 2)] + [[2*i+1, 4, 2*i] for i in range(num_rel // 2)]

    E = E_hh + E_ht + E_th + E_tt + E_inv
    return torch.tensor(E)
from dataset import TrainKG, InferenceKG
from relgraph import create_relation_graph
from tqdm import tqdm
import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("-d", type=int, default=16)
args = parser.parse_args()

trainKG = TrainKG(args.data, 1, 6, 6, 1)
inferenceKG = InferenceKG(args.data, 1, 6, 6, "valid")

def compute_LapPE(graph, num_ent):
    ### Adjacency matrix ###
    A = torch.zeros(num_ent, num_ent).cuda()
    for h, _, t in tqdm(graph):
        A[h, t] += 1
    ### Adjacency matrix ###

    ### Degree matrix ###
    Dinv = torch.diag(torch.clamp(torch.sum(A, 0), min=1) ** -0.5).cuda()
    ### Degree matrix ###

    ### Graph Laplacian ###
    L = torch.eye(num_ent).cuda() - Dinv @ A @ Dinv
    ### Graph Laplacian ###

    ### Eigendecomposition ###
    L, V = torch.linalg.eig(L)
    idx = L.cpu().numpy().argsort()
    idx = torch.tensor(idx).long()
    L, V = L[idx], V[:, idx].real
    ### Eigendecomposition ###

    ### Compute LapPE ###
    eigval, eigvec = L[1:args.d+1], V[:, 1:args.d+1]
    print(eigval); print(eigvec)
    print(eigvec.shape)
    ### Compute LapPE ###

    lapPE = {"eigval": eigval, "eigvec": eigvec}

    return lapPE

LapPE_train_ent = compute_LapPE(trainKG.background, trainKG.num_ent)
LapPE_inference_ent = compute_LapPE(inferenceKG.background, inferenceKG.num_ent)

relgraph_train = create_relation_graph(trainKG.background.cpu(), trainKG.num_ent, trainKG.num_rel)
relgraph_inference = create_relation_graph(inferenceKG.background.cpu(), inferenceKG.num_ent, inferenceKG.num_rel)

LapPE_train_rel = compute_LapPE(relgraph_train, trainKG.num_rel)
LapPE_inference_rel = compute_LapPE(relgraph_inference, inferenceKG.num_rel)

os.makedirs("./pos_enc", exist_ok=True)
torch.save(LapPE_train_ent, f"./pos_enc/LapPE_{args.data}_train_ent_{args.d}.pt")
torch.save(LapPE_inference_ent, f"./pos_enc/LapPE_{args.data}_inference_ent_{args.d}.pt")
torch.save(LapPE_train_rel, f"./pos_enc/LapPE_{args.data}_train_rel_{args.d}.pt")
torch.save(LapPE_inference_rel, f"./pos_enc/LapPE_{args.data}_inference_rel_{args.d}.pt")
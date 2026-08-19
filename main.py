import time
import argparse
import torch
import torch.nn as nn
import os
import random
import numpy as np
import logging
from dataset import TrainKG, InferenceKG
from model import Ours
from utils import get_cosine_scheduler_with_warmup, compute_rank, compute_metric
from relgraph import create_relation_graph

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("-K", type=int, default=1)
parser.add_argument("-d", type=int, default=32)
parser.add_argument("-nlr", type=int, default=3)
parser.add_argument("-nls", type=int, default=6)
parser.add_argument("-nle", type=int, default=6)
parser.add_argument("-pos", type=str, default="lappe")
parser.add_argument("-p", type=float, default=0.0)
parser.add_argument("-nh", type=int, default=4)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-gc", type=float, default=5.0)
parser.add_argument("-bs", type=int, default=16)
parser.add_argument("-ne", type=int, default=20000)
parser.add_argument("-we", type=int, default=1000)
parser.add_argument("-pe", type=int, default=10)
parser.add_argument("-ve", type=int, default=500)
parser.add_argument("-seed", type=int, default=0)
parser.add_argument("-exp", type=str, default="debug")
parser.add_argument("-save", action="store_true")
args = parser.parse_args()

args_dict = vars(args).copy()
name = "--".join(f"{key}_{args_dict[key]}" for key in args_dict)

### Reproducibility/PyTorch Settings ###
OMP_NUM_THREADS = 8
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
# torch.autograd.set_detect_anomaly(True) # For debug only
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()
### Reproducibility/PyTorch Settings ###

### Logging ###
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

os.makedirs(f"./log/{args.exp}", exist_ok=True)
file_handler = logging.FileHandler(f"./log/{args.exp}/{name}.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
### Logging ###

trainKG = TrainKG(args.data, args.K, args.nls, args.nle, args.bs)
validKG = InferenceKG(args.data, args.K, args.nls, args.nle, "valid")
testKG = InferenceKG(args.data, args.K, args.nls, args.nle, "test")

model = Ours(
    nlr=args.nlr,
    nls=args.nls,
    nle=args.nle,
    d=args.d,
    nh=args.nh,
    p=args.p,
    pos=args.pos
).cuda()

### Loss & Optimizer ###
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = get_cosine_scheduler_with_warmup(optimizer, args.ne, args.we)
### Loss & Optimizer ###

### Positional Encodings ###
if args.pos == "none":
    pos_ent_train, pos_rel_train = None, None
    pos_ent_inference, pos_rel_inference = None, None
elif args.pos == "lappe":
    lappe_ent_train = torch.load(f"./pos_enc/LapPE_{args.data}_train_ent_{args.d // 2}.pt")
    lappe_rel_train = torch.load(f"./pos_enc/LapPE_{args.data}_train_rel_{args.d // 2}.pt")

    lappe_ent_inference = torch.load(f"./pos_enc/LapPE_{args.data}_inference_ent_{args.d // 2}.pt")
    lappe_rel_inference = torch.load(f"./pos_enc/LapPE_{args.data}_inference_rel_{args.d // 2}.pt")

    le_train, ve_train = lappe_ent_train["eigval"].real.cuda(), lappe_ent_train["eigvec"].cuda()
    lr_train, vr_train = lappe_rel_train["eigval"].real.cuda(), lappe_rel_train["eigvec"].cuda()

    le_inference, ve_inference = lappe_ent_inference["eigval"].real.cuda(), lappe_ent_inference["eigvec"].cuda()
    lr_inference, vr_inference = lappe_rel_inference["eigval"].real.cuda(), lappe_rel_inference["eigvec"].cuda()
### Positional Encodings ###

logger.info("Start training...")
start = time.time()
total_loss = 0
for epoch in range(args.ne):
    graph_a, offset_a, entities_a, graph_s, offset_s, entities_s, graph_q, offset_q, entities_q, query, cand, idx_gt = trainKG.get_batch()

    if args.pos == "lappe":
        ### Sign flip ###
        sign_ent = torch.rand(ve_train.shape[1]).cuda()
        sign_ent = torch.where(sign_ent > 0.5, 1, -1).unsqueeze(0)
        ve_train = sign_ent * ve_train

        sign_rel = torch.rand(vr_train.shape[1]).cuda()
        sign_rel = torch.where(sign_rel > 0.5, 1, -1).unsqueeze(0)
        vr_train = sign_rel * vr_train
        ### Sign flip ###

        pos_ent_train, pos_rel_train = model.pos_encoder(le_train, ve_train, lr_train, vr_train)
        pos_rel_train = torch.cat((pos_rel_train, torch.mean(pos_rel_train, 0, keepdim=True)), 0)

    if len(graph_a) > 0:
        tri_sup = torch.tensor([[h, trainKG.num_rel, h + 1] for h in offset_a[:-1]]).cuda()
        bg_rel = torch.cat((graph_a, tri_sup), 0)
        graph_rel = create_relation_graph(bg_rel.cpu(), offset_a[-1].item(), trainKG.num_rel + 1).cuda()
    else:
        graph_rel = []

    if args.pos == "none":
        init_emb_rel = torch.zeros(trainKG.num_rel + 1, args.d).cuda()
    else:
        init_emb_rel = torch.zeros(trainKG.num_rel + 1, args.d // 2).cuda()
    init_emb_rel[-1] = 1

    emb_rel = model.compute_emb_rel(graph_rel, init_emb_rel, pos_rel_train)

    attn_vec = model.compute_attn_vec(graph_a, offset_a, entities_a, emb_rel, pos_ent_train)
    emb_s = model.compute_emb_ent(graph_s, offset_s, entities_s, emb_rel, attn_vec, pos_ent_train)
    emb_q = model.compute_emb_ent(graph_q, offset_q, entities_q % trainKG.num_ent, emb_rel, attn_vec, pos_ent_train)
    
    scores = model.predict(emb_s, offset_s, entities_s, emb_q, entities_q, query, cand, trainKG.num_ent, pos_ent_train)
    loss = loss_fn(scores, idx_gt)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc, error_if_nonfinite=False)
    optimizer.step()
    scheduler.step()
    
    total_loss += loss.item()

    if (epoch + 1) % args.pe == 0:
        logger.info(f"Epoch {epoch + 1}: loss: {total_loss / args.pe:.6f} | lr: {scheduler.get_last_lr()[0]:.6f} ({time.time() - start:.2f} s)")
        total_loss = 0

    if (epoch + 1) % args.ve == 0:
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            if args.pos == "lappe":
                pos_ent_inference, pos_rel_inference = model.pos_encoder(le_inference, ve_inference, lr_inference, vr_inference)
                pos_rel_inference = torch.cat((pos_rel_inference, torch.mean(pos_rel_inference, 0, keepdim=True)), 0)

            if args.pos == "none":
                init_emb_rel = torch.zeros(validKG.num_rel + 1, args.d).cuda()
            else:
                init_emb_rel = torch.zeros(validKG.num_rel + 1, args.d // 2).cuda()
            init_emb_rel[-1] = 1

            ranks = []
            for idx_task in validKG.tasks:
                ranks_task = []
                graph_a, offset_a, entities_a, graph_s, offset_s, entities_s = validKG.get_batch_support(idx_task)

                if len(graph_a) > 0:
                    tri_sup = torch.tensor([[h, validKG.num_rel, h + 1] for h in offset_a[:-1]]).cuda()
                    bg_rel = torch.cat((graph_a, tri_sup), 0)
                    graph_rel = create_relation_graph(bg_rel.cpu(), offset_a[-1].item(), validKG.num_rel + 1).cuda()
                else:
                    graph_rel = []
                
                emb_rel = model.compute_emb_rel(graph_rel, init_emb_rel, pos_rel_inference)
                attn_vec = model.compute_attn_vec(graph_a, offset_a, entities_a, emb_rel, pos_ent_inference)
                emb_s = model.compute_emb_ent(graph_s, offset_s, entities_s, emb_rel, attn_vec, pos_ent_inference)

                for idx_batch in range(len(validKG.tasks[idx_task]) - args.K):
                    graph_q, offset_q, entities_q, query, cand = validKG.get_batch_query(idx_task, idx_batch)
                    emb_q = model.compute_emb_ent(graph_q, offset_q, entities_q, emb_rel, attn_vec, pos_ent_inference)

                    scores = model.predict(emb_s, offset_s, entities_s, emb_q, entities_q, query, cand, validKG.num_ent, pos_ent_inference).flatten()
                    score_pos = scores[0].item()
                    score_neg = scores[1:].detach().cpu().numpy()

                    rank = compute_rank(score_pos, score_neg)
                    ranks.append(rank)
                    ranks_task.append(rank)

                mr, mrr, hit10, hit5, hit1 = compute_metric(ranks_task)
                logger.info(f"{validKG.id2task[idx_task]} ({len(ranks_task)}): {mr:.2f} {mrr:.4f} {hit10:.4f} {hit5:.4f} {hit1:.4f}")

            mr, mrr, hit10, hit5, hit1 = compute_metric(ranks)
            logger.info(f"Epoch {epoch + 1}: [Valid] MR: {mr:.2f} MRR: {mrr:.6f} Hit@10: {hit10:.6f} Hit@5: {hit5:.6f} Hit@1: {hit1:.6f}")

            ranks = []
            for idx_task in testKG.tasks:
                ranks_task = []
                graph_a, offset_a, entities_a, graph_s, offset_s, entities_s = testKG.get_batch_support(idx_task)

                if len(graph_a) > 0:
                    tri_sup = torch.tensor([[h, testKG.num_rel, h + 1] for h in offset_a[:-1]]).cuda()
                    bg_rel = torch.cat((graph_a, tri_sup), 0)
                    graph_rel = create_relation_graph(bg_rel.cpu(), offset_a[-1].item(), testKG.num_rel + 1).cuda()
                else:
                    graph_rel = []
                
                emb_rel = model.compute_emb_rel(graph_rel, init_emb_rel, pos_rel_inference)
                attn_vec = model.compute_attn_vec(graph_a, offset_a, entities_a, emb_rel, pos_ent_inference)
                emb_s = model.compute_emb_ent(graph_s, offset_s, entities_s, emb_rel, attn_vec, pos_ent_inference)

                for idx_batch in range(len(testKG.tasks[idx_task]) - args.K):
                    graph_q, offset_q, entities_q, query, cand = testKG.get_batch_query(idx_task, idx_batch)
                    emb_q = model.compute_emb_ent(graph_q, offset_q, entities_q, emb_rel, attn_vec, pos_ent_inference)

                    scores = model.predict(emb_s, offset_s, entities_s, emb_q, entities_q, query, cand, testKG.num_ent, pos_ent_inference).flatten()
                    score_pos = scores[0].item()
                    score_neg = scores[1:].detach().cpu().numpy()

                    rank = compute_rank(score_pos, score_neg)
                    ranks.append(rank)
                    ranks_task.append(rank)

                mr, mrr, hit10, hit5, hit1 = compute_metric(ranks_task)
                logger.info(f"{testKG.id2task[idx_task]} ({len(ranks_task)}): {mr:.2f} {mrr:.4f} {hit10:.4f} {hit5:.4f} {hit1:.4f}")

            mr, mrr, hit10, hit5, hit1 = compute_metric(ranks)
            logger.info(f"Epoch {epoch + 1}: [Test] MR: {mr:.2f} MRR: {mrr:.6f} Hit@10: {hit10:.6f} Hit@5: {hit5:.6f} Hit@1: {hit1:.6f}")

        if args.save:
            os.makedirs(f"./ckpt/{args.exp}", exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                        f"./ckpt/{args.exp}/{name}--epoch_{epoch + 1}.pt")

        model.train()
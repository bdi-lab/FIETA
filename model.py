import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


### Layer of \phi_\text{rel} (Section 4.2) ###
class RelationGNNLayer(nn.Module):
    def __init__(self, d, nh, pos):
        super(RelationGNNLayer, self).__init__()

        self.d = d
        self.nh = nh
        self.pos = pos

        assert self.d % self.nh == 0
        self.dh = self.d // self.nh
        if self.pos == "none":
            self.do = self.d
        else:
            self.do = self.d // 2

        self.W_attn_r1 = nn.Linear(self.d, self.d, bias=False)
        self.W_attn_r2 = nn.Linear(self.d, self.d, bias=False)
        self.W_attn_R = nn.Linear(self.d, self.d)
        self.y_attn = nn.Parameter(torch.zeros(1, self.nh, self.dh))
        self.W_proj_r1 = nn.Linear(self.d, self.do, bias=False)
        self.W_proj_R = nn.Linear(self.d, self.do, bias=False)
        self.W_resi = nn.Linear(self.d, self.do)

        self.layer_norm = nn.LayerNorm(self.do)
        self.act = nn.LeakyReLU(0.2)

        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.W_attn_r1.weight, gain=gain)
        nn.init.xavier_normal_(self.W_attn_r2.weight, gain=gain)
        nn.init.xavier_normal_(self.W_attn_R.weight, gain=gain)
        nn.init.xavier_normal_(self.y_attn, gain=gain)
        nn.init.xavier_normal_(self.W_proj_r1.weight, gain=gain)
        nn.init.xavier_normal_(self.W_proj_R.weight, gain=gain)
        nn.init.xavier_normal_(self.W_resi.weight, gain=gain)

        nn.init.zeros_(self.W_attn_R.bias)
        nn.init.zeros_(self.W_resi.bias)

    def forward(self, graph, emb_rel, emb_meta_rel):
        num_rel = len(emb_rel)

        idx_r1 = graph[:, 0].long()
        idx_R = graph[:, 1].long()
        idx_r2 = graph[:, 2].long()
        
        ### Compute attention values ###
        attn_mat = self.W_attn_r1(emb_rel)[idx_r1] + self.W_attn_r2(emb_rel)[idx_r2] + self.W_attn_R(emb_meta_rel[idx_R])
        attn_val_raw = (self.y_attn * self.act(attn_mat.view(-1, self.nh, self.dh))).sum(-1, keepdim=True)

        scatter_idx = idx_r2.unsqueeze(-1).repeat(1, self.nh).unsqueeze(-1)
        attn_val_max = torch.zeros(num_rel, self.nh, 1).cuda().scatter_reduce(0, scatter_idx, attn_val_raw, reduce="amax", include_self=False)

        attn_val = torch.exp(attn_val_raw - attn_val_max[idx_r2])
        attn_sum = torch.zeros(num_rel, self.nh, 1).cuda().index_add(0, idx_r2, attn_val)
        beta = attn_val / (attn_sum[idx_r2] + EPS)
        ### Compute attention values ###

        ### Compute message ###
        msg_mat = self.W_proj_r1(emb_rel)[idx_r1] + self.W_proj_R(emb_meta_rel)[idx_R]
        msg = beta * msg_mat.view(-1, self.nh, self.do // self.nh)
        ### Compute message ###
        
        ### Aggregation ###
        output = self.W_resi(emb_rel).index_add(0, idx_r2, msg.flatten(1, -1))
        output = self.act(self.layer_norm(output))
        ### Aggregation ###

        return output
### Layer of \phi_\text{rel} (Section 4.2) ###


### \phi_\text{rel} (Section 4.2) ###
class RelationGNN(nn.Module):
    def __init__(self, nl, d, nh, pos):
        super(RelationGNN, self).__init__()

        self.nl = nl
        self.d = d
        self.nh = nh
        self.pos = pos

        self.emb_meta_rel = nn.Parameter(torch.zeros(5, self.d))

        self.layers = nn.ModuleList()
        for _ in range(self.nl):
            self.layers.append(RelationGNNLayer(self.d, self.nh, self.pos))

        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.emb_meta_rel)

    def forward(self, graph_rel, emb_rel, pos_rel):
        if self.pos != "none":
            emb_rel = torch.cat((emb_rel, pos_rel), -1)
        
        if len(graph_rel) > 0:
            for layer in self.layers:
                emb_rel = layer(graph_rel, emb_rel, self.emb_meta_rel)
                if self.pos != "none":
                    emb_rel = torch.cat((emb_rel, pos_rel), -1)

        return emb_rel
### \phi_\text{rel} (Section 4.2) ###


### Layer of \phi_\text{sup} (Section 4.3) ###
class SupportGNNLayer(nn.Module):
    def __init__(self, d, nh, pos):
        super(SupportGNNLayer, self).__init__()

        self.d = d
        self.nh = nh
        self.pos = pos

        assert self.d % self.nh == 0
        self.dh = self.d // self.nh
        if self.pos == "none":
            self.do = self.d
        else:
            self.do = self.d // 2

        self.W_attn_h = nn.Linear(self.d, self.d, bias=False)
        self.W_attn_r = nn.Linear(self.d, self.d)
        self.W_attn_t = nn.Linear(self.d, self.d, bias=False)
        self.y_attn = nn.Parameter(torch.zeros(1, self.nh, self.dh))
        self.W_proj_h = nn.Linear(self.d, self.do, bias=False)
        self.W_proj_r = nn.Linear(self.d, self.do, bias=False)
        self.W_resi = nn.Linear(self.d, self.do)

        self.layer_norm = nn.LayerNorm(self.do)
        self.act = nn.LeakyReLU(0.2)

        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.W_attn_h.weight, gain=gain)
        nn.init.xavier_normal_(self.W_attn_r.weight, gain=gain)
        nn.init.xavier_normal_(self.W_attn_t.weight, gain=gain)
        nn.init.xavier_normal_(self.y_attn, gain=gain)
        nn.init.xavier_normal_(self.W_proj_h.weight, gain=gain)
        nn.init.xavier_normal_(self.W_proj_r.weight, gain=gain)
        nn.init.xavier_normal_(self.W_resi.weight, gain=gain)

        nn.init.zeros_(self.W_attn_r.bias)
        nn.init.zeros_(self.W_resi.bias)

    def forward(self, graph, emb_ent, emb_rel, seen_ent):
        num_ent = len(emb_ent)

        idx_h = graph[:, 0].long()
        idx_r = graph[:, 1].long()
        idx_t = graph[:, 2].long()

        ### Compute attention weights ###
        attn_mat = self.W_attn_h(emb_ent[idx_h]) + self.W_attn_r(emb_rel)[idx_r] + self.W_attn_t(emb_ent[idx_t])
        attn_val_raw = (self.y_attn * self.act(attn_mat.view(-1, self.nh, self.dh))).sum(-1, keepdim=True)
        beta = torch.sigmoid(attn_val_raw)
        ### Compute attention weights ###

        ### Compute message ###
        msg_mat = self.W_proj_h(emb_ent[idx_h]) + self.W_proj_r(emb_rel)[idx_r]
        msg = beta * msg_mat.view(-1, self.nh, self.do // self.nh)
        ### Compute message ###


        ### Aggregation ###
        resi = torch.zeros(num_ent, self.do).cuda()
        resi[seen_ent > 0] = self.W_resi(emb_ent[seen_ent > 0])

        output = resi.index_add(0, idx_t, msg.flatten(1, -1))
        output = self.act(self.layer_norm(output))
        ### Aggregation ###

        return output
### Layer of \phi_\text{sup} (Section 4.3) ###


### \phi_\text{sup} (Section 4.3) ###
class SupportGNN(nn.Module):
    def __init__(self, nl, d, nh, pos):
        super(SupportGNN, self).__init__()

        self.nl = nl
        self.d = d
        self.nh = nh
        self.pos = pos

        self.layers = nn.ModuleList()
        for _ in range(self.nl):
            self.layers.append(SupportGNNLayer(self.d, self.nh, self.pos))

    def forward(self, graph, offset, pos_ent, emb_rel):
        if len(graph) == 0:
            return torch.zeros(1, 2 * self.d).cuda()

        if self.pos == "none":
            emb_ent = torch.zeros(offset[-1], self.d).cuda()
        else:
            emb_ent = torch.zeros(offset[-1], self.d // 2).cuda()
        emb_ent[offset[:-1]] = 1

        seen_ent = torch.zeros(offset[-1]).cuda()
        seen_ent[offset[:-1]] = 1

        if self.pos != "none":
            emb_ent = torch.cat((emb_ent, pos_ent), -1)

        for layer in self.layers:
            graph_idx = graph[seen_ent[graph[:, 0]] > 0]
            seen_ent[graph_idx[:, 2]] = 1
            emb_ent = layer(graph_idx, emb_ent, emb_rel, seen_ent)
            if self.pos != "none":
                emb_ent = torch.cat((emb_ent, pos_ent), -1)

        ### Computing task representations ###
        emb_sup_h = emb_ent[offset[:-1]]
        emb_sup_t = emb_ent[offset[:-1] + 1]

        emb_sup = torch.cat((emb_sup_h, emb_sup_t), -1)
        emb_sup_mean = torch.mean(emb_sup, 0, keepdim=True)
        
        beta = torch.softmax(torch.sum(emb_sup * emb_sup_mean, -1, keepdim=True), 0)
        attn_vec = torch.sum(beta * emb_sup, 0, keepdim=True)
        ### Computing task representations ###

        return attn_vec
### \phi_\text{sup} (Section 4.3) ###


### Layer of \phi_\text{ent} (Section 4.4) ###
class EntityGNNLayer(nn.Module):
    def __init__(self, d, nh, pos):
        super(EntityGNNLayer, self).__init__()

        self.d = d
        self.nh = nh
        self.pos = pos

        assert self.d % self.nh == 0
        self.dh = self.d // self.nh
        if self.pos == "none":
            self.do = self.d
        else:
            self.do = self.d // 2

        self.W_attn_h = nn.Linear(self.d, self.d, bias=False)
        self.W_attn_r = nn.Linear(self.d, self.d, bias=False)
        self.W_attn_t = nn.Linear(self.d, self.d, bias=False)
        self.W_attn_a = nn.Linear(2 * self.d, self.d)
        self.y_attn = nn.Parameter(torch.zeros(1, self.nh, self.dh))
        self.W_proj_h = nn.Linear(self.d, self.do, bias=False)
        self.W_proj_r = nn.Linear(self.d, self.do, bias=False)
        self.W_resi = nn.Linear(self.d, self.do)

        self.layer_norm = nn.LayerNorm(self.do)
        self.act = nn.LeakyReLU(0.2)

        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.W_attn_h.weight, gain=gain)
        nn.init.xavier_normal_(self.W_attn_r.weight, gain=gain)
        nn.init.xavier_normal_(self.W_attn_t.weight, gain=gain)
        nn.init.xavier_normal_(self.W_attn_a.weight, gain=gain)
        nn.init.xavier_normal_(self.y_attn, gain=gain)
        nn.init.xavier_normal_(self.W_proj_h.weight, gain=gain)
        nn.init.xavier_normal_(self.W_proj_r.weight, gain=gain)
        nn.init.xavier_normal_(self.W_resi.weight, gain=gain)

        nn.init.zeros_(self.W_attn_a.bias)
        nn.init.zeros_(self.W_resi.bias)

    def forward(self, graph, emb_ent, emb_rel, attn_vec, seen_ent):
        num_ent = len(emb_ent)

        idx_h = graph[:, 0]
        idx_r = graph[:, 1]
        idx_t = graph[:, 2]

        ### Compute attention weights ###
        attn_mat = self.W_attn_h(emb_ent[idx_h]) + self.W_attn_r(emb_rel)[idx_r] + self.W_attn_t(emb_ent[idx_t]) + self.W_attn_a(attn_vec)
        attn_val_raw = (self.y_attn * self.act(attn_mat.view(-1, self.nh, self.dh))).sum(-1, keepdim=True)
        beta = torch.sigmoid(attn_val_raw)
        ### Compute attention weights ###

        ### Compute message ###
        msg_mat = self.W_proj_h(emb_ent[idx_h]) + self.W_proj_r(emb_rel)[idx_r]
        msg = beta * msg_mat.view(-1, self.nh, self.do // self.nh)
        ### Compute message ###

        ### Aggregation ###
        resi = torch.zeros(num_ent, self.do).cuda()
        resi[seen_ent > 0] = self.W_resi(emb_ent[seen_ent > 0])

        output = resi.index_add(0, idx_t, msg.flatten(1, -1))
        output = self.act(self.layer_norm(output))
        ### Aggregation ###

        return output
### Layer of \phi_\text{ent} (Section 4.4) ###


### \phi_\text{ent} (Section 4.4) ###
class EntityGNN(nn.Module):
    def __init__(self, nl, d, nh, pos):
        super(EntityGNN, self).__init__()

        self.nl = nl
        self.d = d
        self.nh = nh
        self.pos = pos

        self.layers = nn.ModuleList()
        for _ in range(self.nl):
            self.layers.append(EntityGNNLayer(self.d, self.nh, self.pos))

    def forward(self, graph, offset, pos_ent, emb_rel, attn_vec):
        if self.pos == "none":
            emb_ent = torch.zeros(offset[-1], self.d).cuda()
        else:
            emb_ent = torch.zeros(offset[-1], self.d // 2).cuda()
        emb_ent[offset[:-1]] = 1

        if len(graph) == 0:
            return emb_ent

        seen_ent = torch.zeros(offset[-1]).cuda()
        seen_ent[offset[:-1]] = 1

        for layer in self.layers:
            if self.pos != "none":
                emb_ent = torch.cat((emb_ent, pos_ent), -1)

            graph_idx = graph[seen_ent[graph[:, 0]] > 0]
            seen_ent[graph_idx[:, 2]] = 1
            emb_ent = layer(graph_idx, emb_ent, emb_rel, attn_vec, seen_ent)
        
        return emb_ent
### \phi_\text{ent} (Section 4.4) ###


### FIETA ###
class Ours(nn.Module):
    def __init__(self, nlr, nls, nle, d, nh, p, pos):
        super(Ours, self).__init__()

        self.nlr = nlr
        self.nls = nls
        self.nle = nle
        self.d = d
        self.nh = nh
        self.p = p
        self.pos = pos

        self.relationGNN = RelationGNN(self.nlr, self.d, self.nh, self.pos)
        self.supportGNN = SupportGNN(self.nle, self.d, self.nh, self.pos)
        self.entityGNN = EntityGNN(self.nle, self.d, self.nh, self.pos)

        self.W_1 = nn.Linear(2 * self.d, 2 * self.d)
        self.W_2 = nn.Linear(2 * self.d, 2 * self.d, bias=False)

        self.layer_norm = nn.LayerNorm(2 * self.d)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(self.p)

        if self.pos == "lappe":
            self.W_le = nn.Linear(1, self.d)
            self.W_ve = nn.Linear(1, self.d)
            self.W_pe = nn.Linear(self.d, self.d // 2)

            self.W_lr = nn.Linear(1, self.d)
            self.W_vr = nn.Linear(1, self.d)
            self.W_pr = nn.Linear(self.d, self.d // 2)

        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W_1.weight, gain=gain)
        nn.init.xavier_normal_(self.W_2.weight, gain=gain)

        nn.init.zeros_(self.W_1.bias)
        
        if self.pos == "lappe":
            nn.init.xavier_normal_(self.W_le.weight, gain=gain)
            nn.init.xavier_normal_(self.W_ve.weight, gain=gain)
            nn.init.xavier_normal_(self.W_pe.weight, gain=gain)
            nn.init.xavier_normal_(self.W_lr.weight, gain=gain)
            nn.init.xavier_normal_(self.W_vr.weight, gain=gain)
            nn.init.xavier_normal_(self.W_pr.weight, gain=gain)

            nn.init.zeros_(self.W_le.bias)
            nn.init.zeros_(self.W_ve.bias)
            nn.init.zeros_(self.W_pe.bias)
            nn.init.zeros_(self.W_lr.bias)
            nn.init.zeros_(self.W_vr.bias)
            nn.init.zeros_(self.W_pr.bias)

    def mlp(self, x):
        output = self.W_1(self.drop(self.act(x)))
        output = self.W_2(self.drop(self.act(output)))
        output = self.layer_norm(x + output)
        return output

    def pos_encoder(self, l_ent, v_ent, l_rel, v_rel):
        pos_ent = self.W_le(l_ent.unsqueeze(0).unsqueeze(-1)) + self.W_ve(v_ent.unsqueeze(-1))
        pos_ent = self.W_pe(self.act(pos_ent))
        pos_ent = torch.sum(pos_ent, 1)

        pos_rel = self.W_lr(l_rel.unsqueeze(0).unsqueeze(-1)) + self.W_vr(v_rel.unsqueeze(-1))
        pos_rel = self.W_pr(self.act(pos_rel))
        pos_rel = torch.sum(pos_rel, 1)

        return pos_ent, pos_rel

    def compute_emb_rel(self, graph_rel, init_emb_rel, pos_rel):
        return self.relationGNN(graph_rel, init_emb_rel, pos_rel)

    def compute_attn_vec(self, graph, offset, entities, emb_rel, pos_ent):
        if self.pos != "none":
            pos_ent = pos_ent[entities]
        return self.supportGNN(graph, offset, pos_ent, emb_rel)

    def compute_emb_ent(self, graph, offset, entities, emb_rel, attn_vec, pos_ent):
        if self.pos != "none":
            pos_ent = pos_ent[entities]
        return self.entityGNN(graph, offset, pos_ent, emb_rel, attn_vec)

    ### Attention-based Scoring ###
    def predict(self, emb_s, offset_s, entities_s, emb_q, entities_q, query, cand, num_ent, pos_ent):
        bs = len(query)

        if offset_s[-1] == 0:
            emb_sup_h = torch.ones(1, self.d).cuda()
            emb_sup_t = torch.zeros(1, self.d).cuda()
        else:
            if self.pos != "none":
                emb_s = torch.cat((emb_s, pos_ent[entities_s]), -1)
            emb_sup_h = emb_s[offset_s[:-1]]
            emb_sup_t = emb_s[offset_s[:-1] + 1]

        if self.pos == "none":
            emb_ent = torch.zeros(bs * num_ent, self.d).cuda()
        else:
            emb_ent = torch.zeros(bs * num_ent, self.d // 2).cuda()

        if len(entities_q) > 0:
            emb_ent[entities_q] = emb_q
        
        if self.pos == "none":
            emb_ent = emb_ent.view(bs, -1, self.d)
        else:
            emb_ent = emb_ent.view(bs, -1, self.d // 2)
            emb_ent = torch.cat((emb_ent, pos_ent.unsqueeze(0).repeat(bs, 1, 1)), -1)

        emb_cand_h = emb_ent[torch.arange(bs).cuda(), query[:, 0]].unsqueeze(1)
        emb_cand_t = emb_ent[:, cand]

        ### Compute triplet encodings ###
        emb_sup = torch.cat((emb_sup_h, emb_sup_t), -1)
        emb_cand = torch.cat((emb_cand_h.expand_as(emb_cand_t), emb_cand_t), -1)

        emb_sup = self.mlp(emb_sup).unsqueeze(0).unsqueeze(2)
        emb_cand = self.mlp(emb_cand).unsqueeze(1)
        ### Compute triplet encodings ###
        
        ### Compute attention weights ###
        scores_raw = torch.sum(emb_sup * emb_cand, -1)
        beta = torch.softmax(scores_raw, 1)
        ### Compute attention weights ###

        ### Obtain the final scores ###
        scores = torch.sum(beta * scores_raw, 1)
        ### Obtain the final scores ###

        return scores
    ### Attention-based Scoring ###
### FIETA ###
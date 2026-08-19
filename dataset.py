from relgraph import create_relation_graph

import random
import torch
import pickle

class KG(object):
    def __init__(self, data, split, intrain=True):
        self.data = data
        self.split = split
        self.intrain = intrain

        assert self.split in ["train", "valid", "test"]
        if self.split in ["valid", "test"]:
            assert not self.intrain

        self.read_background()
        self.read_tasks()
        self.read_candidates()

        self.relgraph = create_relation_graph(self.background, len(self.id2ent), len(self.id2rel)).cuda()

        self.background = torch.tensor(self.background).cuda()

    def remove_duplicate(self, x):
        return list(dict.fromkeys(x))

    def read_background(self):
        if self.split == "train":
            with open(f"./data/{self.data}/bg-train.txt", "r") as f:
                background_raw = [line.strip().split("\t") for line in f.readlines()]
        else:
            with open(f"./data/{self.data}/bg-inference.txt", "r") as f:
                background_raw = [line.strip().split("\t") for line in f.readlines()]

        id2ent, id2rel = [], []
        for h, r, t in background_raw:
            id2ent += [h, t]
            id2rel += [r, r + "_inv"]

        self.id2ent = self.remove_duplicate(id2ent)
        self.id2rel = self.remove_duplicate(id2rel)
        self.ent2id = {ent: idx for idx, ent in enumerate(self.id2ent)}
        self.rel2id = {rel: idx for idx, rel in enumerate(self.id2rel)}
        self.num_ent = len(self.id2ent)
        self.num_rel = len(self.id2rel)
        del id2ent, id2rel

        print(f"[{self.split.upper()}] {len(self.id2ent)} entities")
        print(f"[{self.split.upper()}] {len(self.id2rel)} relations")

        self.background = []
        for h, r, t in background_raw:
            self.background.append([self.ent2id[h], self.rel2id[r], self.ent2id[t]])
            self.background.append([self.ent2id[t], self.rel2id[r + "_inv"], self.ent2id[h]])
        del background_raw

        print(f"[{self.split.upper()}] {len(self.background)} triplets")

    def read_tasks(self):
        with open(f"./data/{self.data}/task-{self.split}.txt", "r") as f:
            tasks_raw = [line.strip().split("\t") for line in f.readlines()]
        if self.intrain:
            with open(f"./data/{self.data}/bg-train.txt", "r") as f:
                tasks_raw += [line.strip().split("\t") for line in f.readlines()]

        self.id2task = self.remove_duplicate([r for _, r, _ in tasks_raw])
        self.task2id = {task: idx for idx, task in enumerate(self.id2task)}
        print(f"[{self.split.upper()}] {len(self.id2task)} tasks")

        self.tasks = dict()
        for h, r, t in tasks_raw:
            if h not in self.ent2id or t not in self.ent2id:
                continue

            if self.task2id[r] not in self.tasks:
                self.tasks[self.task2id[r]] = []
            
            self.tasks[self.task2id[r]].append([self.ent2id[h], self.ent2id[t]])
        del tasks_raw

    def read_candidates(self):
        with open(f"./data/{self.data}/candidates_intrain.txt", "r") as f:
            candidates_raw = [line.strip().split("\t") for line in f.readlines()]

        self.candidates = dict()
        for r, t in candidates_raw:
            if r not in self.task2id or t not in self.ent2id:
                continue

            if self.task2id[r] not in self.candidates:
                self.candidates[self.task2id[r]] = []

            self.candidates[self.task2id[r]].append(self.ent2id[t])
        del candidates_raw

    def compute_hop(self, e, num_ent, edges, L):
        hop = torch.ones(num_ent).cuda() * (L + 1)
        hop[e] = 0

        for l in range(L):
            list_h = edges[:, 0][hop[edges[:, -1]] == l]
            list_t = edges[:, -1][hop[edges[:, 0]] == l]

            hop[list_h[hop[list_h] == L + 1]] = l + 1
            hop[list_t[hop[list_t] == L + 1]] = l + 1

        return hop


class TrainKG(KG):
    def __init__(self, data, K, nls, nle, bs):
        super(TrainKG, self).__init__(data, "train")
        self.K = K
        self.nls = nls
        self.nle = nle
        self.bs = bs

        self.list_task = list(range(len(self.id2task)))
        self.idx_batch = -1

    def increase_batch_idx(self):
        self.idx_batch += 1
        if self.idx_batch % len(self.id2task) == 0:
            self.idx_batch = 0
            random.shuffle(self.list_task)

        if len(self.tasks[self.list_task[self.idx_batch]]) < 20 or len(self.candidates[self.list_task[self.idx_batch]]) < 20:
            self.increase_batch_idx()

    def extract_graph_a(self, background, r, h, t, offset):
        hop_h = self.compute_hop(h, self.num_ent, background, self.nls).cuda()
        dist = hop_h[t]
        if dist == 0:
            return [], offset, []
        if dist > self.nls:
            return [], offset, []

        hop_t = self.compute_hop(t, self.num_ent, background, self.nls).cuda()

        cri_hop = hop_h + hop_t <= self.nls
        cri_hop[h] = False
        cri_hop[t] = False

        indices = torch.nonzero(cri_hop).flatten()
        entities = torch.cat((torch.tensor([h, t]).cuda(), indices), 0).long()

        map_ent = -torch.ones(self.num_ent).long().cuda()
        map_ent[entities] = torch.arange(len(entities)).cuda() + offset

        cri = hop_h[background[:, 0]] + hop_t[background[:, 2]] < self.nls
        cri_inv = hop_t[background[:, 0]] + hop_h[background[:, 2]] < self.nls
        idx = torch.logical_or(cri, cri_inv)
        graph = background[idx]
        graph[:, ::2] = map_ent[graph[:, ::2]]

        return graph, offset + len(entities), entities

    def extract_graph_s(self, background, r, h, t, offset):
        hop_h = self.compute_hop(h, self.num_ent, background, self.nle).cuda()
        dist = hop_h[t]
        if dist == 0:
            return [], offset, []
        if dist > self.nle:
            return [], offset, []

        cri_hop = hop_h <= self.nle
        cri_hop[h] = False
        cri_hop[t] = False

        indices = torch.nonzero(cri_hop).flatten()
        entities = torch.cat((torch.tensor([h, t]).cuda(), indices), 0).long()

        map_ent = -torch.ones(self.num_ent).long().cuda()
        map_ent[entities] = torch.arange(len(entities)).cuda() + offset

        idx = hop_h[background[:, 0]] < self.nle
        graph = background[idx]
        graph[:, ::2] = map_ent[graph[:, ::2]]

        return graph, offset + len(entities), entities

    def extract_graph_q(self, background, r, h, t, offset):
        hop_h = self.compute_hop(h, self.num_ent, background, self.nle).cuda()
        dist = hop_h[t]
        if dist == 0:
            return [], offset, []
        if dist > self.nle:
            return [], offset, []

        cri_hop = hop_h <= self.nle
        cri_hop[h] = False

        indices = torch.nonzero(cri_hop).flatten()
        entities = torch.cat((torch.tensor([h]).cuda(), indices), 0).long()

        map_ent = -torch.ones(self.num_ent).long().cuda()
        map_ent[entities] = torch.arange(len(entities)).cuda() + offset

        idx = hop_h[background[:, 0]] < self.nle
        graph = background[idx]
        graph[:, ::2] = map_ent[graph[:, ::2]]

        return graph, offset + len(entities), entities

    def get_batch(self):
        self.increase_batch_idx()

        r = self.list_task[self.idx_batch]
        cand = self.candidates[r]
        tasks = self.tasks[r]
        random.shuffle(tasks)

        support, remaining = tasks[:self.K], tasks[self.K:]
        if len(remaining) < self.bs:
            query_all = remaining + [random.choice(remaining) for _ in range(self.bs - len(remaining))]
        else:
            query_all = remaining[:self.bs]

        if self.id2task[r] in self.rel2id:
            id_r = self.rel2id[self.id2task[r]]
            assert id_r % 2 == 0

            idx_bg = torch.logical_and(self.background[:, 1] != id_r, self.background[:, 1] != id_r + 1)
            background = self.background[idx_bg]
        else:
            background = self.background

        graph_a, offset_a, entities_a = [], [0], []
        for h, t in support:
            graph_inst, offset_inst, entities_inst = self.extract_graph_a(background, r, h, t, offset_a[-1])
            
            if len(graph_inst) > 0:
                graph_a.append(graph_inst)
                offset_a.append(offset_inst)
                entities_a.append(entities_inst)
        
        if len(graph_a) == 0:
            return self.get_batch()

        support_tensor = torch.tensor([[h, self.num_rel, t] for h, t in support]).cuda()
        background = torch.cat((background, support_tensor))

        graph_s, offset_s, entities_s = [], [0], []
        for h, t in support:
            graph_inst, offset_inst, entities_inst = self.extract_graph_s(background, r, h, t, offset_s[-1])
            
            if len(graph_inst) > 0:
                graph_s.append(graph_inst)
                offset_s.append(offset_inst)
                entities_s.append(entities_inst)

        graph_q, offset_q, entities_q, query = [], [0], [], []
        for h, t in query_all:
            graph_inst, offset_inst, entities_inst = self.extract_graph_q(background, r, h, t, offset_q[-1])
            
            if len(graph_inst) > 0:
                graph_q.append(graph_inst)
                offset_q.append(offset_inst)
                entities_q.append(entities_inst + len(query) * self.num_ent)
                query.append([h, t])
        
        if len(graph_q) == 0:
            return self.get_batch()

        graph_a = torch.cat(graph_a, 0)
        offset_a = torch.tensor(offset_a).cuda()
        entities_a = torch.cat(entities_a, 0)

        graph_s = torch.cat(graph_s, 0)
        offset_s = torch.tensor(offset_s).cuda()
        entities_s = torch.cat(entities_s, 0)

        graph_q = torch.cat(graph_q, 0)
        offset_q = torch.tensor(offset_q).cuda()
        entities_q = torch.cat(entities_q, 0)

        idx_gt = []
        for _, t in query:
            assert t in cand
            idx_gt.append(cand.index(t))

        query = torch.tensor(query).long().cuda()
        cand = torch.tensor(cand).long().cuda()
        idx_gt = torch.tensor(idx_gt).long().cuda()
        
        return graph_a, offset_a, entities_a, graph_s, offset_s, entities_s, graph_q, offset_q, entities_q, query, cand, idx_gt


class InferenceKG(KG):
    def __init__(self, data, K, nls, nle, split):
        super(InferenceKG, self).__init__(data, split, False)
        self.K = K
        self.nls = nls
        self.nle = nle
        self.split = split
        
        self.list_task = list(range(len(self.id2task)))
        self.idx_batch = -1

        self.create_filter_dict()

    def create_filter_dict(self):
        self.filter_dict = dict()
        for r in self.tasks:
            for h, t in self.tasks[r]:
                if (h, r) not in self.filter_dict:
                    self.filter_dict[(h, r)] = []
                self.filter_dict[(h, r)].append(t)

    def extract_graph_a(self, background, r, h, t, offset):
        hop_h = self.compute_hop(h, self.num_ent, background, self.nls).cuda()
        dist = hop_h[t]
        if dist == 0:
            return [], offset, []
        if dist > self.nls:
            return [], offset, []

        hop_t = self.compute_hop(t, self.num_ent, background, self.nls).cuda()

        cri_hop = hop_h + hop_t <= self.nls
        cri_hop[h] = False
        cri_hop[t] = False

        indices = torch.nonzero(cri_hop).flatten()
        entities = torch.cat((torch.tensor([h, t]).cuda(), indices), 0).long()

        map_ent = -torch.ones(self.num_ent).long().cuda()
        map_ent[entities] = torch.arange(len(entities)).cuda() + offset

        cri = hop_h[background[:, 0]] + hop_t[background[:, 2]] < self.nls
        cri_inv = hop_t[background[:, 0]] + hop_h[background[:, 2]] < self.nls
        idx = torch.logical_or(cri, cri_inv)
        graph = background[idx]
        graph[:, ::2] = map_ent[graph[:, ::2]]

        return graph, offset + len(entities), entities

    def extract_graph_s(self, background, r, h, t, offset):
        hop_h = self.compute_hop(h, self.num_ent, background, self.nle).cuda()
        dist = hop_h[t]

        cri_hop = hop_h <= self.nle
        cri_hop[h] = False
        cri_hop[t] = False

        indices = torch.nonzero(cri_hop).flatten()
        entities = torch.cat((torch.tensor([h, t]).cuda(), indices), 0).long()

        map_ent = -torch.ones(self.num_ent).long().cuda()
        map_ent[entities] = torch.arange(len(entities)).cuda() + offset

        idx = hop_h[background[:, 0]] < self.nle
        graph = background[idx]
        graph[:, ::2] = map_ent[graph[:, ::2]]

        return graph, offset + len(entities), entities

    def extract_graph_q(self, background, r, h, t, offset):
        hop_h = self.compute_hop(h, self.num_ent, background, self.nle).cuda()

        cri_hop = hop_h <= self.nle
        cri_hop[h] = False

        indices = torch.nonzero(cri_hop).flatten()
        entities = torch.cat((torch.tensor([h]).cuda(), indices), 0).long()

        map_ent = -torch.ones(self.num_ent).long().cuda()
        map_ent[entities] = torch.arange(len(entities)).cuda() + offset

        idx = hop_h[background[:, 0]] < self.nle
        graph = background[idx]
        graph[:, ::2] = map_ent[graph[:, ::2]]

        return graph, offset + len(entities), entities

    def get_batch_support(self, r):
        cand = self.candidates[r]
        tasks = self.tasks[r]

        support = tasks[:self.K]

        graph_a, offset_a, entities_a = [], [0], []
        for h, t in support:
            graph_inst, offset_inst, entities_inst = self.extract_graph_a(self.background, r, h, t, offset_a[-1])
            
            if len(graph_inst) > 0:
                graph_a.append(graph_inst)
                offset_a.append(offset_inst)
                entities_a.append(entities_inst)

        support_tensor = torch.tensor([[h, self.num_rel, t] for h, t in support]).cuda()
        background = torch.cat((self.background, support_tensor))

        graph_s, offset_s, entities_s = [], [0], []
        for h, t in support:
            graph_inst, offset_inst, entities_inst = self.extract_graph_s(background, r, h, t, offset_s[-1])
            
            if len(graph_inst) > 0:
                graph_s.append(graph_inst)
                offset_s.append(offset_inst)
                entities_s.append(entities_inst)

        if len(graph_a) > 0:
            graph_a = torch.cat(graph_a, 0)
            offset_a = torch.tensor(offset_a).cuda()
            entities_a = torch.cat(entities_a, 0)

        if len(graph_s) > 0:
            graph_s = torch.cat(graph_s, 0)
            offset_s = torch.tensor(offset_s).cuda()
            entities_s = torch.cat(entities_s, 0)

        return graph_a, offset_a, entities_a, graph_s, offset_s, entities_s

    def get_batch_query(self, r, idx_batch):
        cand = self.candidates[r]
        tasks = self.tasks[r]

        support = tasks[:self.K]
        remaining = tasks[self.K:]
        query = [remaining[idx_batch]]

        support_tensor = torch.tensor([[h, self.num_rel, t] for h, t in support]).cuda()
        background = torch.cat((self.background, support_tensor))

        graph_q, offset_q, entities_q = [], [0], []
        for idx, (h, t) in enumerate(query):
            graph_inst, offset_inst, entities_inst = self.extract_graph_q(background, r, h, t, offset_q[-1])
            
            if len(graph_inst) > 0:
                graph_q.append(graph_inst)
                offset_q.append(offset_inst)
                entities_q.append(entities_inst + idx * self.num_ent)
        
        if len(graph_q) > 0:
            graph_q = torch.cat(graph_q, 0)
            offset_q = torch.tensor(offset_q).cuda()
            entities_q = torch.cat(entities_q, 0)

        h, t = query[0]
        assert t in cand

        cand_query = [t]
        for e in cand:
            if e not in self.filter_dict[(h, r)]:
                cand_query.append(e)

        query = torch.tensor(query).long().cuda()
        cand = torch.tensor(cand_query).long().cuda()

        return graph_q, offset_q, entities_q, query, cand

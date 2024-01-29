from io import open
import os
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import tempfile

from asyncio import DatagramProtocol
import numpy as np
from scipy.sparse import *
from sklearn import preprocessing
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

import UVDecVectors
import utils
import evaluating_indicator

# 数据处理
class DataUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def read_data(self,filename=None):
        if filename is None:
            filename = os.path.join(self.model_path,"rating_test.dat")
        user,item,rate=[],[],[]
        with open(filename, "r", encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                duser, ditem, drate = line.strip().split()
                user.append(int(duser))
                item.append(int(ditem))
                rate.append(float(drate))
                line = fin.readline()
        return user,item,rate

# 设置种子
def set_rng_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 初始化特征向量
def init_embedding_vectors(ulen,vlen, args):
    U0 = np.zeros((ulen, args.d), dtype=np.float32)
    V0 = np.zeros((vlen, args.d), dtype=np.float32)
    # user
    for i in range(ulen):
        U0[i] =vectors= np.random.random([1, args.d])
        U0[i] = preprocessing.normalize(vectors, norm='l2') # l2归一化
    # item
    for i in range(vlen):
        V0[i] = vectors=  np.random.random([1, args.d])
        V0[i] = preprocessing.normalize(vectors, norm='l2')
    return U0,V0

# 从稠密矩阵构建graph
def nx_graph_from_biadjacency_matrix(M):
    # Give names to the nodes in the two node sets
    U = [ "u{}".format(i) for i in range(M.shape[0]) ]
    V = [ "v{}".format(i) for i in range(M.shape[1]) ]
    # Create the graph and add each set of nodes
    Graph = nx.Graph()
    Graph.add_nodes_from(U, bipartite=0)
    Graph.add_nodes_from(V, bipartite=1)
    # Find the non-zero indices in the biadjacency matrix to connect
    Graph.add_edges_from([ (U[i], V[j]) for i, j in zip(*M.nonzero()) ])
    return Graph

def computeResult(test_data,model,itemlist):
    test_user, test_item, test_rate = evaluating_indicator.read_data(test_data)# 读取测试集
    node_list_u_,node_list_v_={},{}
    i = 0
    for item in model.u.weight:
        node_list_u_[str(i)] = {}
        node_list_u_[str(i)]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1
    i = 0
    for item in model.v.weight:
        node_list_v_[str(itemlist[i])] = {}
        node_list_v_[str(itemlist[i])]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1

    f1, map, mrr, mndcg = evaluating_indicator.top_N(test_user,test_item,test_rate,node_list_u_,node_list_v_,top_n=10)
    print("f1:",f1,"map:",map,"mrr:",mrr,"mndcg:",mndcg)

# 邻居采样
class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        print("节点数量：", n)
        # assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)#[节点数量,邻居数量+1] 包括自己的情况
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)#[节点数量,邻居数量] 不包括自己的情况
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v))) 
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids
        # print("111:",self.nb_all)

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        # print("2222:",self.nb_all)
        return self.nb_all

# 线性层
class InputLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InputLinear, self).__init__()
        weight = nn.Parameter(torch.FloatTensor(in_dim,out_dim),requires_grad=True)
        bias = nn.Parameter(torch.FloatTensor(out_dim),requires_grad=True)
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1)) # 均匀分布采样
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        # nn.init.kaiming_uniform_(self.weight) # kaiming初始化
        # nn.init.zeros_(self.bias)
        
    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias  #WTu+b

# 路由卷积层
class RoutingConv(nn.Module):
    def __init__(self, dim, k, a, att, dev): 
        super(RoutingConv, self).__init__()
        assert dim % k == 0 # 向量维度要能整除k
        self.d, self.k = dim, k 
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)
        self.a = a
        self.att = att
        self.dev = dev


    def forward(self, x, neighbors, max_iter):

        dev = x.device 
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, delta_d = self.d, self.k, self.d // self.k

        z = torch.cat([x, self._cache_zero_d], dim=0) 
        z = z[neighbors].view(n, m, k, delta_d) 
        u = None

        zl0=z.size()[0]
        zl1=z.size()[1]
        # 初始化存储注意力系数的矩阵e
        e = torch.empty(zl0, zl1,k,1).to(self.dev)
        # 计算节点对之间的注意力系数
        # a_input = torch.cat([x.repeat(1, x.size(0)).view(x.size(0) * x.size(0), -1), 
        #                      x.repeat(x.size(0), 1)], dim=1)
        # e = torch.mm(a_input, self.att).sigmoid().view(x.size(0), x.size(0))
        act=0
        for i in range(zl0):
            for j in range(zl1):
                if(act%100000==0):
                    print(act)
                act=act+1
                a_input = torch.cat([x[i], z[i][j].view(-1)], dim=-1)
                e_ij = torch.mm(a_input.view(-1, 1).t(), self.att)
                e[i, j] = e_ij
        # 通过softmax函数将原始的注意力系数转换为有效的概率分布
        attention = F.softmax(e, dim=1)
        # 计算注意力加权的节点特征
        e_prime = torch.mul(attention, z)
        e_prime = torch.sum(e_prime, dim=3)


        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k) 
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3) 
            p = F.softmax(p, dim=2) 
            u = torch.sum(z * e_prime.view(n, m, k, 1), dim=1) 
            u = self.a*u + x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = F.normalize(u, dim=2) 
        return u.view(n, d)

# 二分图解离卷积网络
class BiDisen(nn.Module):
    def __init__(self, args,G,M,N,u0,v0,duser,ditemid,drate):
        super(BiDisen, self).__init__()
        self.k = args.k # k个潜在因子（胶囊网络），nhidden：k内隐藏单元数
        self.a = args.a
        self.b = args.b
        self.hid_dim = args.d

        dev = torch.device(args.dev)
        self.G,self.M,self.N= G.to(dev),M.to(dev),N.to(dev)
        self.M = self.M/torch.sum(self.M,1).unsqueeze(-1) 
        self.N = self.N/torch.sum(self.N,1).unsqueeze(-1) 
        self.mlen = self.M.size(0)
        self.nlen = self.N.size(0)
        self.duser,self.ditemid,self.drate = duser,ditemid,drate
        self.u = torch.nn.Embedding(ulen, args.d, dtype=torch.float).to(dev)
        self.v =  torch.nn.Embedding(vlen, args.d, dtype=torch.float).to(dev)
        self.u.to(dev)
        self.v.to(dev)
        self.u.weight.data = nn.Parameter(u0,requires_grad=True)
        self.v.weight.data = nn.Parameter(v0,requires_grad=True)
        self.u.weight.data = self.u.weight.data.to(dev)
        self.v.weight.data = self.v.weight.data.to(dev)
        # 初始化注意力参数att
        self.att = nn.Parameter(torch.zeros(size=(args.d*2, 1)),requires_grad=True).to(dev)
        self.att.data = self.att.data.to(dev)
        nn.init.xavier_uniform_(self.att.data, gain=1.414) 

        conv_ls = []
        for i in range(args.nlayer): 
            conv = RoutingConv(self.hid_dim, self.k,self.a,self.att,dev)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls

        self.mlp = nn.Linear(self.hid_dim,args.d)
        self.dropout = args.dropout
        self.routit = args.routit

    def _dropout(self, x):
        return F.dropout(x, self.dropout, training=self.training)

    def forward(self,neiberm,neibern,u,v):
        neiberm = neiberm.view(-1)  
        neibern = neibern.view(-1)  
        
        user_emb = self.u.weight.data
        item_emb = self.v.weight.data
        ui_emb = torch.cat([user_emb, item_emb], dim=0)  
        nbs = torch.cat([neiberm, neibern.add(self.mlen)], dim=0)
        for conv in self.conv_ls:
            ui_emb = conv(ui_emb, nbs, self.routit) 
           
        user_emb, item_emb = ui_emb[:self.mlen], ui_emb[self.mlen:]
        self.u.weight.data = user_emb
        self.v.weight.data = item_emb
        self.u.weight.data = self.u.weight.data.to(dev)
        self.v.weight.data = self.v.weight.data.to(dev)
        # 一阶KLloss
        pr = torch.mul(self.u(self.duser.to(dev)),self.v(self.ditemid.to(dev))) 
        pr = torch.sum(pr, dim=1) 
        positivebatch = F.logsigmoid(pr)
        # 二阶KLloss
        um = torch.mm(self.u.weight,self.u.weight.t())
        vm = torch.mm(self.v.weight,self.v.weight.t())
        tu = torch.sigmoid(um)
        logp_x_u = F.log_softmax(tu.float().to(dev), dim=-1)
        p_y_u = F.softmax(self.M.float().to(dev), dim=-1)
        kl_sum_u = F.kl_div(logp_x_u.float().to(dev), p_y_u.float().to(dev), reduction='sum') 
        tv = torch.sigmoid(vm)
        logp_x_v = F.log_softmax(tv.float().to(dev),dim=-1)
        p_y_v = F.softmax(self.N.float().to(dev),dim=-1)
        kl_sum_v = F.kl_div(logp_x_v.float().to(dev), p_y_v.float().to(dev), reduction='sum')
        
        loss =  -torch.mean(positivebatch) + self.b*kl_sum_u + self.b*kl_sum_v
        return self.u.weight,self.v.weight,loss,pr

if __name__ == '__main__':
    # 运行参数
    parser = ArgumentParser("BiDisen",formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
    parser.add_argument('--dataset', default=r'dblp',help='Input dataset name.')
    parser.add_argument('--d', default=128, type=int,help='embedding size.')
    parser.add_argument('--dev', default="cuda:0",help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--lr', type=float, default=0.005,help='Initial learning rate.') # 
    parser.add_argument('--reg', type=float, default=0.0036,help='Weight decay (L2 loss on parameters).') #权重衰减
    parser.add_argument('--nlayer', type=int, default=5,help='Number of conv layers.')
    parser.add_argument('--k', type=int, default=4,help='Maximum number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=32,help='Number of hidden units per capsule.')
    parser.add_argument('--dropout', type=float, default=0,help='0.35 Dropout rate (1 - keep probability).')
    parser.add_argument('--routit', type=int, default=5,help='Number of iterations when routing.') #路由时的迭代次数
    parser.add_argument('--a', type=float, default=0.00001,help='Controls the hardness of the assignmen.') #路由时的迭代次数
    parser.add_argument('--b', type=float, default=0.1,help='Loss allocation parameter.') #路由时的迭代次数
    parser.add_argument('--nbsz', type=int, default=128,help='Size of the sampled neighborhood.') #采样的邻域的大小
    parser.add_argument('--nepoch', type=int, default=25,help='Max number of epochs to train.') #最大迭代次数
    args = parser.parse_args()
    dev = torch.device(args.dev)
    # 读取数据
    print('========== 处理数据 ===========')
    model_path = os.path.join('./')
    dul = DataUtils(model_path)
    train_data = os.path.join('./data/',args.dataset,'rating_train.dat')
    test_data = os.path.join('./data/',args.dataset,'rating_test.dat')
    duser,ditem,drate = dul.read_data(train_data) 
    user,item = list(set(duser)), list(set(ditem)) # 去重
    duserid,ditemid = [0] * len(duser), [0] * len(ditem)
    for i in range(len(duser)):
        duserid[i] =  user.index(duser[i]) # 稠密矩阵项目列映射为其index 如：1246->1234
    for i in range(len(ditem)):
        ditemid[i] =  item.index(ditem[i])
    Rmat = csr_matrix((drate, (duserid, ditemid))) # csr稠密矩阵
    rsp = Rmat.todense()
    G = torch.from_numpy(rsp)# 稀疏矩阵，行为用户，列为项目
    Gmean = torch.mean(G.float())
    ulen,vlen = G.shape[0],len(item)
    print('len(ditem), len(item),Rmat.size,G.shape:\n', len(ditem), len(item),Rmat.size, G.shape)
    print('========== 同构矩阵nx图 ===========')
    G = G.float()
    M,N = torch.mm(G, G.t()),torch.mm(G.t(),G)
    graphm = nx.from_numpy_array(M.numpy())
    graphn = nx.from_numpy_array(N.numpy())
    neiberm = NeibSampler(graphm,args.nbsz).sample().to(dev)  
    neibern = NeibSampler(graphn,args.nbsz).sample().to(dev)
    # set_rng_seed(23) #设置种子

    udat,vdat = utils.read_file(os.path.join('./uvinit/',args.dataset,'u-vec.dat'),os.path.join('./uvinit/',args.dataset,'v-vec.dat'))
    u0,v0= torch.tensor(udat), torch.tensor(vdat)


    print('========== 模型 ===========')
    duserid = torch.tensor(duserid, dtype=torch.long).to(dev)
    ditemid = torch.tensor(ditemid, dtype=torch.long).to(dev)
    drate =  torch.tensor(drate,dtype=torch.float).to(dev)

    model = BiDisen(args,G,M,N,u0,v0,duserid,ditemid,drate).to(dev)
    optmz = optim.Adam(model.parameters(),lr=args.lr)
    print('========== 迭代训练 ===========')
    for t in range(args.nepoch):
        model.train()
        optmz.zero_grad()
        u,v,loss,pr = model(neiberm,neibern,u0,v0) # 取得用户和项目的表示向量

        print('+++第%2d次loss: %.4f' % (t,loss)) 
        # 计算效果
        computeResult(test_data,model,item)
        loss.backward()
        optmz.step()
        
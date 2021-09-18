import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

def SiLU(x): #NonLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)

def ReLU(x):
    return F.relu(x)

def NonLU(x):
    return ReLU(x)

class basicDNN(nn.Module):
    def __init__(self, inputFeatures, layers, nodes, pDropout):
        super(basicDNN, self).__init__()
        self.name = 'FC%dx%d_pdrop%.2f'%(layers, nodes, pDropout)
        fc=[]
        fc.append(nn.Linear(inputFeatures, nodes))
        fc.append(nn.ReLU())
        #fc.append(nn.Dropout(p=pDropout))
        for l in range(layers):
            fc.append(nn.Linear(nodes, nodes))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(p=pDropout))
            #if l < layers-1: fc.append(nn.Dropout(p=pDropout))
        fc.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*fc)
        
    def forward(self, x, p, a):
        return self.net(x)


class basicCNN(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout):
        super(basicCNN, self).__init__()
        self.name = 'basicCNN_%d_%d_%d_%d_pdrop%.2f'%(dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  ##kernel=3
        self.conv1 = nn.Sequential(*[nn.Conv1d(     jetFeatures,        dijetFeatures, 2, stride=2), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv1d(   dijetFeatures,      quadjetFeatures, 2, stride=2), nn.ReLU()])
        self.conv3 = nn.Sequential(*[nn.Conv1d( quadjetFeatures, combinatoricFeatures, 3, stride=1), nn.ReLU()])

        self.line1 = nn.Sequential(*[nn.Linear(combinatoricFeatures, nodes), nn.ReLU()])
        self.line2 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line3 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line4 =                 nn.Linear(nodes, 1)

    def forward(self, x, a):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        
        x = self.line1(x)
        x = self.line2(x)
        x = self.line3(x)
        x = self.line4(x)
        return x


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, useOthJets=False):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        self.ks = 4 if useOthJets else 3
        self.nx = self.ks-1
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            

        # if we are using other jet info:
        # |1|2|o|1,2|3|4|o|3,4|1|3|o|1,3|2|4|o|2,4|1|4|o|1,4|2|3|o|2,3|  ##stride=4 kernel=4 reinforce dijet features with output of other jet LSTM
        #       |1,2|     |3,4|     |1,3|     |2,4|     |1,4|     |2,3|  
        self.conv = nn.Conv1d(self.nd, self.nd, self.ks, stride=self.ks)

    def forward(self, x, d):
        n = x.shape[0]
        d = torch.cat( (x[:,:, self.nx*0: self.nx*1], d[:,:,0].view(n, self.nd, 1),
                        x[:,:, self.nx*1: self.nx*2], d[:,:,1].view(n, self.nd, 1),
                        x[:,:, self.nx*2: self.nx*3], d[:,:,2].view(n, self.nd, 1),
                        x[:,:, self.nx*3: self.nx*4], d[:,:,3].view(n, self.nd, 1),
                        x[:,:, self.nx*4: self.nx*5], d[:,:,4].view(n, self.nd, 1),
                        x[:,:, self.nx*5: self.nx*6], d[:,:,5].view(n, self.nd, 1)), 2 )
        return self.conv(d)

class dijetResNetBlock(nn.Module):
    def __init__(self, dijetFeatures, useOthJets=False):
        super(dijetResNetBlock, self).__init__()
        self.nd = dijetFeatures
        self.reinforce1 = dijetReinforceLayer(self.nd, useOthJets)
        self.reinforce2 = dijetReinforceLayer(self.nd, useOthJets)
        # self.reinforce3 = dijetReinforceLayer(self.nd, useOthJets)
        # self.reinforce4 = dijetReinforceLayer(self.nd)
        # self.reinforce5 = dijetReinforceLayer(self.nd)
        # self.reinforce6 = dijetReinforceLayer(self.nd)


    def forward(self, x, d):
        d0 = d.clone()
        #d = NonLU(d)
        d = self.reinforce1(x, d)
        d = d+d0
        d = NonLU(d)
        d = self.reinforce2(x, d)
        #d2 = d.clone()
        d = d+d0
        # d = NonLU(d)
        # d = self.reinforce3(x, d)
        # d = d+d0
        # d = self.ReLU3(d)
        # d = self.reinforce4(x, d)
        # #d4 = d.clone()
        # d = d+d3
        # d = self.ReLU4(d)
        # d = self.reinforce5(x, d)
        # d = d+d3
        # d = self.ReLU5(d)
        # d = self.reinforce6(x, d)
        # d = d+d3
        # d = self.ReLU6(d)
        return d


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.conv = nn.Conv1d(self.nq, self.nq, 3, stride=3)

    def forward(self, x, q):
        n = x.shape[0]
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4:6], q[:,:,2].view(n,self.nq,1)), 2)
        return self.conv(q)


class quadjetResNetBlock(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetResNetBlock, self).__init__()
        self.nq = quadjetFeatures
        self.reinforce1 = quadjetReinforceLayer(self.nq)
        self.reinforce2 = quadjetReinforceLayer(self.nq)
        # self.reinforce3 = quadjetReinforceLayer(self.nq)
        # self.reinforce4 = quadjetReinforceLayer(self.nq)
        # self.reinforce5 = quadjetReinforceLayer(self.nq)
        # self.reinforce6 = quadjetReinforceLayer(self.nq)

    def forward(self, x, q):
        q0 = q.clone()
        #q = NonLU(q)
        q = self.reinforce1(x, q)
        q = q+q0
        q = NonLU(q)
        q = self.reinforce2(x, q)
        #q2 = q.clone()
        q = q+q0
        # q = NonLU(q)
        # q = self.reinforce3(x, q)
        # q = q+q0
        # q = self.ReLU3(q)
        # q = self.reinforce4(x, q)
        # #q4 = q.clone()
        # q = q+q0
        # q = self.ReLU4(q)
        # q = self.reinforce5(x, q)
        # q = q+q0
        # q = self.ReLU5(q)
        # q = self.reinforce6(x, q)
        # q = q+q0
        # q = self.ReLU6(q)
        return q


class jetRNN(nn.Module):
    def __init__(self, jetFeatures, hiddenFeatures):
        super(jetRNN, self).__init__()
        self.nj = jetFeatures
        self.nh = hiddenFeatures
        self.lstm = nn.LSTM(self.nj, self.nh, num_layers=1, batch_first=True)

    def forward(self, o):#j[event][jet][mu] l[event][nj]
        ls = (o[:,1,:]!=0).sum(dim=1) # count how many jets in each batch have pt > 0. pt==0 for padded entries
        ls = ls + torch.tensor(ls==0, dtype=torch.long).to("cuda") # add 1 to ls when there are no other jets to return the 
        js = torch.transpose(o,1,2) # switch jet and mu indices because RNN expects jet index before jet component index

        batch_size, seq_len, feature_len = js.size()
        hs, _ = self.lstm(js)
        
        hs = hs.contiguous().view(batch_size*seq_len, self.nh)
        idxs = [(l-1)*batch_size + i for i,l in enumerate(ls)]
        idxs = torch.tensor(idxs, dtype=torch.int64).to("cuda")
        hs = hs.index_select(0,idxs)
        hs = hs.view(batch_size,self.nh)
        return hs

class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillaryFeatures, useOthJets=False):
        super(ResNet, self).__init__()
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
        self.nq, self.nAq = quadjetFeatures, 2 #total quadjet features, engineered quadjet features
        self.nAv = nAncillaryFeatures
        self.nc = combinatoricFeatures ##self.nq+self.nAv
        self.name = 'ResNet'+('+LSTM' if useOthJets else '')+'_%d_%d_%d'%(dijetFeatures, quadjetFeatures, self.nc)

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        self.otherJetRNN = None
        if useOthJets:
            self.otherJetRNN = jetRNN(self.nj,self.nd)

        self.toDijetFeatureSpace = nn.Conv1d(self.nj, self.nd, 1)
        self.dijetAncillaryEmbedder = nn.Conv1d(self.nAd, self.nd, 1)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        kernStride=2
        if useOthJets: kernStride=3 #add otherJetRNN output as a pixel next to each jet pair
        self.dijetBuilder = nn.Conv1d(self.nd, self.nd, kernStride, stride=kernStride)
        # ancillary dijet features get appended to output of dijetBuilder

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nd, useOthJets)

        self.toQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
        self.quadjetAncillaryEmbedder = nn.Conv1d(self.nAq, self.nq, 1)
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Conv1d(self.nq, self.nq, 2, stride=2)
        # ancillary quadjet features get appended to output of quadjetBuilder

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nq)
        #self.viewAncillaryEmbedder = nn.Conv1d(self.nAv, self.nc)
        # ancillary view features get appended to output of quadjetResNetBlock

        self.viewConv1 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv2 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewSelector = nn.Conv1d(self.nc, 1, 3, stride=1)

    def rotate(self, p, R): # p[event, mu, jet], mu=2 is phi
        pR = p.clone()
        pR[:,2,:] = (pR[:,2,:] + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        return pR

    def flipPhi(self, p): # p[event, mu, jet], mu=2 is phi
        pF = p.clone()
        pF[:,2,:] = -1*pF[:,2,:]
        return pF

    def flipEta(self, p): # p[event, mu, jet], mu=1 is eta
        pF = p.clone()
        pF[:,1,:] = -1*pF[:,1,:]
        return pF

    def invPart(self,p,o,da,qa):
        n = p.shape[0]
        p = self.toDijetFeatureSpace(p)

        #if self.otherJetRNN:
        #    hs = self.otherJetRNN(o)
        #    hs = hs.view(n,self.nd,1)        
        #    p = torch.cat( (p[:,:,0:2], hs, p[:,:,2:4], hs, p[:,:,4:6], hs, p[:,:,6:8], hs, p[:,:,8:10], hs, p[:,:,10:12], hs), 2)

        d = self.dijetBuilder(p)
        d = NonLU(d)
        #d = torch.cat( (d, da), 1 ) # manually add dijet mass and dRjj to dijet feature space
        d = d + self.dijetAncillaryEmbedder(da)
        d = self.dijetResNetBlock(p,d)
        
        d = self.toQuadjetFeatureSpace(d)
        q = self.quadjetBuilder(d)
        q = NonLU(q)
        #q = torch.cat( (q, qa), 1) # manually add features to quadjet feature space
        q = q + self.quadjetAncillaryEmbedder(qa)
        q = self.quadjetResNetBlock(d,q) 
        return q

    def forward(self, x, p, o, da, qa, va):#, js, ls):
        n = p.shape[0]
        da = torch.cat( (da[:,0:6].view(n,1,6), da[:,6:12].view(n,1,6)), 1) #format dijet masses and dRjjs 
        qa = torch.cat( (qa[:,0:3].view(n,1,3), qa[:,3: 6].view(n,1,3)), 1) #format delta R between boson candidates and mZH's for quadjet feature space
        #qa = qa[:,0:3].view(n,1,3) #format delta R between boson candidates 
        #va = va[:,:self.nAv].view(n,self.nAv,1) # |va|
        #va = torch.cat( (va, va, va), 2) # |va|va|va|
        
        #mask = o[:,4,:]!=-1
        #o = o[:,0:4,:]

        if self.training: #random permutation
            # c = torch.randperm(3)
            # p = p.view(n,-1,3,4)[:,:,c,:].view(n,-1,12)
            # qa = qa[:,:,c]
            # c = torch.cat( (0+torch.randperm(2), 2+torch.randperm(2), 4+torch.randperm(2)), 0)
            # p = p.view(n,-1,6,2)[:,:,c,:].view(n,-1,12)
            # da = da[:,:,c]
            nPermutationChunks=5
            cn=n//nPermutationChunks

            if cn != 0:
                r =n%cn
                for i in range(nPermutationChunks):
                    l = i*cn
                    u = (i+1)*cn + (r if i+1==nPermutationChunks else 0)

                    c = torch.randperm(3)
                    p [l:u] = p [l:u].view(u-l,-1,3,4)[:,:,c,:].view(u-l,-1,12)
                    qa[l:u] = qa[l:u,:,c]

                    c = torch.cat( (0+torch.randperm(2), 2+torch.randperm(2), 4+torch.randperm(2)), 0)
                    p [l:u] = p [l:u].view(u-l,-1,6,2)[:,:,c,:].view(u-l,-1,12)
                    da[l:u] = da[l:u,:,c]


        ps, qs = [], []
        randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
        for i in range(self.nR):
            ps.append(self.rotate(p, self.R[i]+randomR[i]))
            #os.append(self.rotate(o, self.R[i]+randomR[i]))
            qs.append(self.invPart(ps[-1], [], da, qa))
            if self.doFlip:
                #flip phi of original
                ps.append(self.flipPhi(ps[-1]))
                #os.append(self.flipPhi(os[-1]))
                qs.append(self.invPart(ps[-1], [], da, qa))

                #flip phi and eta of original
                ps.append(self.flipEta(ps[-1]))
                #os.append(self.flipEta(os[-1]))
                qs.append(self.invPart(ps[-1], [], da, qa))

                #flip eta of original
                ps.append(self.flipEta(ps[-3]))
                #os.append(self.flipEta(os[-3]))
                qs.append(self.invPart(ps[-1], [], da, qa))

        q = sum(qs)/self.nRF

        v = q

        #v = torch.cat( (q, va), 1) # manually add features to event view feature space

        v0 = v.clone()
        #v = NonLU(v)
        v = self.viewConv1(v)
        v = v+v0
        v = NonLU(v)
        v = self.viewConv2(v)
        v = v+v0
        v = NonLU(v)

        v = self.viewSelector(v)
        v = v.view(n, -1)
        return v


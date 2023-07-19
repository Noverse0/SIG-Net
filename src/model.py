import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn.pytorch import GATConv, SAGEConv, RelGraphConv, GINConv
from torch.autograd import Variable 

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
    
class Multi_RGCN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, bi, dropout, batch_size, device):
        super(Multi_RGCN, self).__init__()
        self.bi = bi
        self.h_feats = h_feats
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
    
        self.convlayers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.convlayers.append(
                    RelGraphConv(in_feats, h_feats, num_rels= 11, self_loop=True)
                )
            else:
                self.convlayers.append(
                    RelGraphConv(h_feats, h_feats, num_rels= 11, self_loop=True)
                )
        
        self.lstm = nn.LSTM(input_size=h_feats * num_layers * 2, hidden_size=h_feats * num_layers * 2, num_layers=2, bidirectional=bi, dropout=dropout, batch_first=True)
        
        if bi:
            self.W1 = nn.Linear(h_feats * num_layers * 2 * 2, h_feats * num_layers * 2, bias = True)
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
            
        else:
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
            
    # 학습 초기화를 위한 함수
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.num_layers, self.h_feats, self.batch_size),
                torch.zeros(self.num_layers, self.h_feats, self.batch_size))
            
    def forward(self, g1, g2, g3, training):
        self.training = training
        h1 = g1.ndata['feature']
        h2 = g2.ndata['feature']
        h3 = g3.ndata['feature']
        etype1 = g1.edata['etype']
        etype2 = g2.edata['etype']
        etype3 = g3.edata['etype']
        
        for i, layer in enumerate(self.convlayers):
            h1 = F.relu(layer(g1, h1, etype1))
            h2 = F.relu(layer(g2, h2, etype2))
            h3 = F.relu(layer(g3, h3, etype3))
            
            if i == 0:
                output1 = h1
                output2 = h2
                output3 = h3

            else:
                output1 = torch.cat([output1, h1], 1)         
                output2 = torch.cat([output2, h2], 1)         
                output3 = torch.cat([output3, h3], 1)   

        enroll1 = g1.ndata['target'] == 1
        enroll2 = g2.ndata['target'] == 1
        enroll3 = g3.ndata['target'] == 1
        course1 = g1.ndata['target'] == 2
        course2 = g2.ndata['target'] == 2
        course3 = g3.ndata['target'] == 2
        
        em1 = torch.cat([output1[enroll1], output1[course1]], 1)
        em2 = torch.cat([output2[enroll2], output2[course2]], 1)
        em3 = torch.cat([output3[enroll3], output3[course3]], 1)
        em = torch.stack([em1,em2,em3], dim=1)
        
        output, (hn, cn) = self.lstm(em) #lstm with input, hidden, and internal state
        
        outputs = output[:, -1] 
        if self.bi:
            x = F.relu(self.W1(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.W2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
        else:
            x = F.relu(self.W2(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
        return x.reshape(-1)

class Multi_GraphSage(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, bi, dropout, batch_size, device):
        super(Multi_GraphSage, self).__init__()
        self.bi = bi
        self.h_feats = h_feats
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
    
        self.convlayers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.convlayers.append(
                    SAGEConv(in_feats, h_feats, "mean")
                )
            else:
                self.convlayers.append(
                    SAGEConv(h_feats, h_feats, "mean")
                )
        
        self.lstm = nn.LSTM(input_size=h_feats * num_layers * 2 , hidden_size=h_feats * num_layers * 2, num_layers=2, bidirectional=bi, dropout=dropout, batch_first=True)
        
        if bi:
            self.W1 = nn.Linear(h_feats * num_layers * 2 * 2, h_feats * num_layers * 2, bias = True)
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
        else:
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
            
    # 학습 초기화를 위한 함수
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.num_layers, self.h_feats, self.batch_size),
                torch.zeros(self.num_layers, self.h_feats, self.batch_size))
            
    def forward(self, g1, g2, g3, training):
        self.training = training
        h1 = g1.ndata['feature']
        h2 = g2.ndata['feature']
        h3 = g3.ndata['feature']
        
        for i, layer in enumerate(self.convlayers):
            h1 = F.relu(layer(g1, h1))
            h2 = F.relu(layer(g2, h2))
            h3 = F.relu(layer(g3, h3))
            
            if i == 0:
                output1 = h1
                output2 = h2
                output3 = h3

            else:
                output1 = torch.cat([output1, h1], 1)         
                output2 = torch.cat([output2, h2], 1)         
                output3 = torch.cat([output3, h3], 1)   

        enroll1 = g1.ndata['target'] == 1
        enroll2 = g2.ndata['target'] == 1
        enroll3 = g3.ndata['target'] == 1
        course1 = g1.ndata['target'] == 2
        course2 = g2.ndata['target'] == 2
        course3 = g3.ndata['target'] == 2
        
        em1 = torch.cat([output1[enroll1], output1[course1]], 1)
        em2 = torch.cat([output2[enroll2], output2[course2]], 1)
        em3 = torch.cat([output3[enroll3], output3[course3]], 1)
        em = torch.stack([em1,em2,em3], dim=1)
        
        h_0 = Variable(torch.zeros(self.num_layers*2, em.size(0), self.h_feats*self.num_layers*2)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers*2, em.size(0), self.h_feats*self.num_layers*2)).to(self.device) #internal state   
        
        output, (hn, cn) = self.lstm(em) #lstm with input, hidden, and internal state
        
        outputs = output[:, -1]  
        if self.bi:
            x = F.relu(self.W1(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.W2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
        else:
            x = F.relu(self.W2(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
            
        return x.reshape(-1)
    
class Multi_GAT(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, bi, dropout, batch_size, device):
        super(Multi_GAT, self).__init__()
        self.bi = bi
        self.h_feats = h_feats
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
    
        self.convlayers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.convlayers.append(
                    GATConv(in_feats, h_feats, num_heads=3, allow_zero_in_degree=True)
                )
            else:
                self.convlayers.append(
                    GATConv(h_feats, h_feats, num_heads=3, allow_zero_in_degree=True)
                )
        self.lstm = nn.LSTM(input_size=h_feats * num_layers * 2 , hidden_size=h_feats * num_layers * 2, num_layers=2, bidirectional=bi, dropout=dropout, batch_first=True)

        if bi:
            self.W1 = nn.Linear(h_feats * num_layers * 2 * 2, h_feats * num_layers * 2, bias = True)
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
            
        else:
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
            
    # 학습 초기화를 위한 함수
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.num_layers, self.h_feats, self.batch_size*3),
                torch.zeros(self.num_layers, self.h_feats, self.batch_size*3))
            
    def forward(self, g1, g2, g3, training):
        self.training = training
        h1 = g1.ndata['feature']
        h2 = g2.ndata['feature']
        h3 = g3.ndata['feature']
        
        for i, layer in enumerate(self.convlayers):
            h1 = F.relu(torch.sum(layer(g1, h1), dim=1))
            h2 = F.relu(torch.sum(layer(g2, h2), dim=1))
            h3 = F.relu(torch.sum(layer(g3, h3), dim=1))
            
            if i == 0:
                output1 = h1.reshape(g1.num_nodes(), -1)
                output2 = h2.reshape(g2.num_nodes(), -1)
                output3 = h3.reshape(g3.num_nodes(), -1)
                
            else:
                output1 = torch.cat([output1, h1.reshape(g1.num_nodes(), -1)], 1)         
                output2 = torch.cat([output2, h2.reshape(g2.num_nodes(), -1)], 1)         
                output3 = torch.cat([output3, h3.reshape(g3.num_nodes(), -1)], 1)  
                
        enroll1 = g1.ndata['target'] == 1
        enroll2 = g2.ndata['target'] == 1
        enroll3 = g3.ndata['target'] == 1
        course1 = g1.ndata['target'] == 2
        course2 = g2.ndata['target'] == 2
        course3 = g3.ndata['target'] == 2
        
        em1 = torch.cat([output1[enroll1], output1[course1]], 1)
        em2 = torch.cat([output2[enroll2], output2[course2]], 1)
        em3 = torch.cat([output3[enroll3], output3[course3]], 1)
        em = torch.stack([em1,em2,em3], dim=1)
        
        h_0 = Variable(torch.zeros(self.num_layers*2, em.size(0), self.h_feats*self.num_layers*2)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers*2, em.size(0), self.h_feats*self.num_layers*2)).to(self.device) #internal state   
        
        output, (hn, cn) = self.lstm(em) #lstm with input, hidden, and internal state
        
        outputs = output[:, -1] 
        if self.bi:
            x = F.relu(self.W1(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.W2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
        else:
            x = F.relu(self.W2(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
            
        return x.reshape(-1)
    
class Multi_GIN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, bi, dropout, batch_size, device):
        super(Multi_GIN, self).__init__()
        self.bi = bi
        self.h_feats = h_feats
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
        self.convlayers = nn.ModuleList()
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_feats, h_feats, h_feats)
            else:
                mlp = MLP(h_feats, h_feats, h_feats)
            self.convlayers.append(
                GINConv(mlp, learn_eps=False)
            ) 
        self.lstm = nn.LSTM(input_size=h_feats * num_layers * 2 , hidden_size=h_feats * num_layers * 2, num_layers=2, bidirectional=bi, dropout=dropout, batch_first=True)
        
        if bi:
            self.W1 = nn.Linear(h_feats * num_layers * 2 * 2, h_feats * num_layers * 2, bias = True)
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
        else:
            self.W2 = nn.Linear(h_feats * num_layers * 2, h_feats * num_layers, bias = True)
            self.W3 = nn.Linear(h_feats * num_layers, 1, bias = True)
            
    def forward(self, g1, g2, g3, training):
        self.training = training
        h1 = g1.ndata['feature']
        h2 = g2.ndata['feature']
        h3 = g3.ndata['feature']
        
        for i, layer in enumerate(self.convlayers):
            h1 = layer(g1, h1)
            h2 = layer(g2, h2)
            h3 = layer(g3, h3)
            
            h1 = F.relu(h1)
            h2 = F.relu(h2)
            h3 = F.relu(h3)
            if i == 0:
                output1 = h1
                output2 = h2
                output3 = h3
            else:
                output1 = torch.cat([output1, h1], 1)         
                output2 = torch.cat([output2, h2], 1)         
                output3 = torch.cat([output3, h3], 1)   

        enroll1 = g1.ndata['target'] == 1
        enroll2 = g2.ndata['target'] == 1
        enroll3 = g3.ndata['target'] == 1
        course1 = g1.ndata['target'] == 2
        course2 = g2.ndata['target'] == 2
        course3 = g3.ndata['target'] == 2

        em1 = torch.cat([output1[enroll1], output1[course1]], 1)
        em2 = torch.cat([output2[enroll2], output2[course2]], 1)
        em3 = torch.cat([output3[enroll3], output3[course3]], 1)
        em = torch.stack([em1,em2,em3], dim=1)
        
        h_0 = Variable(torch.zeros(self.num_layers*2, em.size(0), self.h_feats*self.num_layers*2)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers*2, em.size(0), self.h_feats*self.num_layers*2)).to(self.device) #internal state   
        
        output, (hn, cn) = self.lstm(em) #lstm with input, hidden, and internal state
        
        outputs = output[:, -1]  # 최종 예측 Hidden Layer
        if self.bi:
            x = F.relu(self.W1(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.W2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
        else:
            x = F.relu(self.W2(outputs))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W3(x)
            x = torch.sigmoid(x)
            
        return x.reshape(-1)
    
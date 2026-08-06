import torch
import math
from torch import nn
from torch.nn import functional as F

# class Transformer(nn.Module):
#     def __init__(self,embedding_dim,head_num,FFN_dim,vocab_size,position_size=512,sentence_nums=2):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
#         self.position_embedding = nn.Embedding(position_size,embedding_dim)
#         self.sentence_embedding = nn.Embedding(sentence_nums,embedding_dim)
#         self.q = nn.Linear(embedding_dim,embedding_dim)
#         self.k = nn.Linear(embedding_dim,embedding_dim)
#         self.v = nn.Linear(embedding_dim,embedding_dim)
#         self.MHAlinear = nn.Linear(embedding_dim,embedding_dim)
#         self.FFN1 = nn.Linear(embedding_dim,FFN_dim)
#         self.FFN2 = nn.Linear(FFN_dim,embedding_dim)
#         self.head_num = head_num
#         self.embedding_dim = embedding_dim
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.activation = nn.GELU()
#     def forward(self,X,segment_ids):
#         position_input = torch.arange(X.shape[1]).reshape(1,-1)
#         token_emb = self.token_embedding(X)
#         position_emb = self.position_embedding(position_input)
#         sentence_emb = self.sentence_embedding(segment_ids)
#         emb_input = token_emb + position_emb + sentence_emb
#         Q = self.q(emb_input)
#         K = self.k(emb_input)
#         V = self.v(emb_input)
#         head_dim = self.embedding_dim//self.head_num


#         headed_Qs = [Q[:,:,i*head_dim:(i+1)*head_dim] for i in range(self.head_num)]
#         headed_Ks = [K[:,:,i*head_dim:(i+1)*head_dim] for i in range(self.head_num)]
#         headed_Vs = [V[:,:,i*head_dim:(i+1)*head_dim] for i in range(self.head_num)]
#         headed_output = []
#         for i in range(len(headed_Qs)):
#             L = torch.matmul(headed_Qs[i],headed_Ks[i].transpose(-2,-1))
#             normed_L = F.softmax(L/math.sqrt(head_dim),dim=-1)
#             headed_V = torch.matmul(normed_L,headed_Vs[i])
#             headed_output.append(headed_V)
#         MHA = torch.cat(headed_output,dim=-1)
#         MHA = self.MHAlinear(MHA)
#         residual1 = self.norm1(emb_input+MHA)
#         FFN_hid = self.activation(self.FFN1(residual1))
#         FFN_output = self.FFN2(FFN_hid)
#         return self.norm2(residual1+FFN_output) 

# x = torch.randint(1,100,(2,10))
# segment_id = torch.zeros((2,10),dtype=torch.long)
# net = Transformer(768,12,3072,512)
# print(net(x,segment_id).shape)



#-------------------------------------------------------------------------
class MutiHeadSelfAttention(nn.Module):
    def __init__(self,embedding_dim,d_k):
        super().__init__()
        assert embedding_dim%d_k == 0
        self.head_num = embedding_dim//d_k
        self.d_k = d_k
        self.qkv = nn.Linear(embedding_dim,embedding_dim*3)
        self.linear = nn.Linear(embedding_dim,embedding_dim)
    def forward(self,X,mask=None):
        B,T,D = X.shape
        q,k,v = self.qkv(X).chunk(3,dim=-1)
        q_head = q.reshape((B,T,self.head_num,self.d_k)).transpose(1,2)
        k_head = k.reshape((B,T,self.head_num,self.d_k)).transpose(1,2)
        v_head = v.reshape((B,T,self.head_num,self.d_k)).transpose(1,2)   #(B,num,T,dim)
        L_socre = torch.matmul(q_head,k_head.transpose(-2,-1))                 #(B,num,T,T)
        mul_v = torch.matmul(F.softmax(L_socre/math.sqrt(self.d_k),dim=-1),v_head)
        return self.linear(mul_v.transpose(1,2).reshape(B,T,-1))

class Encoder(nn.Module):
    def __init__(self,embedding_dim,ffn_dim,d_k):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_dim = ffn_dim
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)      
        self.attn = MutiHeadSelfAttention(embedding_dim,d_k)
        self.fnn = nn.Sequential(nn.Linear(embedding_dim,ffn_dim),nn.GELU(),nn.Linear(ffn_dim,embedding_dim))
    def forward(self,X,mask=None):
        residual = self.norm1(self.attn(X)+X)
        output = self.norm2(residual+self.fnn(residual))
        return output

class TransformerEncoder(nn.Module):
    def __init__(self,layer_num,embedding_dim,ffn_dim,d_k):
        super().__init__()
        self.layers = nn.ModuleList([Encoder(embedding_dim,ffn_dim,d_k) for _ in range(layer_num)])
    def forward(self,X,mask=None):
        for layer in self.layers:
            X = layer(X)
        return X

X = torch.randn(2,10,512)
net = TransformerEncoder(6,512,1024,64)
print(net(X).shape)    
#问题：定义mutiheadselfattention时，使用“头”的数量作为参数传入，似乎比使用“头”的维度作为传参更有意义
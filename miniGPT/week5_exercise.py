#训练基于transformer的单向语言模型，并完成文本生成。
#文本来源：corpus.txt

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

import math
import glob
import argparse
import random

#----------------------1.数据预处理----------------------------
def load_corpus(pattern='*.txt'):
    text_file = glob.glob(pattern)
    text = []
    for path in text_file:
        with open(path,encoding='utf-8',errors='ignore') as f:
            text.append(f.read())
    return ''.join(text)

def bulid_vocab(text):
    purified = set(text)
    char2idx = {c:i for i,c in enumerate(purified)}
    idx2char = {i:c for i,c in enumerate(purified)}
    return char2idx,idx2char

class CharDataset(Dataset):
    def __init__(self,text,char2idx,seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tokens = torch.tensor([char2idx[char] for char in text],dtype=torch.long)
    def __len__(self):
        return max(0,len(self.tokens)-self.seq_len)
    def __getitem__(self, index):
        x = self.tokens[index:index+self.seq_len]
        y = self.tokens[index+1:index+self.seq_len+1]
        return x,y

#--------------------2.transformer模型搭建-----------------------
class TransEmbedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim,position_dim):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size,embedding_dim)
        self.position_embed = nn.Embedding(position_dim,embedding_dim)
    def forward(self,x):                                                            #x:(B,T)
        token_embed = self.token_embed(x)                                           #(B,T,E)
        temp = torch.arange(x.shape[1],dtype=torch.long).reshape(1,-1).to('cuda' if torch.cuda.is_available() else 'cpu')            #(T)
        return token_embed + self.position_embed(temp)

class MutiHeadSelfAttention(nn.Module):
    def __init__(self,embedding_dim,nhead):
        super().__init__()
        assert embedding_dim%nhead == 0,f"请保证embedding_dim可被nhead整除"
        self.nhead = nhead
        self.d_k = embedding_dim//nhead
        self.qkv = nn.Linear(embedding_dim,embedding_dim*3)
        self.linear = nn.Linear(embedding_dim,embedding_dim)
    def forward(self,x):
        B,T,E = x.shape
        qkv = self.qkv(x)
        q,k,v = torch.chunk(qkv,3,dim=-1)
        q = q.reshape(B,T,self.nhead,self.d_k).transpose(1,2)
        k = k.reshape(B,T,self.nhead,self.d_k).transpose(1,2)
        v = v.reshape(B,T,self.nhead,self.d_k).transpose(1,2)
        L = q@k.transpose(-2,-1)

        mask = torch.tril(torch.ones(L.shape[-1],L.shape[-1],device=x.device))
        L = L.masked_fill(mask==0,float('-inf'))



        after_qkv = torch.matmul(F.softmax(L/math.sqrt(self.d_k),dim=-1),v).reshape(B,T,E)
        final_linear = self.linear(after_qkv)
        return final_linear

class Decoder(nn.Module):
    def __init__(self,embedding_dim,ffn_dim,nhead,dropout):
        super().__init__()
        self.attn = MutiHeadSelfAttention(embedding_dim,nhead)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim,ffn_dim),nn.GELU(),nn.Linear(ffn_dim,embedding_dim))
    def forward(self,x):
        attn = self.attn(self.norm1(x))
        residual = x + self.dropout(attn)
        return residual + self.dropout(self.ffn(self.norm2(residual)))

class TransDecoderLayers(nn.Module):
    def __init__(self,embedding_dim,position_dim,ffn_dim,nhead,vocab_size,num_layers,dropout):
        super().__init__()
        self.embedding = TransEmbedding(vocab_size,embedding_dim,position_dim)
        self.decoder = nn.ModuleList([Decoder(embedding_dim,ffn_dim,nhead,dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(embedding_dim,vocab_size)
    def forward(self,x):
        x = self.embedding(x)
        for layer in self.decoder:
            x = layer(x)
        return self.linear(x)


#-----------------3.定义训练范式------------------
def run_epoch(loader,model,criterion,optimizer,train=True):
    model.train(train)
    total_loss,total_tokens = 0,0
    for X,y in loader:
        X,y = X.to('cuda' if torch.cuda.is_available() else 'cpu'),y.to('cuda' if torch.cuda.is_available() else 'cpu')
        y_hat = model(X)                                     #y_hat(B,T,V)  y(B,T)
        l = criterion(y_hat.reshape(-1,y_hat.shape[-1]),y.reshape(-1))
        if train:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        total_loss += l.item()*y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss/total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss,ppl


#---------------------4.主程序----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,  default=10)
    parser.add_argument('--lr',           type=int,  default=1e-3)
    parser.add_argument('--seq_len',      type=int,  default=16)
    parser.add_argument('--ratio',        type=float,default=0.05)
    parser.add_argument('--batch_size',   type=int,  default=64)
    parser.add_argument('--embedding_dim',type=int,  default=256)
    parser.add_argument('--position_dim', type=int,  default=256)
    parser.add_argument('--ffn_dim',      type=int,  default=512)
    parser.add_argument('--nhead',        type=int,  default=8)
    parser.add_argument('--num_layers',   type=int,  default=2)
    parser.add_argument('--dropout',      type=float,default=0.1)
    parser.add_argument('--weight_decay', type=float,default=1e-5)
    parser.add_argument('--file_name',               default='miniGPTbaseline.bin')
    args = parser.parse_args()

    text = load_corpus()
    print(f"文本字符总长度：{len(text)}")

    char2idx,idx2char = bulid_vocab(text)
    print(f"token数量：{len(char2idx)}")

    text_list = text.splitlines()
    random.shuffle(text_list)
    text_fresh = '\n'.join(text_list)

    split = int(len(text_fresh)*(1-args.ratio))

    train_data = CharDataset(text_fresh[:split],char2idx,args.seq_len)
    valid_data = CharDataset(text_fresh[split:],char2idx,args.seq_len)

    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,drop_last=True)
    valid_loader = DataLoader(valid_data,batch_size=args.batch_size,shuffle=True,drop_last=True)

    model = TransDecoderLayers(args.embedding_dim,args.position_dim,args.ffn_dim,args.nhead,len(char2idx),args.num_layers,args.dropout).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    print(f"参数量：{sum([p.numel() for p in model.parameters()])}")

    best_ppl = float('inf')

    print(f"\n{'epoch':>12}{'train_loss':>12}{'train_ppl':>12}{'valid_loss':>12}{'valid_ppl':>12}")
    print("--"*52)
    for epoch in range(1,args.epochs+1):
        train_loss,train_ppl = run_epoch(train_loader,model,criterion,optimizer,train=True)
        with torch.no_grad():
            valid_loss,valid_ppl = run_epoch(valid_loader,model,criterion,optimizer,train=False)

        if valid_ppl < best_ppl:
            marker = '*'
            best_ppl = valid_ppl
            torch.save({'model_dict':model.state_dict(),
                        'char2idx':char2idx,
                        'idx2char':idx2char,
                        'hyperparams':vars(args)},args.file_name)
        else:
            marker = ''
        print(f"{epoch:>12}{train_loss:>12.4f}{train_ppl:>12.2f}{valid_loss:>12.4f}{valid_ppl:>12.2f}{marker:^6}")

if __name__ == '__main__':
    main()
#任务描述---------------------------
#设计一个多分类任务，“你”字出现在文本的第几个位置，就是第几类，如果没有出现，则为“0”类
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import random

#超参数——————————————————————————————————————————————————
SAMPLE_NUM = 6400
MAX_LEN = 7
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCH_NUM = 20
DROP_OUT = 0.3


#文本搭建————————————————————————————————————————————————————

temp_positives = [
                "{}好呀",
                "{}牛啊",
                "想见{}",
                "风遇{}而来",
                "星光落{}肩头",
                "等一场{}赴约",
                "寻{}",
                "念{}千万遍",
                "温柔予{}",
                "人间偏爱{}",
                "携月色寻{}",
                "盼{}归",
                "万般美好为{}",
                "望向{}眼眸",
                "寄心意于{}",
                "懂{}",
                "山河皆因{}",
                "晚风轻唤{}",
                "赠欢喜给{}",
                "牵挂{}",
                "岁岁陪着{}"
                ]
# temp_positives = [

#     "{}真厉害",
#     "我很喜欢{}",
#     "{}今天真棒",
#     "感谢{}帮助",
#     "{}说得对",
#     "看到{}真开心",
#     "{}让我感动",
#     "一直支持{}",
#     "希望{}顺利",
#     "{}做得很好",
#     "我相信{}",
#     "{}值得信赖",
#     "遇见{}很幸运",
#     "{}给我力量",
#     "期待{}的表现"
# ]


temp_negatives = [
    "晚风",
    "星辰",
    "山野",
    "朝暮",
    "观云起",
    "赴山海",
    "揽清风",
    "赏秋月",
    "林间听泉",
    "落日归山",
    "浅酌清茶",
    "静候花开",
    "山野藏清风",
    "云间藏月色",
    "晨起闻花香",
    "晚坐观星河",
    "携风漫步林间",
    "静坐静待花开",
    "远山藏尽温柔",
    "落日漫过街巷"
]

# temp_negatives = [
#     "今天真开心",
#     "天气很不错",
#     "大家都来了",
#     "明天再联系",
#     "这个很有趣",
#     "事情解决了",
#     "感觉还不错",
#     "工作完成了",
#     "刚刚下过雨",
#     "周末去爬山",
#     "电影很好看",
#     "已经到家了",
#     "学习有进步",
#     "晚上早点睡",
#     "生活很充实"
# ]

def make_positive():
    key_word = '你'
    tp = random.choice(temp_positives)
    return tp.format(key_word)

def make_negative():
    return random.choice(temp_negatives)



#创建数据和词表————————————————————————————————————————————————————

#创建数据集，文本的结果用七维one-hot向量表示，文本的第几个位置有“你”，则属于第几类。若文本没有“你”字，则属于0类
def build_dataset(sample_num,maxlen=MAX_LEN):
    dataset = []
    for i in range(sample_num//2):
        y_row = [0] * maxlen
        negative_y = [0] * maxlen
        negative_y[0] = 1
        a = make_positive()
        y_row[a.find('你')+1] = 1
        dataset.append((a,y_row))
        dataset.append((make_negative(),negative_y))
    random.shuffle(dataset)
    return dataset

#将所有文本进行进行编码
def vocab(dataset):
    vocab = {'<pad>':0,'<unk>':1}
    for e,_ in dataset:
        for char in e:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

#将句子解码
def encode(sent,vocab):
    ids = [vocab.get(char,1) for char in sent]
    ids = ids[:MAX_LEN]
    ids += [0]*(MAX_LEN-len(ids))
    return ids

#将句子解码为可输入模型的格式：
class TextLoader(Dataset):
    def __init__(self,dataset,vocab):
        super().__init__()
        self.x = [encode(e,vocab) for e,_ in dataset]
        self.y = [e for _,e in dataset]
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return torch.LongTensor(self.x[index]),torch.FloatTensor(self.y[index])
    
#开始设置模型，使用模型：RNN————————————————————————————————————————————
class RNNModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,maxlen=MAX_LEN,dropout=DROP_OUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0) #(vocab_size,embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_dim,batch_first=True)  #(batch,maxlen,hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim*maxlen)       #
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim*maxlen,maxlen) 
        self.activation = nn.Softmax(dim=1)
    def forward(self,x):
        embeded = self.embedding(x)
        rnned,_ = self.rnn(embeded)
        # pooled = rnned.max(dim=1)[0]
        flatten = rnned.reshape(rnned.size(0),-1)
        bned = self.bn(flatten)
        dropped = self.dropout(bned)
        lineared = self.linear(dropped)
        return lineared
    
#设置主程序——————————————————————————————————————————————————————————
def train():
    #构建数据集和词表，80%训练，20%评估：
    dataset = build_dataset(SAMPLE_NUM)
    total_vocab = vocab(dataset)
    train_x = dataset[:int(SAMPLE_NUM*TRAIN_RATIO)]
    eval_x = dataset[int(SAMPLE_NUM*TRAIN_RATIO):]

    train_dataset = TextLoader(train_x,total_vocab)
    eval_dataset = TextLoader(eval_x,total_vocab)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    eval_loader = DataLoader(eval_dataset,batch_size=BATCH_SIZE)

    #模型设置：
    model = RNNModel(len(total_vocab))
    loss_compute = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    print(f"开始训练模型：")
    for i in range(1,EPOCH_NUM+1):
        total_loss = 0
        model.train()
        for x,y in train_loader:
            y_pred = model(x)
            loss = loss_compute(y_pred,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        total_loss = total_loss/len(train_loader)
        acc = evaluate(model,eval_loader)
        print(f"第{i}轮训练结束：\ntotal_loss：{total_loss:.4f}\n准确率：{acc:.4f}\n")

    #将模型参数保存
    checkpoint = {'model.state_dict()':model.state_dict(),'vocab_size':len(total_vocab),'vocab':total_vocab}
    torch.save(checkpoint,'rnnmodel.bin')
    return

#设置evaluate函数，来计算验证集上的准确率:
def evaluate(model,loader):
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for x,y in loader:
            y_pred = model(x)
            activation = nn.Softmax(dim=1)
            y_pred = activation(y_pred)   #y_pred shape(batch,hidden_dim)
            y_pred = y_pred.argmax(dim=1)
            y = y.argmax(dim=1)
            correct += (y_pred == y).sum().item()
            total += len(y)
    return correct/total


#将测试部分分出来，单独构建test函数：
def test(model_path,testset):
    checkpoint = torch.load(model_path)
    model = RNNModel(checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model.state_dict()'])
    model.eval()
    total_vocab = checkpoint['vocab']
    activation = nn.Softmax(dim=1)
    with torch.no_grad():
      for sent in testset:
        ids = encode(sent,total_vocab)
        x = torch.LongTensor([ids])
        y_pred1 = model(x)
        y_pred = activation(y_pred1)
        print(f"预测“{sent}”属于第{int(y_pred.argmax())}类，概率值为{float(y_pred.max()):.4f}")



# train()

if __name__ == '__main__':
    test_texts = [
    "你好",
    "天气好",
    "我想你",
    "明天见",
    "你真棒",
    "吃饭了",
    "喜欢你",
    "今天很忙",
    "我等你",
    "早点睡",
    "你来了",
    "工作顺利",
    "想见你",
    "下雨了",
    "谢谢你",
    "学习中",
    "你在哪",
    "周末愉快",
    "陪你走",
    "很开心"
]
    test('rnnmodel.bin',test_texts)
 
    





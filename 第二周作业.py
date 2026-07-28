import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#作业：尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。


#目标搭建：
def build_sample():
    #创建一个五维随机向量 
    x = np.random.random(5)
    y = np.zeros(5)
    y[int(x.argmax())] = 1
    #返回特征与其标签
    return x,y

#创建训练集：
def build_dataset(data_num):
    X =[]
    Y= []
    for i in range(data_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    #返回特征张量，与标签张量
    return torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.float32)

#搭建模型：
class TorchModel(nn.Module):
    #outputsize默认值为5
    def __init__(self,input_size,output_size=5):
        #继承父类：
        super().__init__()
        self.linear = nn.Linear(input_size,output_size) #线性层
        #nn.CrossEntropyLoss()内置了softmax激活函数，不需要手动设置激活函数了。
        self.loss = nn.CrossEntropyLoss()  #使用交叉熵损失函数
    def forward(self,x,y=None):        #模型正向传播过程
        logits = self.linear(x)          #第一步线性变化
        #不需要手动激活了
        #提供标签则返回损失函数，不提供标签返回预测值
        if y is None:
            #softmax方法的分母，是多元素因子的累加，要考虑dim问题
            y_pred = torch.softmax(logits,dim=1)
            return y_pred
        else:
            return self.loss(logits,y)



#学习过程主干：
#并将每轮训练的acc与loss值可视化
def main():
    #参数预备：
    input_size = 5
    epoch_num = 50
    train_num = 5000
    batch_size = 20
    learning_rate = 0.001
    #模型选择:
    model = TorchModel(input_size)
    #优化器选择：
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #训练集的构建：
    train_x,train_y = build_dataset(train_num)
    #存储要可视化的acc和loss值
    log=[]
    #设置20轮的训练总数
    for i in range(epoch_num):
        watch_loss = []
        for batch_index in range(train_num//batch_size):
            #模型调整为训练模式
            model.train()
            #一个batch的所有数据集
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model(x,y)   #计算损失函数
            loss.backward()     #计算梯度
            optim.step()        #梯度更新
            optim.zero_grad()   #梯度清零
            watch_loss.append(loss.item())
        #acc和loss的处理与存储
        watch_loss = sum(watch_loss)/len(watch_loss)
        acc = evaluate(model)
        log.append([watch_loss,acc])
        print(f"第{i+1}次训练--loss值：{watch_loss}，准确率{acc}")
    #保存model的参数：
    torch.save(model.state_dict(),'homeworkdemo.bin')
    plt.plot(range(len(log)),[l[0] for l in log],label='loss')
    plt.plot(range(len(log)),[l[1] for l in log],label='acc')
    plt.legend()
    plt.show()
    return


#每轮训练的评估步骤，包裹在evaluate函数中，该函数输出为准确率
def evaluate(model):
    #模型调整为测试/预测模式
    model.eval()
    #选择100条数据作为验证
    sample_num = 100
    x,y = build_dataset(sample_num)
    correct,wrong = 0,0
    #测试/预测模式下，不需要计算梯度，直接出预测值
    with torch.no_grad():
        logits = model(x)
        for y_p,y_t in zip(logits,y):
            if y_p.argmax() == y_t.argmax(): #numpy的True和false也可以用作if/while判断
                correct += 1
            else:
                wrong += 1
    acc = correct/(correct+wrong)
    return acc


#创建预测的方法，评估模型的准确程度（视input_vec为列表）
def predict(model_path,input_vec):
    #模型设置，以及读取加载已有模型的参数
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #转化为张量
    x = torch.tensor(input_vec,dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(x)
    for vec,res in zip(input_vec,y_pred):
        label_true = vec.index(max(vec))
        label_pred = int(res.argmax())
        print(f"对于向量{vec}，它属于{label_true}类，预测结果为{label_pred}类，概率值{res.max()}")


# main()

if __name__ == '__main__':
    test_vec = [[0.27, 0.83, 0.11, 0.49, 0.62],
                [0.35, 0.71, 0.09, 0.58, 0.22],
                [0.94, 0.17, 0.76, 0.33, 0.51],
                [0.06, 0.44, 0.88, 0.29, 0.69],
                [0.55, 0.13, 0.39, 0.78, 0.02],
                [0.81, 0.25, 0.66, 0.19, 0.47],
                [0.31, 0.59, 0.04, 0.92, 0.73],
                [0.15, 0.68, 0.41, 0.08, 0.85],
                [0.79, 0.23, 0.54, 0.12, 0.97],
                [0.46, 0.01, 0.89, 0.37, 0.64]]
    predict('homeworkdemo.bin',test_vec)
    
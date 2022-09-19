from random import sample
from sqlite3 import PARSE_COLNAMES
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import open3d as o3d
import matplotlib.pyplot as plt
from gettwogroups import gettwogroups
import os

class PointCloudDataset(Dataset):
    def __init__(self):
        wholepointswithoutpot_np, _, incompletepoints = gettwogroups()

        labels = np.ones(wholepointswithoutpot_np.shape[0])
        labels[incompletepoints] = 0

        self.cloud_points = wholepointswithoutpot_np
        self.labels = labels
        self.total_points = wholepointswithoutpot_np.shape[0]
    #在真实数据中，有多少个样本，len就为多少，这个地方比较特殊
    def __len__(self):
        return 2

    def __getitem__(self, index):
        np.random.seed(0)
        sampler = np.random.choice(self.total_points, size = int(0.01*self.total_points), replace = False)
        return self.cloud_points[sampler, :], self.labels[sampler]

#Model
class Tnet(nn.Module):
   def __init__(self, k=3):
    #当需要继承父类的构造函数中的内容，且子类需要在父类的基础上进行补充时，使用super().__init__()方法
    #--super()属性的查找顺序是从当前位置开始找，根据mro列表，当前没有就往上找
    #super()函数用于调用父类的一个方法
    #super是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没有问题，如果是用多继承，会涉及到查找顺序(MRO)，重复调用(钻石继承)等种种问题
    #MRO就是类的方法解析顺序表，其实也就是继承父类方法时的顺序表
      super().__init__()
      self.k=k
      #cov1:一维卷积 参数如输入信号的通道， 卷积产生的通道，卷积核的尺寸
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      #nn.linear:参数input_sample的尺寸和output_sample的尺寸
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)
      #parameters:num_features是number of features or channels C of the input, 其他都有default
      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       
    #__init__()是网络结构，而forward是前向传播
    # eg:
    # class Module(nn.Module):
    #     定义网络结构
    #     def __init__(self):
    #         super(Module, self).__init__()
    #         写内容
    #     前向传播
    #     def forward(self, x):
    #         写内容
    #         return x
    # 输入数据
    # data = ...
    # 实例化网络
    # module = Module()
    # 前向传播
    # module(data)
    # 而不是使用Module.forward(data)

   def forward(self, input):
      # input.shape == (bs,n,3)
      #input.size就是input.shape input的第一位batch_size
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      #touch.eye的parameters如self.k是rows的数量
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      #Tensor.is_cuda:Is True if the Tensor is stored on the GPU,False otherwise
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        #[2,3,32]
        xb = F.relu(self.bn1(self.conv1(xb)))
        #[2,64,32]
        matrix64x64 = self.feature_transform(xb)
        matrixnx64 = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)


        #[2,64,32]
        xb = F.relu(self.bn2(self.conv2(matrixnx64)))
        #[2,128,32]
        xb = self.bn3(self.conv3(xb))
        #[2,1024,32]
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        #[2,1024,1]
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64, matrixnx64


class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, 2, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(2)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        xb, matrix3x3, matrix64x64, matrixnx64 = self.transform(input)
        xb = xb.unsqueeze(-1).expand(-1, -1, input.shape[-1])
        xb = torch.cat((matrixnx64, xb), dim = 1)

        xb = F.relu(self.bn1(self.conv1(xb)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        
        xb = F.relu(self.bn4(self.conv4(xb)))
        xb = self.bn5(self.conv5(xb))
        output = torch.nn.LogSoftmax(dim=2)(xb.transpose(1, 2))
        return output, matrix3x3, matrix64x64


def pointnetsegmentationloss(output, labels, m3x3, m64x64, alpha = 0.0):
    criterion = torch.nn.NLLLoss()
    batchsize = output.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(batchsize,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(batchsize,1,1)
    if output.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    output = output.reshape(-1, 2)
    labels = labels.reshape(-1)
    return criterion(output, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(batchsize)


def train(pointnet, train_loader, val_loader=None,  epochs=1400, save=True):
#Training loop:可以在这里找到预训练模型
#training loop中的常见概念：batch size:我们的数据是分批喂给模型的，值得是单个批次的样本数量，
#训练一个批次称为一个迭代iteration，计算一次平均损失函数，更新一次参数，每一次迭代得到的结果都会
#被作为下一个迭代的初始值。如果数据集比较小，我们可以全数据集喂入，以为全数据集可以更好的代表整体。
#如果数据集比较大，那么此时内存可能会不够用，且采样差异性可能会导致梯度差异值相互抵消。
#如果bs为1也不行每次修正各自为政，也很难达到收敛。因此应该在合理的范围内增大bs，使内存利用率
#和并行化效率提高，使跑完一次epoch所需的迭代次数step减少，下降的方向越准，引起的训练震荡越小。
#但bs增大的同时，达到相同精度所需要的epoch数量就会相应增多。
#epoch，轮。每次使用过全部训练数据完成一次前后向训练称为完成了一轮
    best_loss = float('inf')
    best_accuracy = 0.0
    for epoch in range(epochs): 
        pointnet.train()
        running_loss = 0.0
        losses = []
        
        # for points, labels in tqdm(train_loader):
        for points, labels in train_loader:
            points = points.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()
            output, m3x3, m64x64 = pointnet(points.transpose(1,2))
            loss = pointnetsegmentationloss(output, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print statistics
            running_loss += loss.item()
            # if i % 10 == 9:   # print every 10 mini-batches
            # print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' % (epoch + 1, i + 1, len(train_loader), loss.item()))

        pointnet.eval()
        correct = total = 0

        # validation
        correct = 0.0
        total = 0.0
        i = 0
        if val_loader:
            with torch.no_grad():
                for points, labels in train_loader:
                    points = points.to(device).float()
                    labels = labels.to(device).long()
                    output, __, __ = pointnet(points.transpose(1,2))
                    predicted_label = torch.argmax(output.reshape(-1, 2), dim = -1)
                    labels = labels.reshape(-1)
                    correct = correct + (predicted_label == labels).sum().item()
                    total = total +  (len(labels))
                accuracy = correct / total
                i = i+1
                print('%d, Valid accuracy: %f , loss: %.3f' % (i, accuracy, loss.item()))

        model_save_dir = "./trainingresult/"

        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses)
            best_accuracy = accuracy
            best_model_pth_name = f'best_model.pth'
            torch.save(pointnet.state_dict(), model_save_dir + best_model_pth_name)
        
        # if epoch % 10 == 0:
        print('Valid accuracy: %f , loss: %.3f, ________' % (best_accuracy, best_loss))
        # print(f'Best_acc____{best_accuracy}, best_loss_{best_loss}')
        
        # figure_save_dir = f'./figures'
        # if not os.path.exists(figure_save_dir):
        #     os.mkdir(figure_save_dir)

        # figure_save_path = f'{figure_save_dir}/epoch_{epoch}.png'

        # show_trained_moel(model_save_dir + best_model_pth_name, figure_save_path)


def show_trained_moel(model_save_path, figure_save_path = None):
    pointnet = PointNet()
    pointnet.to(device)
    pointnet.load_state_dict(torch.load(model_save_path))
    datasets = PointCloudDataset()
    points = datasets[0][0]
    points = torch.from_numpy(points).unsqueeze(0).transpose(1,2).to(device).float()
    pointnet.eval()
    outputs, _, _ = pointnet.forward(points)
    outputs = outputs.squeeze()
    predict_labels = torch.argmax(outputs, dim =1 ).numpy()

    c=['r' if predict_labels[i] == 1 else 'k' for i in range(len(predict_labels))]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],c = c,s=0.5)
    plt.title("segmentation result")
    # plt.savefig(figure_save_path)
    # plt.close()
    plt.show()
    


#测试
if __name__ == '__main__':

    datasets = PointCloudDataset()
    train_loader = DataLoader(dataset=datasets, batch_size=2, shuffle=True)
    valid_loader = DataLoader(dataset=datasets, batch_size=64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    pointnet = PointNet()
    pointnet.to(device);
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.0001)
    
    show_trained_moel(model_save_path = f"./trainingresult/best_model.pth")

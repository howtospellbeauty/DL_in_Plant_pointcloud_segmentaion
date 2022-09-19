import numpy as np
import math
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from path import Path
import torch.nn as nn
import torch.nn.functional as F
import json
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

#读文件
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


#Sample points
class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points

        
#归一化：Unit sphere
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud


#增加：加入点云和随机噪声
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


#将原始的PILImage格式或者numpy.array格式转换为可被pytorch快速处理的长两个是
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)


#对图像进行各种转换操作，并用函数compose将这些转换操作组合起来
def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])


#Dataset
#创建一个自定义的Pytorch数据集
class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 
                'category': self.classes[category]}


#Model
class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
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

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        matrixnx64 = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(matrixnx64)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
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


def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):

    """
    output shape == (batch_size, 2, num_points)
    lables shape  (batch_size          
    """


    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)        


def pointnetsegmentloss(outputs, labels, m3x3, m64x64, alpha = 0.0):
    """
    output shape == (batch_size, 2, num_points)
    lables shape  (batch_size, num_points)
    """

    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1          
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    criterion = torch.nn.NLLLoss()
    outputs = outputs.reshape(-1 ,2)
    labels = labels.reshape(-1)
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)



#Training loop:可以在这里找到预训练模型
def train(pointnet, train_loader, val_loader=None,  epochs=200, save=True):
    for epoch in tqdm(range(epochs)): 
        pointnet.train()
        running_loss = 0.0
        i = 0
        for points, labels in train_loader:
            points = points.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(points.transpose(1,2))
            loss = pointnetsegmentloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:    # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                running_loss = 0.0
            i = i+1

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for points, labels in train_loader:
                    points = points.to(device).float()
                    labels = labels.to(device).float()
                    outputs, __, __ = pointnet(points.transpose(1,2))
                    outputs = outputs.squeeze()
                    predict_labels = torch.argmax(outputs, dim = 2)

                    total += labels.size(0) * labels.size(1)
                    correct += (predict_labels == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save:
            torch.save(pointnet.state_dict(), "save_"+str(epoch)+".pth")
        
class PointsCloudsDataset(Dataset):
    def __init__(self):
        #r代表了文件打开方式为只读，rb:以二进制打开一个文件用于只读且文件指针将会放在文件开头rb是默认格式，etc
        #also can be written in: f = open('path',) next:data = json.load(f)
        with open('import-pointclouds-pcd/ann/sink_pointcloud.pcd.json','r') as f:
            data = json.load(f)

        pcd = o3d.io.read_point_cloud("./import-pointclouds-pcd/pointcloud/sink_pointcloud.pcd")
        pcd = np.asarray(pcd.points)
        labels = np.ones(pcd.shape[0])
        incomplete_data = data['figures'][0]['geometry']['indices']
        pen_indices = data['figures'][1]['geometry']['indices']
        whole_indices = pen_indices+incomplete_data
        labels[whole_indices] = 0

        self.cloud_points = pcd
        self.labels = labels
        self.total_points = pcd.shape[0]
    #在真实数据中，有多少个样本，len就为多少，这个地方比较特殊
    def __len__(self):
        return 2
    def __getitem__(self, index):
        sampler = np.random.choice(self.total_points, size = int(0.9*self.total_points), replace = False)
        return self.cloud_points[sampler, :], self.labels[sampler]
    

def show_trained_moel():
    pointnet = PointNet()
    pointnet.to(device)
    pointnet.load_state_dict(torch.load('save_199.pth'))
    datasets = PointsCloudsDataset()
    points = datasets.cloud_points
    points = torch.from_numpy(points).unsqueeze(0).transpose(1,2).to(device).float()
    pointnet.eval()
    outputs, _, _ = pointnet.forward(points)
    outputs = outputs.transpose(1,2).squeeze()
    predict_labels = torch.argmax(outputs, dim = 0).numpy()

    c=['r' if predict_labels[i] == 1 else 'k' for i in range(len(predict_labels))]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(datasets.cloud_points[:, 0], datasets.cloud_points[:, 1], datasets.cloud_points[:, 2],c = c,s=0.5)
    plt.title("result")
    plt.show()    


#测试
if __name__ == '__main__':
    datasets = PointsCloudsDataset()
    train_loader = DataLoader(dataset=datasets, batch_size=2, shuffle=True)
    valid_loader = DataLoader(dataset=datasets, batch_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pointnet = PointNet()
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)

    train(pointnet, train_loader, valid_loader,  save=True)
    # show_trained_moel()

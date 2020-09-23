import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import models
import os
from torch import optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
path1 = r"./face_model/model60.t"
path2 = r"./face_model/arc60.t"
summary = SummaryWriter("./logs")

class Arc(nn.Module):
    def __init__(self,feature_dim,cls_dim):
        super().__init__()
        self.W=nn.Parameter(torch.randn(feature_dim,cls_dim))

    def forward(self, feature,m=1,s=10):
        x = F.normalize(feature,dim=1)#x/||x||
        w = F.normalize(self.W, dim=0)#w/||w||
        cos = torch.matmul(x, w)/10
        a = torch.acos(cos)
        top = torch.exp(s*torch.cos(a+m))
        down2 = torch.sum(torch.exp(s*torch.cos(a)),dim=1,keepdim=True)-torch.exp(s*torch.cos(a))

        out=torch.log(top/(top+down2))
        return out
data_dir = r'./detect_imgv5'
train_dataset = torchvision.datasets.ImageFolder(root=data_dir,
                                                 transform=transforms.Compose(
                                                     [
                                                         #transforms.RandomResizedCrop(224),
                                                         transforms.Resize([224,224]),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))
                                                     ]))
# train_dataset = torchvision.datasets.ImageFolder(root=data_dir,
#                                                  transform=transforms.Compose(
#                                                      [
#                                                          transforms.RandomResizedCrop(224),
#                                                          transforms.RandomHorizontalFlip(),
#                                                          transforms.ToTensor(),
#                                                          transforms.Normalize(
#                                                              mean=(0.485, 0.456, 0.406),
#                                                              std=(0.229, 0.224, 0.225))
#                                                      ]))

# val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'),
#                                                transform=transforms.Compose(
#                                                      [
#                                                          transforms.RandomResizedCrop(224),
#                                                          transforms.RandomHorizontalFlip(),
#                                                          transforms.ToTensor(),
#                                                          transforms.Normalize(
#                                                              mean=(0.485, 0.456, 0.406),
#                                                              std=(0.229, 0.224, 0.225))
#                                                      ]))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=True)
model.to(device)
if os.path.exists(path1):
    model.load_state_dict(torch.load(path1))

arc = Arc(1000,500).to(device)
if os.path.exists(path2):
    arc.load_state_dict(torch.load(path2))
loss_fun = nn.NLLLoss()

optimzer_model = torch.optim.Adam(model.parameters())
optmizer_arc = torch.optim.Adam(arc.parameters())

def Train():
    epoch = 0
    while True:

        feat_loader = []
        label_loader = []
        for i, sample_batch in enumerate(train_dataloader):
            y = sample_batch[0].to(device)
            label = sample_batch[1].to(device)

            model.train()
            mo = model(y)
            out = arc(mo)
            loss = loss_fun(out,label)
            optimzer_model.zero_grad()
            optmizer_arc.zero_grad()
            loss.backward()
            optimzer_model.step()
            optmizer_arc.step()
            #feat_loader.append(mo)
            #label_loader.append(label)

            if i % 10 == 0:
                print("epoch:{0}    loss:{1}".format(epoch+1,loss.item()))
                summary.add_scalar("loss",loss.item(),epoch)
        #feat = torch.cat(feat_loader, 0)
        #labels = torch.cat(label_loader, 0)
        epoch += 1
        if epoch % 10 ==0:
        #visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
            torch.save(model.state_dict(), f"./face_model/model.{epoch}t")
            torch.save(arc.state_dict(), f"./face_model/arc.{epoch}t")

if __name__ == '__main__':
    train = Train()
    print(train)
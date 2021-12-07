# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from unet import Unet
from DataHelper import *
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

# 设置随机数种子
# seed = int(np.random.randint(0, 100))

PATH = './unet_model.pt'

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train_model(model, criterion, optimizer, dataload,scheduler,dataloaders_test, num_epochs=11):
    best_model = model
    min_loss = None
    for epoch in range(num_epochs):
        s  = 'Epoch:'+str(epoch+1)+'/'+str(num_epochs)
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d\\%d,train_loss:%0.3f" % (step, (dt_size - 1) \\\\ dataload.batch_size + 1, loss.item()))
        print("epoch:%d lr:%.6f loss:%0.6f" % (epoch,scheduler.get_lr()[-1],epoch_loss/step))
        
        if not min_loss or (epoch_loss/step) < min_loss:
            min_loss = (epoch_loss/step)
            best_model = model
        if (epoch+1) % 5 == 0:
            scheduler.step()    
            torch.save(best_model,PATH)
    test(dataloaders_test, best_model)
    # return best_model

def loss(pre, target):
    batch = pre.shape[0]
    flag_1 = (target > 0.01).int()
    flag_2 = (target <= 0.01).int()
    Ht = pre*flag_1 + (1-pre)*flag_2
    bce = 0.5*target*torch.log(pre) 
    focal = 0.5/100*(1-Ht)*torch.log(Ht) 
    loss_ = -(bce+focal).mean()
    return loss_


# 训练模型
def train():
    train_dataset = TrainDataset('TrainingData')
    batch_size = 4
    dataloaders = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    test_dataset = TrainDataset("Test1Data", 2)
    dataloaders_test = DataLoader(test_dataset, batch_size=1)
    for seed in range(1):
        print('-'*30)
        print('the seed is %d'%seed)
        setup_seed(seed)
        model = Unet(1, 20).to(device)
        criterion = loss
        optimizer = optim.Adam(model.parameters(),lr = 0.0001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        train_model(model, criterion, optimizer, dataloaders,scheduler,dataloaders_test)
        print('-'*30)


def test(dataloaders, model):
    model.eval()
    with torch.no_grad():
        num = 0
        all_dis = 0
        d20 = 0
        d25 = 0
        d30 = 0
        d40 = 0
        for x,pos in dataloaders:
            num += 19
            # print(num/19)
            y = model(x.to(device)).cpu()
            y = y.reshape(20,-1)
            # print(y)
            # print(y)
            ind = y.argmax(dim=1)
            for i in range(19):
                pos_x, pos_y = float(ind[i]//290), float(ind[i]%290)
                pos_x_t, pos_y_t = pos[i]
                pos_x_t, pos_y_t = float(pos_x_t), float(pos_y_t)
                x_pix = abs(pos_x-pos_x_t)*0.4
                y_pix = abs(pos_y-pos_y_t)*0.4
                dis = x_pix**2+y_pix**2
                all_dis += dis
                if dis <= 2:
                    d20 +=1
                    d25 +=1
                    d30 +=1
                    d40 +=1
                elif dis <= 2.5:
                    d25 +=1
                    d30 +=1
                    d40 +=1
                elif dis < 3:
                    d30 +=1
                    d40 +=1  
                elif dis <=4:
                    d40 +=1                 
        print('MRE:%.2f'%(all_dis/num))
        print('SDR2.0:%.2f'%(d20/num))
        print('SDR2.5:%.2f'%(d25/num))
        print('SDR3.0:%.2f'%(d30/num))
        print('SDR4.0:%.2f'%(d40/num))


if __name__ == '__main__':
    print("开始训练")
    train()
    print("训练完成，保存模型")
    print("-"*20)
    # print("开始预测")
    # test()


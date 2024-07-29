from dataset import ben
from model import model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
#import numpy as np
#from metrics import calculate_metrics
from metrics import batch_metrics
#from optim_to import optimizer_to

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = '/data/Tselkovoy/dataset/BigEarthNet-v1.0'
train_data = ben(root, train = True)
test_data = ben(root, train = False)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)
print(len(train_loader))

net = model()
#net = nn.DataParallel(net)
#print(torch.cuda.device_count())
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#PATH_8 = 'file_4.pt'
#PATH = 'file.pt'
#checkpoint = torch.load(PATH)
#net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.to(device)
#optimizer_to(optimizer,device)
net.train()

test_freq = 20
for epoch in range(4):
    running_loss = 0.0
    train_acc = 0.0
    for iteration, batch in enumerate(train_loader):
        x1, x2, x3, label = batch
        optimizer.zero_grad()
        y_pred = net(x1.to(device), x2.to(device), x3.to(device))
        loss = loss_fn(y_pred, label.to(device))
        loss.backward()
        optimizer.step()
        print(iteration)
        running_loss += loss.item()
        train_acc += batch_metrics(y_pred.cpu(), label.cpu())
        '''if iteration % 100 == 99:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss/10,
                'acc': train_acc/10
                }, PATH_8)'''
        if iteration % test_freq == test_freq - 1:
            print('train loss: %.3f' % (running_loss/test_freq))
            print('train acc: %.3f' % (train_acc/test_freq))
            
            '''with open('00new_train_loss.txt', 'a') as f:
                f.write(str(running_loss/test_freq) + '\n')
                
            with open('00new_train_acc.txt', 'a') as f:
                f.write(str(train_acc/test_freq) + '\n')'''
            
            train_acc = 0.0
            running_loss = 0.0
        if iteration % test_freq == test_freq - 1:
            net.eval()
            with torch.no_grad():
                #model_result = []
                #targets = []
                test_loss = 0.0
                test_loss_01 = 0.0
                test_loss_10 = 0.0
                test_acc = 0.0
                test_acc_01 = 0.0
                test_acc_10 = 0.0
                for i_test, b_test in enumerate(test_loader):
                    x1, x2, x3, label = b_test
                    y_pred = net(x1.to(device), x2.to(device), x3.to(device))
                    loss = loss_fn(y_pred, label.to(device))
                    test_loss += loss.item()
                    test_acc += batch_metrics(y_pred.cpu(), label.cpu())
                    
                    x_1 = torch.zeros_like(x1)
                    y_pred = net(x_1.to(device), x2.to(device), x3.to(device))
                    loss = loss_fn(y_pred, label.to(device))
                    test_loss_01 += loss.item()
                    test_acc_01 += batch_metrics(y_pred.cpu(), label.cpu())
                    
                    x_2 = torch.zeros_like(x2)
                    y_pred = net(x1.to(device), x_2.to(device), x3.to(device))
                    loss = loss_fn(y_pred, label.to(device))
                    test_loss_10 += loss.item()
                    test_acc_10 += batch_metrics(y_pred.cpu(), label.cpu())
                    
                    #model_result.extend(y_pred.cpu().numpy())
                    #targets.extend(label.cpu().numpy())
    
                    if i_test == 9:
                        print('test loss: %.3f' % (test_loss/10))
                        print('test acc: %.3f' % (test_acc/10))
                        
                        with open('00new3_test_loss_11.txt', 'a') as f:
                            f.write(str(test_loss/10) + '\n')
                            
                        with open('00new3_test_loss_01.txt', 'a') as f:
                            f.write(str(test_loss_01/10) + '\n')
                            
                        with open('00new3_test_loss_10.txt', 'a') as f:
                            f.write(str(test_loss_10/10) + '\n')
                            
                        with open('00new3_test_acc_11.txt', 'a') as f:
                            f.write(str(test_acc/10) + '\n')
                            
                        with open('00new3_test_acc_01.txt', 'a') as f:
                            f.write(str(test_acc_01/10) + '\n')
                            
                        with open('00new3_test_acc_10.txt', 'a') as f:
                            f.write(str(test_acc_10/10) + '\n')
                        
                        break
            #result = calculate_metrics(np.array(model_result), np.array(targets))
            #print("micro f1: {:.3f} ".format(result['micro/f1']))
    
            net.train()



        
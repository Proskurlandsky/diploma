from dataset import ben
from model import model
from torch.utils.data import DataLoader
import torch
#import torch.nn as nn
from metrics import batch_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = '/data/Tselkovoy/dataset/BigEarthNet-v1.0'
test_data = ben(root, train = False)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)
net = model()

PATH = 'file_4.pt'
checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['model_state_dict'])
net.to(device)
net.eval()
#loss_fn = nn.BCELoss()

l = len(test_loader)
print(l)

with torch.no_grad():
    #model_result = []
    #targets = []
    test_loss = 0.0
    test_acc = 0.0
    for i_test, b_test in enumerate(test_loader):
        x1, x2, x3, label = b_test
        y_pred = net(x1.to(device), x2.to(device), x3.to(device))
        #loss = loss_fn(y_pred, label.to(device))
        #test_loss += loss.item()
        test_acc += batch_metrics(y_pred.cpu(), label.cpu())
        #model_result.extend(y_pred.cpu().numpy())
        #targets.extend(label.cpu().numpy())
        print(i_test)

    #print('test loss: %.3f' % (test_loss/l))
    print('test acc: %.3f' % (test_acc/l))
    
    
#result = calculate_metrics(np.array(model_result), np.array(targets))
#print("micro f1: {:.3f} ".format(result['micro/f1']))
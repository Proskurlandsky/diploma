from dataset import ben
#from second import model
#from torch.utils.data import DataLoader
#import torch
#from metrics import batch_metrics

root = '/data/Tselkovoy/dataset/BigEarthNet-v1.0'
train_data = ben(root, train = True)
test_data = ben(root, train = False)
#train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
#test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
#net = model()
#PATH = 'file.pt'
#checkpoint = torch.load(PATH)
#net.load_state_dict(checkpoint['model_state_dict'])
#net.eval()

#x1, x2, x3, label = next(iter(train_loader))
#x = net(x1,x2,x3)
print(len(train_data))
print(len(test_data))
#print(label)

#acc = batch_metrics(x, label)
#print(f"acc {acc}")

#x1, x2, x3, label = next(iter(test_loader))
#x = net(x1,x2,x3)
#print(x)
#print(label)

#acc = batch_metrics(x, label)
#print(f"acc {acc}")

#print(checkpoint['epoch'])

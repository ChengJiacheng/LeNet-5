# from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import visdom
import onnx

# viz = visdom.Visdom()
from PIL import Image
import numpy as np
from network import LeNet, MLP

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

# net = LeNet()
net = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        # if viz.check_connection():
        #     cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
        #                              win=cur_batch_win, name='current_batch_loss',
        #                              update=(None if cur_batch_win is None else 'replace'),
        #                              opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    torch.onnx.export(net, dummy_input, "lenet.onnx")

    onnx_model = onnx.load("lenet.onnx")
    onnx.checker.check_model(onnx_model)

def get_feature(embedding, loader, dim_embedding, l2_normalize=0, tencrop=False, eval=True):
    embedding = embedding.cuda()
    if eval:
        embedding.eval()  # Set model to evaluate mode
    else:
        embedding.train()
    features = np.zeros((len(loader.dataset), dim_embedding))
    labels = np.zeros((0))

    with torch.no_grad():
        idx = 0
        # Iterate over data.
        for step, (inputs, targets) in enumerate(loader):
            if step%100 == 0:
                print(step)

            inputs = inputs.cuda()
            targets = targets.cuda()

            if tencrop:
                bs, ncrops, c, h, w = inputs.size()
                # outputs = embedding(inputs.view(-1, c, h, w))  # fuse batch size and ncrops
                outputs =  nn.parallel.data_parallel(embedding, inputs.view(-1, c, h, w))
                outputs = outputs.view(bs, ncrops, -1).mean(1)
            else:
                # outputs = embedding(inputs)
                outputs = nn.parallel.data_parallel(embedding, inputs)

            if l2_normalize:
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)

            outputs = outputs.detach().cpu().numpy().squeeze()
            # features = np.vstack([features, outputs])
            features[idx:idx+len(targets), :] = outputs
            idx = idx + len(targets)
            targets = targets.detach().cpu().numpy().squeeze()
            labels = np.append(labels, targets)

    return features, labels



def main():
    for e in range(1, 16):
        train_and_test(e)


if __name__ == '__main__':
    main()
    # torch.save(net.state_dict(), 'net.pth')
    # torch.save(net.state_dict(), 'mlp.pth')
    
    
    # net.load_state_dict(torch.load('net.pth'))
    # net.load_state_dict(torch.load('mlp.pth'))
    
    # transform=transforms.Compose([
    #                   transforms.Resize((32, 32)),
    #                   transforms.ToTensor()])
    
    # # X_train = data_train.data.numpy()
    # # X_train = Image.fromarray(X_train)
    
    # import copy
    # embedding = copy.deepcopy(net)
    # embedding.f5 = nn.Sequential()

    import copy
    embedding = copy.deepcopy(net)
    embedding.fc3 = nn.Sequential()
    
    features, labels = get_feature(embedding, data_train_loader, dim_embedding=20)

    import scipy.io as sio
    # sio.savemat("result.mat", {'X_train': features, 'y_train': labels})
    
    sio.savemat("result_mlp.mat", {'X_train': features, 'y_train': labels})

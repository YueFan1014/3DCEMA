import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
import os
import argparse
import model
import torch
import dataset
import matrices
import bin_normalization
import datetime


parser = argparse.ArgumentParser(description='3DCEMA Training')
parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
parser.add_argument('--num_of_classes', default=3, type=int, help='num of classes')
parser.add_argument('--max_limit', default=5000, type=int, help='limit of matrices')
parser.add_argument('--training_directory', type=str, help='training directory')
parser.add_argument('--testing_directory', type=str, help='testing directory')
parser.add_argument('--model_name', type=str, help='model name')

start = datetime.datetime.now()
args = parser.parse_args()
print('Training: ', args.training_directory)
print('Testing: ', args.testing_directory)
bin_normalization.build_bin_normalized_matrix(args.training_directory)
bin_normalization.build_bin_normalized_matrix(args.testing_directory)
if args.num_of_classes == 3:
    matrices.create_in_silico_training_matrices(path=args.training_directory, k_limit=args.max_limit)
    matrices.create_in_silico_training_matrices(path=args.testing_directory, k_limit=args.max_limit)
else:
    matrices.create_real_data_training_matrices(path=args.training_directory, k_limit=args.max_limit)
    matrices.create_real_data_training_matrices(path=args.testing_directory, k_limit=args.max_limit)
training_set = dataset.DataSet(root=args.training_directory+r'dataset\\')
trainloader = torch.utils.data.DataLoader(training_set, batch_size=100, shuffle=True, num_workers=0)
testing_set = dataset.DataSet(root=args.testing_directory+r'dataset\\')
testloader = torch.utils.data.DataLoader(testing_set, batch_size=100, shuffle=False, num_workers=0)
end = datetime.datetime.now()
print(end-start)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
print('==> Building model..')
net = model.resnet10(sample_size=16, sample_duration=16, num_classes=args.num_of_classes)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.LongTensor)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs.float())
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #print('targets')
            #print(targets)
            #print('predicted')
            #print(predicted)
    acc = 100.*correct/total
    print(acc)
    if not os.path.isdir('trained_models'):
        os.mkdir('trained_models')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, r'trained_models\\' + args.model_name)
        best_acc = acc


start = datetime.datetime.now()
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
end = datetime.datetime.now()
print(end-start)

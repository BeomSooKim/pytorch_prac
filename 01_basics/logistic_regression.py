#%%
import torch
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
#%%
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root = './data',
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
#%%
model = nn.Linear(input_size, num_classes)

criterion = nn.CrossEntropyLoss() #softmax layer included nn.LogSoftmax + nn.NLLLoss
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch {}/{}, Step {}/{}, loss :{:.4f}'\
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum()

    print('Accuracy : {}%'.format(100*correct/total))

#torch.save(model.state_dict(), 'model.ckpt')
#%%
#nn.CrossEntropyLoss : nn.Softmax + nn.NLLLoss
# nn.BCEWithLogitsLoss : nn.Sigmoid + nn.BCELoss
# label은 one-hot encoding 안해도됨(index로 들어감)
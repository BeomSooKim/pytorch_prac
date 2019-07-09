#%%
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms 
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST('../data',
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)

test_dataset = torchvision.datasets.MNIST('../data',
                                            train = False,
                                            transform = transforms.ToTensor(),
                                            download = True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
#%%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_Step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 ==0:
            print('Epoch {}/{} Step {}/{}, loss {:.4f}'.\
                format(epoch+1, num_epochs, i+1, total_Step, loss.item()))

#%%
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # value, index
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    print('Accuracy : {}'.format(100* correct/ total))

#torch.save(model.state_dict(), 'model.ckpt')

# test 할 떄 no_grad()로 효율성 높임
# max함수의 리턴값은 value와 index

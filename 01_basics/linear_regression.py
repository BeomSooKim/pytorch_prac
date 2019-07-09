#%%
import torch
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
#%%
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001
#%%
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# batch size is 15, input size is 1
print(x_train.shape, y_train.shape)
model = nn.Linear(input_size, output_size)
lossfun = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#%%
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = model(inputs)
    loss = lossfun(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 ==0:
        print('Epoch {}/{}, loss : {:.4f}'.format(epoch+1, num_epochs, loss.item()))

#%%
predicted = model(torch.from_numpy(x_train)).detach().numpy()

plt.plot(x_train, y_train, 'ro', label = 'Original')
plt.plot(x_train, predicted, label = 'Fitted')
plt.legend()
plt.show()

#torch.save(model.state_dict(), model.ckpt)

# state_dict : model을 dict형태로 전환?load시 model을 선언하고 weight를 load하는 식으로 해야하는듯
# loss 계산시 예측치가 먼저
# zero_grad() 는 항상 써주는 버릇
# predict 시에도 input을 torch tensor로 바꿔서 전달해야함
# zero_grad()가 필요한이유 : optimizer는 gradient를 누적해서 계산하기 때문
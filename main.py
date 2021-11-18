#!/usr/bin/env python
# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
import wandb

wandb.init(project="wandb_practice_pytorch", entity="poomstas")

wandb.config = {
    'learning_rate': 0.001, 
    'epochs': 100, 
    'batch_size' : 128,
}

# %%
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

print(train_data)

# %%
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

# %%
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %%
loaders = {
    'train' : DataLoader(train_data, 
                         batch_size=100, 
                         shuffle=True, 
                         num_workers=1),
    
    'test'  : DataLoader(test_data, 
                         batch_size=100, 
                         shuffle=True, 
                         num_workers=1),
}
loaders

# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # Flatten
        output = self.out(x)
        return output, x # Output the last layer as well

# %%
cnn = CNN().to(device)
print(cnn)

loss_func = nn.CrossEntropyLoss()   
loss_func

optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
optimizer

# %%
NUM_EPOCHS = 15

def train(num_epochs, cnn, loaders, device):
    cnn.train()

    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        wandb.watch(cnn)

        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            b_y = Variable(labels).to(device)   # batch y

            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            wandb.log({'loss': loss})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
train(NUM_EPOCHS, cnn, loaders, device)

# %%
def test():
    cnn.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            test_output, _ = cnn(images.to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels.to(device)).sum().item() / float(labels.size(0))
        print('Test Accuracy of the model on the 10000 test images: %.3f' % accuracy)

test()

# %%

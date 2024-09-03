import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import tqdm

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BATCH_SIZE = 64
set_seed(42)

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("shape of train_dataset:", train_dataset[0][0].shape)
print("train_dataset:", train_dataset)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
          torch.nn.Linear(28*28, 512),
          torch.nn.ReLU(),
          torch.nn.Linear(512, 256),
          torch.nn.ReLU(),
          torch.nn.Linear(256, 10),
          torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.model(x)
        return x

model = Model()
criterion = torch.nn.NLLLoss()
# Adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# SGD optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# Momentum optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
  total_loss = 0

  for images, labels in tqdm.tqdm(train_loader):
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  print("epoch:", epoch, "loss:", total_loss)

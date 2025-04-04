import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm


epochs = 1  

# DNN definieren
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
        return x

# FashionMNIST Dataset laden
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model Loss und Optimizer Instanz erstellen
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
for epoch in range(epochs):
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")


# evaluation
model.eval()
test_loss = 0.0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)  # batch loss aufsummieren
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

# durchschnittlicher test loss und genauigkeit
test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)

print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
import torch
from torchvision import transforms
from pokemontcgsdk import Card, RestClient
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, optim
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data.dataset import Dataset
import pickle


class CustomPokemonCardDataset(Dataset):

    cards = list
    
    def __init__(self, d, t, tran):
        super().__init__()
        file = open('dataset')
        self.transform = tran
        self.download = d
        self.train = t
        self.cards = [Card.find('xy1-1'), Card.find('xy1-2')]

    
    def __len__(self):
        return len(self.cards)
    
    def __getitem__(self, idx):
        card = self.cards[idx]
        image_data = requests.get(card.images.large)
        image = Image.open(BytesIO(image_data.content))
        image = self.transform(image)
        label = card.name

        return image, label

mean, std = (0.5,), (0.5,)
trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                              ])

print('getting cards')
training_data = CustomPokemonCardDataset(True, True, trans)
training_loader = DataLoader(training_data, batch_size=3)

print('makeing model')

class card_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(10440, 5240)
    self.fc2 = nn.Linear(5240,2670)
    self.fc3 = nn.Linear(2670,898)
    
  def forward(self, x):
    x = x.view(x.shape[0], -1)
    
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    x = F.log_softmax(x, dim=1)
    
    return x

model = card_model()

print('finished')
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 2

for i in range(num_epochs):
    cum_loss = 0

    for images, labels in training_loader:
        optimizer.zero_grad()
        images = images.view(images.size(0), -1)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        cum_loss += loss.item()
     
    print(f"Training loss: {cum_loss/len(training_loader)}")
print('done')
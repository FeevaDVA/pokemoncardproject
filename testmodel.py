from io import BytesIO
import script1
import requests
import torch
from PIL import Image
import numpy as np
from pokemontcgsdk import Card, RestClient
from torchvision import transforms

RestClient.configure("c8b6aceb-3e30-4d46-b006-8c57a545f3c6")

trans = transforms.Compose([transforms.ToTensor()])
target = transforms.Compose([transforms.Resize((330, 240))])

card = Card.find('swshp-SWSH102')
response = requests.get(card.images.small)
img = Image.open(BytesIO(response.content))
img = trans(img)
img = target(img)
model = torch.load(r"C:\Users\Hvdwi\PycharmProjects\pokemoncardproject\sweetmodel")
model.eval()
output = model(img)
prediction = np.argmax(output)
print(prediction)





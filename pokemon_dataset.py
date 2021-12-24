from os import write
from pokemontcgsdk import Card, RestClient
import pickle

RestClient.configure("c8b6aceb-3e30-4d46-b006-8c57a545f3c6")

cards = [Card.find('xy1-1')]
with open('dataset', 'wb') as f:
    pickle.dump(cards, f)
    print(Card.find('xy1-1').images.large)
    print('done')
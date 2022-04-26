import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# tokenize all words
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# exclude punctuation
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# use set to get unique elements
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # create bag of words on the tokenized patterns
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    # labels are indices of the tags
    label = tags.index(tag)
    y_train.append(label) # sometimes can be one-hot array, but we are using Cross Entropy Loss here

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples


# hyperparameters
batch_size = 8
hidden_size = 8
learning_rate = 0.001
n_epochs = 1000

# input and output size fixed
input_size = len(X_train[0]) # len of bag of words
output_size = len(tags) # len of how many tags we have

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(n_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{n_epochs}, loss = {loss.item():.4f}')

print(f'final loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete, file saved to {FILE}')

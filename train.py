import numpy as np
import random
import json
from pathlib import Path


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data_processing import bag_of_words, tokenize, lemmanization, remove_stopword, remove_punctuation, stemming, case_folding, spelling_correction
from model import NeuralNet

# load our chat-bot intents file
with open('./data/intents.json', mode='r') as json_file:
    intents = json.load(json_file)
    

all_words = [] # all words
tags = [] # target labels
documents = [] # xy

# tokenization
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        word = tokenize(pattern)
        # add to our words list
        all_words.extend(word)
        # add to xy pair
        documents.append((word, tag))

# remove puntuations
all_words = remove_punctuation(all_words)
# correct spelling
all_words = spelling_correction(all_words)
# case folding
all_words = case_folding(all_words)


# vocabulary remove duplicates and sort
vocabulary = sorted(set(all_words))
tags = sorted(list(set(tags)))

 
# print ("documents", documents)
print (len(documents), "documents")
print (len(tags), "tags", tags)
print (len(all_words), "unique stemmed words", all_words)


# create training set
X_train = []
y_train = []

for (pattern_sentence, tag) in documents:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

    

X_train = np.array(X_train)
y_train = np.array(y_train)


# print(f"X_train: {X_train[0]}")
# print(f"y_train: {y_train}")

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}")
print(f"Output size: {output_size}")
print(f'final loss: {loss.item():.4f}')


data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

MODEL_PATH = Path("./model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "chatbot_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict 
print(f"Training complete.\nSaving model to: {MODEL_SAVE_PATH}")
torch.save(obj=data, f=MODEL_SAVE_PATH)


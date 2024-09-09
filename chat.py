import random
import json
import os
import sys
import torch

from model import NeuralNet
from data_processing import bag_of_words, tokenize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('./data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "./model/chatbot_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "BukiBot"

# Evaluation metrics
accuracy = 0
total_samples = 0

def evaluate_model(sentence, intent):
    global accuracy, total_samples
    X = bag_of_words(tokenize(sentence), all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    if tag == intent:
        accuracy += 1
    total_samples += 1

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I don't have access to that information. As an AI language model, I'm constantly learning and improving, so over time I will likely become even more useful in my responses."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "who are you?"
        sentence = input("You: ")
        if sentence == "quit":
            break
        
        # Evaluate model on user input
        for intent in intents['intents']:
            if sentence in intent['patterns']:
                evaluate_model(sentence, intent['tag'])
                break

        resp = get_response(sentence)
        print(resp)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy/total_samples:.3f}")
    print(f"Total samples: {total_samples}")
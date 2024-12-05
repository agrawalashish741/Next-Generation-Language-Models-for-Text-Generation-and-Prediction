import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import random

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    text = re.sub(r'\bm\.\s*([a-z]+)', r'm\1', text)  # Example: "M. de" -> "Mde"
    text = re.sub(r'\bm\.\s*([a-z]+)', r'm\1', text)  # Example: "M. Morrel" -> "Mmorrel"
    text = re.sub(r'\b(mr|mrs|ms|dr)\.', r'\1', text)
    # Replace contractions with their full forms
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " had", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r'([a-zA-Z]+)in\'', r'\1ing', text)

    # Remove commas, semicolons, and other punctuation, but keep full stops and question marks for sentence splitting
    text = re.sub(r'[^\w\s\.\?\n]', '', text)

    # Replace sentence-ending periods and question marks with a special token for splitting
    text = re.sub(r'(?<=\w)[\.\?\!](?=\s|$)', ' <END>', text)  # Replace only periods and question marks that end a sentence with <END>

    text = text.strip()
    return text

with open('/kaggle/input/auguste-dataset/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

cleaned_text = clean_text(raw_text)

# Split text into sentences using the <END> token
sentences = cleaned_text.split('<END>')

# Remove any empty sentences after splitting
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

# Tokenization
def tokenize(text):
    return text.split()

tokenized_data = [tokenize(sentence) for sentence in sentences]

total_size = len(tokenized_data)
train_size = int(total_size * 0.6)
val_size = int(total_size * 0.2)
test_size = total_size - train_size - val_size


train_data = tokenized_data[:train_size]
val_data = tokenized_data[train_size:train_size + val_size]
test_data = tokenized_data[train_size + val_size:]

# Build vocabulary from the training data only
word_counts = Counter(word for sentence in train_data for word in sentence)
vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items())}
vocab['<UNK>'] = len(vocab)
vocab['<START>'] = len(vocab)
vocab['<END>'] = len(vocab)
vocab['<PAD>'] = len(vocab)


# Convert words to indices
def encode_sentence(sentence, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in sentence]

# Encode all data
encoded_train_data = [encode_sentence(sentence, vocab) for sentence in train_data]
encoded_val_data = [encode_sentence(sentence, vocab) for sentence in val_data]
encoded_test_data = [encode_sentence(sentence, vocab) for sentence in test_data]


# Find the maximum sentence length
max_sentence_length = max(len(sentence) for sentence in encoded_train_data)
print(f"Maximum sentence length: {max_sentence_length}")


random.seed(42) 

print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

idx_to_word = {index: word for word, index in vocab.items()}
word_to_idx = {word: index for index, word in idx_to_word.items()}


from torch.utils.data import Dataset

class NgramDataset(Dataset):
    def __init__(self, data, vocab, context_size=5, pad_token='<PAD>'):
        self.data = data
        self.vocab = vocab
        self.context_size = context_size
        self.pad_token = pad_token
        self.pad_index = self.vocab.get(pad_token, len(vocab))
        self.data = self._process_data(self.data)

    def _process_data(self, data):
        processed_data = []
        for sentence in data:
            # Add <START> tokens (context_size-1) times at the beginning
            sentence = [self.vocab['<START>']] * (self.context_size - 1) + sentence + [self.vocab['<END>']]

            for i in range(len(sentence) - self.context_size):
                context = sentence[i:i + self.context_size]
                target = sentence[i + self.context_size]

                # Ensure context does not include <END> token
                if self.vocab['<END>'] in context:
                    continue
                processed_data.append((context, target))

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]

        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor



# Create datasets
train_dataset = NgramDataset(encoded_train_data, vocab)
val_dataset = NgramDataset(encoded_val_data, vocab)
test_dataset = NgramDataset(encoded_test_data, vocab)

batch_size = 64
# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

import gensim.downloader as api
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Load GloVe embeddings using Gensim
glove_vector = api.load('glove-wiki-gigaword-100')

# Create the weight matrix for embeddings
vocab_size = len(vocab)
embedding_dim = 100  
wt_matrix = np.zeros((vocab_size, embedding_dim))

words_found = 0
for word, idx in vocab.items():
    if word in glove_vector:
        wt_matrix[idx] = glove_vector[word]
        words_found += 1
    else:
        wt_matrix[idx] = np.random.normal(scale=0.2, size=embedding_dim)

# Convert the weight matrix to a PyTorch tensor
wt_matrix = torch.tensor(wt_matrix, dtype=torch.float32)

embedding_layer = nn.Embedding.from_pretrained(wt_matrix, freeze=False)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size=5,
                 embedding_matrix=None, dropout_rate=0.5, padding_idx=None, trainable_embeddings=False):
        super(NeuralLanguageModel, self).__init__()

        # Initialize the embedding layer
        if embedding_matrix is not None:
            self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=not trainable_embeddings, padding_idx=padding_idx)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.fc1 = nn.Linear(hidden_dim * context_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)  # Embed the input tokens
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))  # Apply the first linear layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Apply the second fully connected layer with ReLU activation
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # Apply log softmax for output probabilities

# Set model parameters
hidden_dim = 300
context_size = 5
padding_idx = vocab['<PAD>']


model = NeuralLanguageModel(vocab_size, embedding_dim, hidden_dim, context_size, wt_matrix, dropout_rate=0.3, padding_idx=padding_idx).to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, val_loader, criterion, optimizer, epochs=1, scheduler=None):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for context, target in train_loader:
            # Move input and target tensors to the GPU
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                context, target = context.to(device), target.to(device)

                output = model(context)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")


        model.train()

def evaluate(model, test_loader, criterion, device):
    model.to(device)  
    model.eval()
    test_loss = 0
    num_batches = 0

    with torch.no_grad():
        for context, target in test_loader:

            context, target = context.to(device), target.to(device)

            output = model(context)
            loss = criterion(output, target)
            test_loss += loss.item()
            num_batches += 1

    avg_test_loss = test_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Test Loss: {avg_test_loss:.4f}")

def calculate_sentence_and_average_perplexity(model, data_loader, vocab, loss_function, device):
    model.to(device)
    model.eval()

    sentence_perplexity = []
    total_log_sum = 0.0
    num_sentences_processed = 0

    # Invert the vocab dictionary to map indices back to words
    idx_to_word = {index: word for word, index in vocab.items()}

    with torch.no_grad():
        for context_batch, target_batch in data_loader:

            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)

            for i in range(len(context_batch)):
                context = context_batch[i]
                target = target_batch[i]

                # Forward pass
                log_probs = model(context.unsqueeze(0))

                # Compute loss
                loss = loss_function(log_probs, target.unsqueeze(0))
                sentence_loss = loss.item()

                # Calculate perplexity for the sentence
                sentence_perplexity_value = math.exp(sentence_loss) if sentence_loss < 100 else float('inf')

                # Convert the context back to words
                sentence = [idx_to_word[idx.item()] for idx in context]
                sentence_perplexity.append((sentence, sentence_perplexity_value))

                # Accumulate log_sum for average perplexity calculation
                total_log_sum += sentence_loss
                num_sentences_processed += 1

    # Calculate the average perplexity using the corrected formula
    average_perplexity = math.exp(total_log_sum / num_sentences_processed) if num_sentences_processed > 0 else float('inf')

    return sentence_perplexity, average_perplexity


# Define the loss function
loss_function = torch.nn.CrossEntropyLoss()

train(model, train_loader, val_loader, criterion, optimizer, epochs=2)

evaluate(model, test_loader, criterion, device)

sentence_perplexity, average_perplexity = calculate_sentence_and_average_perplexity(model, test_loader, vocab, loss_function, device)

text_file_path = 'neural_test_perplexity1.txt'
try:
    with open(text_file_path, 'w') as file:
        file.write(f"Average Perplexity: {average_perplexity:.4f}\n\n")
        file.write("Sentence\tPerplexity\n")  # Write header
        for sentence, perplexity in sentence_perplexity:
            # Remove <START> and <END> tokens from the sentence
            filtered_sentence = [word for word in sentence if word not in ['<START>', '<END>']]
            # Join the remaining words to form the sentence string
            sentence_str = ' '.join(filtered_sentence)
            file.write(f"{sentence_str}\t{perplexity}\n")
    print(f"Data written to {text_file_path}")
except Exception as e:
    print(f"Error writing to file: {e}")

sentence_perplexity, average_perplexity = calculate_sentence_and_average_perplexity(model, train_loader, vocab, loss_function, device)

text_file_path = 'train_neural_perplexity1.txt'
try:
    with open(text_file_path, 'w') as file:
        file.write(f"Average Perplexity: {average_perplexity:.4f}\n\n")
        file.write("Sentence\tPerplexity\n")  # Write header
        for sentence, perplexity in sentence_perplexity:
            # Remove <START> and <END> tokens from the sentence
            filtered_sentence = [word for word in sentence if word not in ['<START>', '<END>']]

            sentence_str = ' '.join(filtered_sentence)
            file.write(f"{sentence_str}\t{perplexity}\n")
    print(f"Data written to {text_file_path}")
except Exception as e:
    print(f"Error writing to file: {e}")


import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

def train_evaluate_model(vocab_size, embedding_dim, hidden_dim, dropout_rate, optimizer_type):
    # Create model with the specified hyperparameters
    model = NeuralLanguageModel(vocab_size, embedding_dim, hidden_dim, context_size, wt_matrix, dropout_rate)

    # Choose the optimizer
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    else:
        raise ValueError(f"Optimizer {optimizer_type} not recognized.")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=2)  # Consider increasing epochs

    # Calculate perplexities
    train_perplexity, losses, average_train_perplexity = calculate_sentence_and_average_perplexity(model, train_loader, vocab, loss_function, device)
    test_perplexity, losses, average_test_perplexity = calculate_sentence_and_average_perplexity(model, test_loader, vocab, loss_function, device)

    return average_train_perplexity, average_test_perplexity

# Define hyperparameter grids
dropout_rates = [0.1, 0.3]
hidden_dims = [100, 300]
optimizers = ["Adam", "SGD"]


# To store results
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Experiment with different dropout rates
hidden_dim = 200  # Use a fixed hidden dimension for this experiment
optimizer_type = 'Adam'  # Use a fixed optimizer for this experiment

for dropout_rate in dropout_rates:
    train_ppl, test_ppl = train_evaluate_model(vocab_size, embedding_dim, hidden_dim, dropout_rate, optimizer_type)
    results.append((dropout_rate, hidden_dim, optimizer_type, train_ppl, test_ppl))

# 2. Experiment with different hidden dimensions
dropout_rate = 0.1  # Use a fixed dropout rate for this experiment
optimizer_type = 'Adam'  # Use a fixed optimizer for this experiment

for hidden_dim in hidden_dims:
    train_ppl, test_ppl = train_evaluate_model(vocab_size, embedding_dim, hidden_dim, dropout_rate, optimizer_type)
    results.append((dropout_rate, hidden_dim, optimizer_type, train_ppl, test_ppl))

# 3. Experiment with different optimizer types
dropout_rate = 0.3  # Use a fixed dropout rate for this experiment
hidden_dim = 200  # Use a fixed hidden dimension for this experiment

for optimizer_type in optimizers:
    train_ppl, test_ppl = train_evaluate_model(vocab_size, embedding_dim, hidden_dim, dropout_rate, optimizer_type)
    results.append((dropout_rate, hidden_dim, optimizer_type, train_ppl, test_ppl))

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results, columns=['Dropout Rate', 'Hidden Dim', 'Optimizer', 'Train PPL', 'Test PPL'])

results_df.to_csv('hyperparameter_results.csv', index=False)

print(results_df)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame manually with the given data
data = {
    'Dropout Rate': [0.1, 0.3, 0.3, 0.3, 0.3, 0.3],
    'Hidden Dim': [200, 200, 100, 300, 200, 200],
    'Optimizer': ['Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'SGD'],
    'Train PPL': [169.419827,194.725942,216.648998,184.447159,194.266099,1837.246693 ],
    'Test PPL': [286.056779, 293.687542, 309.606266, 286.298025, 297.109286, 1746.867824]
}

df = pd.DataFrame(data)

# Plot Perplexity vs Dropout Rate
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Dropout Rate', y='Train PPL', marker='o', label='Train PPL')
sns.lineplot(data=df, x='Dropout Rate', y='Test PPL', marker='o', label='Test PPL')
plt.title('Perplexity vs Dropout Rate')
plt.xlabel('Dropout Rate')
plt.ylabel('Perplexity')
plt.legend()
plt.show()

# Plot Perplexity vs Optimizer (Line Plot)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Optimizer', y='Train PPL', marker='o', label='Train PPL')
sns.lineplot(data=df, x='Optimizer', y='Test PPL', marker='o', label='Test PPL')
plt.title('Perplexity vs Optimizer')
plt.xlabel('Optimizer')
plt.ylabel('Perplexity')
plt.legend()
plt.show()

# Plot Perplexity vs Hidden Dimension
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Hidden Dim', y='Train PPL', marker='o', label='Train PPL')
sns.lineplot(data=df, x='Hidden Dim', y='Test PPL', marker='o', label='Test PPL')
plt.title('Perplexity vs Hidden Dimension')
plt.xlabel('Hidden Dimension')
plt.ylabel('Perplexity')
plt.legend()
plt.show()

"""**LSTM**"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMDataSet(Dataset):
    def __init__(self, data,  word_to_idx, max_length):
        # Filter out sentences longer than max_length and convert them to indices
        self.data = [
            [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sentence]
            for sentence in data if len(sentence) <= max_length
        ]
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.pad_idx = word_to_idx['<PAD>']

    def pad_sentence(self, sentence):
        # Pad the sentence if it's shorter than max_length
        return [self.pad_idx] * (self.max_length - len(sentence)) + sentence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        padded_sentence = self.pad_sentence(sentence)
        return torch.tensor(padded_sentence, dtype=torch.long), torch.tensor([1 if word != self.pad_idx else 0 for word in padded_sentence], dtype=torch.long)


# Create datasets
lstm_train_dataset = LSTMDataSet(train_data, vocab, 50)
lstm_val_dataset = LSTMDataSet(val_data, vocab, 50)
lstm_test_dataset = LSTMDataSet(test_data, vocab, 50)

# Create data loaders
batch_size = 64
lstm_train_loader = DataLoader(lstm_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
lstm_val_loader = DataLoader(lstm_val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
lstm_test_loader = DataLoader(lstm_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)



class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super(LSTMModel, self).__init__()

        # Embedding layer using pre-trained GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Linear layer to map LSTM output to vocabulary size
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Softmax layer to get the probability distribution over the vocabulary
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden, cell):
        # Embedding lookup
        x = self.embedding(x)

        # LSTM forward pass (return the entire sequence of outputs)
        lstm_out, (hidden, cell) = self.lstm(x, (hidden, cell))

        # Fully connected layer applied to each time step
        out = self.fc(lstm_out)

        return out, hidden, cell

    def init_hidden(self, batch_size, hidden_dim):
 
        return (torch.zeros(1, batch_size, hidden_dim).to(device),
                torch.zeros(1, batch_size, hidden_dim).to(device))


# Model parameters
hidden_dim = 300  # Hidden dimension size
output_dim = len(vocab)  # Output dimension is the size of the vocabulary

# Initialize the model, loss function, and optimizer
model = LSTMModel(vocab_size=len(vocab),
                  embedding_dim=embedding_dim,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  embedding_matrix=wt_matrix).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, dataloader, criterion, optimizer, hidden_dim):
    model.train()
    total_loss = 0

    for batch_x, mask in dataloader:
        batch_x, mask = batch_x.to(device), mask.to(device)
        batch_size = batch_x.size(0)

        # Initialize hidden and cell states for LSTM
        hidden, cell = model.init_hidden(batch_size, hidden_dim)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through the LSTM model
        output, hidden, cell = model(batch_x, hidden, cell)  # output shape: [batch_size, seq_length, vocab_size]

        # Shift the target by one word for next word prediction
        target = batch_x[:, 1:].contiguous().view(-1)  # target shape: [batch_size * (seq_length - 1)]
        output = output[:, :-1, :].contiguous().view(-1, output_dim)  # output shape: [batch_size * (seq_length - 1), vocab_size]

        # Apply mask to ignore <PAD> tokens
        mask = mask[:, 1:].contiguous().view(-1)
        loss = criterion(output[mask == 1], target[mask == 1])

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluate function
def evaluate(model, dataloader, criterion, hidden_dim):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_x, mask in dataloader:
            batch_x, mask = batch_x.to(device), mask.to(device)
            batch_size = batch_x.size(0)

            # Initialize hidden and cell states for LSTM
            hidden, cell = model.init_hidden(batch_size, hidden_dim)

            # Forward pass
            output, hidden, cell = model(batch_x, hidden, cell)

            # Shift the target by one word for next word prediction
            target = batch_x[:, 1:].contiguous().view(-1)  # target shape: [batch_size * (seq_length - 1)]
            output = output[:, :-1, :].contiguous().view(-1, output_dim)  # Reshape output

            # Apply mask to ignore <PAD> tokens
            mask = mask[:, 1:].contiguous().view(-1)
            loss = criterion(output[mask == 1], target[mask == 1])
            total_loss += loss.item()

    return total_loss / len(dataloader)



# Hyperparameters
num_epochs = 8
batch_size = 64
hidden_dim = 300

# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, lstm_train_loader, criterion, optimizer, hidden_dim)
    val_loss = evaluate(model, lstm_val_loader, criterion, hidden_dim)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# After training, evaluate on the test set
test_loss = evaluate(model, lstm_test_loader, criterion, hidden_dim)
print(f'Test Loss: {test_loss:.4f}')


import torch
import math
import csv

# Decode sentence from indices while ignoring <PAD> tokens
def decode_sentence(indices, vocab):
    idx_to_word = {idx: word for word, idx in vocab.items()}
    sentence = [idx_to_word[idx] for idx in indices if idx in idx_to_word and idx != vocab['<PAD>']]  # Ignore PAD tokens
    return ' '.join(sentence)

# Evaluate function to calculate sentence-wise perplexity and write to CSV
def evaluate_and_calculate_perplexity(model, dataloader, criterion, hidden_dim, output_file):
    model.eval()
    total_loss = 0
    perplexities = []
    sentences = []

    # Collect sentence-wise perplexities first
    with torch.no_grad():
        for batch_x, mask in dataloader:
            batch_x, mask = batch_x.to(device), mask.to(device)
            batch_size = batch_x.size(0)

            # Initialize hidden and cell states for LSTM
            hidden, cell = model.init_hidden(batch_size, hidden_dim)

            # Forward pass through the LSTM model
            output, hidden, cell = model(batch_x, hidden, cell)  # output shape: [batch_size, seq_length, vocab_size]

            # Shift the target by one word for next word prediction
            target = batch_x[:, 1:].contiguous()  # target shape: [batch_size, seq_length - 1]
            output = output[:, :-1, :].contiguous()  # output shape: [batch_size, seq_length - 1, vocab_size]

            # Apply mask to ignore <PAD> tokens in the loss calculation
            mask = mask[:, 1:].contiguous()  

            # Loop over each sentence in the batch
            for sentence_idx in range(batch_size):
                sentence_output = output[sentence_idx]  # Shape: [seq_length - 1, vocab_size]
                sentence_target = target[sentence_idx]  # Shape: [seq_length - 1]
                sentence_mask = mask[sentence_idx]  # Mask for current sentence

                # Apply mask to get only valid tokens
                valid_output = sentence_output[sentence_mask == 1]  
                valid_target = sentence_target[sentence_mask == 1]

                # Only calculate loss for valid tokens
                if valid_output.size(0) > 0:
                    valid_output = valid_output.view(-1, valid_output.size(-1))  # Reshape to [num_valid_tokens, vocab_size]
                    valid_target = valid_target.view(-1)  # Reshape to [num_valid_tokens]

                    # Compute loss for the current sentence
                    sentence_loss = criterion(valid_output, valid_target).item()

                    # Calculate perplexity for the current sentence
                    sentence_perplexity = math.exp(sentence_loss)
                    perplexities.append(sentence_perplexity)

                    # Decode the original sentence
                    original_sentence = batch_x[sentence_idx].cpu().numpy().tolist()
                    decoded_sentence = decode_sentence(original_sentence, vocab)

                    # Store the sentence and its perplexity for later writing
                    sentences.append((decoded_sentence, sentence_perplexity))

            # Accumulate the batch loss
            valid_loss = criterion(valid_output, valid_target)
            total_loss += valid_loss.item()

    # Calculate average perplexity across all sentences
    avg_loss = total_loss / len(dataloader)
    avg_perplexity = math.exp(avg_loss)

    with open(output_file, mode='w') as file:
        file.write(f"Average Perplexity: {avg_perplexity:.4f}\n\n")
        file.write("Sentence\tPerplexity\n")  # Write the header

        # Write each sentence and its perplexity
        for sentence, perplexity in sentences:
            file.write(f"{sentence}\t{perplexity:.4f}\n")

    print(f"Perplexity written to {output_file}")
    return avg_perplexity

output_file = "LSTM_test_perplexities.txt"
average_perplexity = evaluate_and_calculate_perplexity(model, lstm_test_loader, criterion, hidden_dim, output_file)

import torch
import math
import csv

# Decode sentence from indices while ignoring <PAD> tokens
def decode_sentence(indices, vocab):
    idx_to_word = {idx: word for word, idx in vocab.items()}
    sentence = [idx_to_word[idx] for idx in indices if idx in idx_to_word and idx != vocab['<PAD>']]  # Ignore PAD tokens
    return ' '.join(sentence)

# Evaluate function to calculate sentence-wise perplexity and write to CSV
def evaluate_and_calculate_perplexity(model, dataloader, criterion, hidden_dim, output_file):
    model.eval()
    total_loss = 0
    perplexities = []
    sentences = []

    # Collect sentence-wise perplexities first
    with torch.no_grad():
        for batch_x, mask in dataloader:
            batch_x, mask = batch_x.to(device), mask.to(device)
            batch_size = batch_x.size(0)

            # Initialize hidden and cell states for LSTM
            hidden, cell = model.init_hidden(batch_size, hidden_dim)

            # Forward pass through the LSTM model
            output, hidden, cell = model(batch_x, hidden, cell)  # output shape: [batch_size, seq_length, vocab_size]

            # Shift the target by one word for next word prediction
            target = batch_x[:, 1:].contiguous()  # target shape: [batch_size, seq_length - 1]
            output = output[:, :-1, :].contiguous()  # output shape: [batch_size, seq_length - 1, vocab_size]

            # Apply mask to ignore <PAD> tokens in the loss calculation
            mask = mask[:, 1:].contiguous()  # Mask for target (shifted)

            # Loop over each sentence in the batch
            for sentence_idx in range(batch_size):
                sentence_output = output[sentence_idx]  # Shape: [seq_length - 1, vocab_size]
                sentence_target = target[sentence_idx]  # Shape: [seq_length - 1]
                sentence_mask = mask[sentence_idx]  # Mask for current sentence

                # Apply mask to get only valid tokens
                valid_output = sentence_output[sentence_mask == 1]  # Only non-PAD tokens
                valid_target = sentence_target[sentence_mask == 1]

                # Only calculate loss for valid tokens
                if valid_output.size(0) > 0:
                    valid_output = valid_output.view(-1, valid_output.size(-1))  # Reshape to [num_valid_tokens, vocab_size]
                    valid_target = valid_target.view(-1)  # Reshape to [num_valid_tokens]

                    # Compute loss for the current sentence
                    sentence_loss = criterion(valid_output, valid_target).item()

                    # Calculate perplexity for the current sentence
                    sentence_perplexity = math.exp(sentence_loss)
                    perplexities.append(sentence_perplexity)

                    # Decode the original sentence
                    original_sentence = batch_x[sentence_idx].cpu().numpy().tolist()
                    decoded_sentence = decode_sentence(original_sentence, vocab)

                    # Store the sentence and its perplexity for later writing
                    sentences.append((decoded_sentence, sentence_perplexity))

            # Accumulate the batch loss
            valid_loss = criterion(valid_output, valid_target)
            total_loss += valid_loss.item()

    # Calculate average perplexity across all sentences
    avg_loss = total_loss / len(dataloader)
    avg_perplexity = math.exp(avg_loss)

    # Write the average perplexity and sentence-level perplexities to the text file
    with open(output_file, mode='w') as file:
        # Write the average perplexity at the top
        file.write(f"Average Perplexity: {avg_perplexity:.4f}\n\n")
        file.write("Sentence\tPerplexity\n")  # Write the header

        # Write each sentence and its perplexity
        for sentence, perplexity in sentences:
            file.write(f"{sentence}\t{perplexity:.4f}\n")

    print(f"Perplexity written to {output_file}")
    return avg_perplexity

output_file = "LSTM_train_perplexities.txt"
average_perplexity = evaluate_and_calculate_perplexity(model, lstm_train_loader, criterion, hidden_dim, output_file)

"""**TRANSFORMER-DECODER**"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

class TransformerTextDataset(Dataset):
    def __init__(self, data, word_to_idx, max_length):
        # Filter out sentences longer than max_length
        self.data = [
            [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sentence]
            for sentence in data if len(sentence) <= max_length
        ]
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.pad_idx = word_to_idx['<PAD>']

    def pad_sentence(self, sentence):
        # Pad the sentence if it's shorter than max_length
        return [self.pad_idx] * (self.max_length - len(sentence)) + sentence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        padded_sentence = self.pad_sentence(sentence)
        return torch.tensor(padded_sentence, dtype=torch.long)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Define the Transformer Decoder Model
class Decoder_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, n_layers=1, nhead=2, dropout=0.1, max_len=50):
        super(Decoder_model, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, target, memory):
        target_emb = self.embedding_layer(target)
        target_emb = self.pos_encoder(target_emb)

        memory = self.pos_encoder(memory)

        # Create target mask for the decoder
        tgt_mask = generate_square_subsequent_mask(target.size(1)).to(target.device)

        output = self.decoder(tgt=target_emb, memory=memory, tgt_mask=tgt_mask)

        return self.fc(output)

def calculate_perplexity(loss):
    return math.exp(loss)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)


# Training function
def train_transformer(model, train_loader, val_loader, criterion, optimizer, num_epochs, vocab_size, device):
    model.train()

    for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      for i, inputs in enumerate(train_loader):
          inputs = inputs.to(device)
          input_seq = inputs[:, :-1]
          target_seq = inputs[:, 1:]

          optimizer.zero_grad()
          memory = model.embedding_layer(input_seq)

          output = model(input_seq, memory)
          batch_size, seq_len, vocab_size = output.shape

          output = output.view(batch_size * seq_len, vocab_size)
          target_seq = target_seq.contiguous().view(-1)

          loss = criterion(output, target_seq)
          loss.backward()

          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()

          total_loss += loss.item()

      scheduler.step()

      avg_loss = total_loss / len(train_loader)
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

      val_loss = evaluate_transformer(model, val_loader, criterion, vocab_size, device)
      print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')



def evaluate_transformer(model, dataloader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)

            # Input: remove last token; Target: remove first token
            input_seq = inputs[:, :-1]
            target_seq = inputs[:, 1:]

            memory = model.embedding_layer(input_seq)  # Assuming model has an embedding layer

            # Forward pass through the transformer model
            output = model(input_seq, memory)

            # Reshape the output and target for calculating loss
            output = output.view(-1, vocab_size)
            target_seq = target_seq.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output, target_seq)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss




# Parameters
max_len = 50  # Max sequence length
nhead = 5  # Number of attention heads
num_layers = 1  # Number of decoder layers
dim_feedforward = 2048  # Dimension of the feedforward layer
dropout = 0.1
num_epochs = 20
batch_size = 64
learning_rate = 0.0001

# Create dataset and dataloader
decoder_train_dataset = TransformerTextDataset(train_data, vocab, 50)
decoder_val_dataset = TransformerTextDataset(val_data, vocab, 50)
decoder_test_dataset = TransformerTextDataset(test_data, vocab, 50)

decoder_train_loader = DataLoader(decoder_train_dataset, batch_size=batch_size, shuffle=True)
decoder_val_loader = DataLoader(decoder_val_dataset, batch_size=batch_size, shuffle=False)
decoder_test_loader = DataLoader(decoder_test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = Decoder_model(vocab_size=len(vocab), embedding_dim=embedding_dim,  hidden_dim=hidden_dim, n_layers=num_layers, dropout=0.1, max_len=max_len).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])  # Ignore padding during loss calculation
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# Train the model
train_transformer(model, decoder_train_loader, decoder_val_loader, criterion, optimizer, num_epochs, criterion, device)

test_loss = evaluate_transformer(model, decoder_test_loader, criterion, vocab_size,device)
print(f'Test Loss: {test_loss:.4f}')

import torch
import math

def decode_sentence(tokens, vocab, pad_token="<PAD>"):
    """
    Converts a list of tokens (indices) back into a sentence using the vocabulary,
    excluding the padding tokens.
    """
    # Reverse vocabulary mapping from indices to words
    reverse_vocab = {index: word for word, index in vocab.items()}

    # Convert tokens to words and remove any PAD tokens
    sentence = [reverse_vocab[token.item()] for token in tokens if reverse_vocab[token.item()] != pad_token]

    return " ".join(sentence)

def calculate_perplexity(loss_value):
    """
    Calculate perplexity from the loss value.
    """
    return math.exp(loss_value)

def evaluate_transformer_with_perplexity(model, dataloader, criterion, vocab, device, txt_filename="decoder_test_perplexity.txt"):
    model.eval()
    total_loss = 0
    sentence_perplexities = []
    sentences = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Ensure inputs are batched; if not, add a batch dimension
            if isinstance(data, tuple) or isinstance(data, list):
                inputs = data[0]  # Assuming inputs are the first element in the batch
            else:
                inputs = data

            inputs = inputs.to(device)

            # Ensure input is 2D [batch_size, sequence_length]
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)  # Add batch dimension if necessary

            # Input: remove last token; Target: remove first token
            input_seq = inputs[:, :-1]
            target_seq = inputs[:, 1:]

            # Memory: Typically from encoder, here use input embedding directly for simplicity
            memory = model.embedding_layer(input_seq)  # Assuming model has an embedding layer

            # Forward pass through the transformer model
            output = model(input_seq, memory)

            # Reshape the output and target for calculating loss
            batch_size, seq_len, vocab_size = output.shape
            output = output.view(batch_size * seq_len, vocab_size)  # Flatten batch and sequence dimensions
            target_seq = target_seq.contiguous().view(-1)  # Flatten target sequence

            # Calculate loss for each sentence in the batch
            loss = criterion(output, target_seq)

            # Calculate perplexity for the batch
            perplexity = calculate_perplexity(loss.item())
            sentence_perplexities.append(perplexity)
            total_loss += loss.item()

            # Decode the sentence for writing to the text file (first sentence in the batch)
            decoded_sentence = decode_sentence(input_seq[0], vocab, pad_token="<PAD>")
            sentences.append((decoded_sentence, perplexity))

    # Calculate average perplexity
    avg_loss = total_loss / len(dataloader)
    avg_perplexity = calculate_perplexity(avg_loss)

    # Write the average perplexity and sentence-level perplexities to the text file
    with open(txt_filename, mode='w') as file:

        file.write(f"Average Perplexity: {avg_perplexity:.4f}\n\n")
        file.write("Sentence\tPerplexity\n")  # Write the header

        # Write each sentence and its perplexity
        for sentence, perplexity in sentences:
            file.write(f"{sentence}\t{perplexity:.4f}\n")

    print(f"Perplexity written to {txt_filename}")
    return avg_perplexity


avg_perplexity = evaluate_transformer_with_perplexity(model, decoder_test_loader, criterion, vocab, device)

import torch
import math

def decode_sentence(tokens, vocab, pad_token="<PAD>"):
    """
    Converts a list of tokens (indices) back into a sentence using the vocabulary,
    excluding the padding tokens.
    """
    # Reverse vocabulary mapping from indices to words
    reverse_vocab = {index: word for word, index in vocab.items()}

    # Convert tokens to words and remove any PAD tokens
    sentence = [reverse_vocab[token.item()] for token in tokens if reverse_vocab[token.item()] != pad_token]

    return " ".join(sentence)

def calculate_perplexity(loss_value):
    """
    Calculate perplexity from the loss value.
    """
    return math.exp(loss_value)

def evaluate_transformer_with_perplexity(model, dataloader, criterion, vocab, device, txt_filename="decoder_train_perplexity.txt"):
    model.eval()
    total_loss = 0
    sentence_perplexities = []
    sentences = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Ensure inputs are batched; if not, add a batch dimension
            if isinstance(data, tuple) or isinstance(data, list):
                inputs = data[0]  # Assuming inputs are the first element in the batch
            else:
                inputs = data

            inputs = inputs.to(device)

            # Ensure input is 2D [batch_size, sequence_length]
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)  # Add batch dimension if necessary

            # Input: remove last token; Target: remove first token
            input_seq = inputs[:, :-1]
            target_seq = inputs[:, 1:]

            # Memory: Typically from encoder, here use input embedding directly for simplicity
            memory = model.embedding_layer(input_seq)  # Assuming model has an embedding layer

            # Forward pass through the transformer model
            output = model(input_seq, memory)

            # Reshape the output and target for calculating loss
            batch_size, seq_len, vocab_size = output.shape
            output = output.view(batch_size * seq_len, vocab_size)  # Flatten batch and sequence dimensions
            target_seq = target_seq.contiguous().view(-1)  # Flatten target sequence

            # Calculate loss for each sentence in the batch
            loss = criterion(output, target_seq)

            # Calculate perplexity for the batch
            perplexity = calculate_perplexity(loss.item())
            sentence_perplexities.append(perplexity)
            total_loss += loss.item()

            # Decode the sentence for writing to the text file (first sentence in the batch)
            decoded_sentence = decode_sentence(input_seq[0], vocab, pad_token="<PAD>")
            sentences.append((decoded_sentence, perplexity))

    # Calculate average perplexity
    avg_loss = total_loss / len(dataloader)
    avg_perplexity = calculate_perplexity(avg_loss)

    # Write the average perplexity and sentence-level perplexities to the text file
    with open(txt_filename, mode='w') as file:
        # Write the average perplexity at the top
        file.write(f"Average Perplexity: {avg_perplexity:.4f}\n\n")
        file.write("Sentence\tPerplexity\n")  # Write the header

        # Write each sentence and its perplexity
        for sentence, perplexity in sentences:
            file.write(f"{sentence}\t{perplexity:.4f}\n")

    print(f"Perplexity written to {txt_filename}")
    return avg_perplexity

avg_perplexity = evaluate_transformer_with_perplexity(model, decoder_train_loader, criterion, vocab, device)
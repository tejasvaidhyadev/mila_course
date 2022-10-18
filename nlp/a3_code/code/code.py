from audioop import reverse
from cProfile import label
from doctest import Example
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### PROVIDED CODE #####

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in batch:
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):

        # read file
        with open(filepath, 'r') as f:
            self.data = f.read()
        self.unique_chars = list(set(self.data))
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars)
        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch
    
    def generate_char_mappings(self, uq):
        char_to_ix = {ch: i for i, ch in enumerate(uq)}
        ind_to_char = {i: ch for i, ch in enumerate(uq)}
        return {"char_to_idx": char_to_ix, "idx_to_char": ind_to_char}
    
    def convert_seq_to_indices(self, seq):
        seq = [self.mappings["char_to_idx"][char] for char in seq]
        return seq    

    def convert_indices_to_seq(self, seq):
        seq = [self.mappings["idx_to_char"][idx] for idx in seq]
        return seq

    def get_example(self):
        for i in range(self.examples_per_epoch):
            start = random.randint(0, len(self.data) - self.seq_len - 1)
            end = start + self.seq_len
            input_seq = self.data[start:end]
            target_seq = self.data[start+1:end+1]
            input_seq = self.convert_seq_to_indices(input_seq)
            target_seq = self.convert_seq_to_indices(target_seq)
            yield torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        # charnn init
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(n_chars, embedding_size)
        # waa is without bias 
        self.waa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wax = nn.Linear(embedding_size, hidden_size)
        self.wya = nn.Linear(hidden_size, n_chars)


    def rnn_cell(self, i, h):        
        a = self.waa(h) + self.wax(i)
        a = torch.tanh(a)
        y = self.wya(a)
        return y, a        

    def forward(self, input_seq, hidden = None):
        # forward pass
        x = self.embedding_layer(input_seq)
        if hidden is None:
            hidden = torch.zeros(self.hidden_size, dtype=torch.float32)
        y = []
        for i in x:
            y_, hidden = self.rnn_cell(i, hidden)
            y.append(y_)
        y = torch.stack(y)
        return y, hidden
        
    def get_loss_function(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def get_optimizer(self, lr):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        seq = [starting_char]
        hidden = None
        for i in range(seq_len):
            x = torch.tensor([seq[-1]], dtype=torch.long)
            y, hidden = self.forward(x, hidden)
            y = y.squeeze()
            y = F.softmax(y/temp, dim=0)
            y = Categorical(y)
            y = y.sample()
            seq.append(y)
        return seq

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars
        self.embedding_layer = nn.Embedding(n_chars, embedding_size)
        self.forget_gate = nn.Linear(embedding_size+ hidden_size, hidden_size)    
        self.input_gate = nn.Linear(embedding_size+ hidden_size, hidden_size)
        self.output_gate = nn.Linear(embedding_size+ hidden_size, hidden_size)
        self.cell_state_layer = nn.Linear(embedding_size+ hidden_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, n_chars)


    def forward(self, input_seq, hidden = None, cell = None):
        # forward pass lstm 
        x = self.embedding_layer(input_seq)
        if hidden is None:
            hidden = torch.zeros(self.hidden_size, dtype=torch.float32)
        if cell is None:
            cell = torch.zeros(self.hidden_size, dtype=torch.float32)
        y = []
        for i in x:
            y_, hidden, cell= self.lstm_cell(i, hidden, cell)
            y.append(y_)
        y = torch.stack(y)
        return y, hidden, cell

    def lstm_cell(self, i, h, c):
        #lstm cell
        #concatenate hidden and input
        x = torch.cat((i, h), dim=0)
        f_t = torch.sigmoid(self.forget_gate(x))
        i_t = torch.sigmoid(self.input_gate(x))
        c_tilda = torch.tanh(self.cell_state_layer(x))
        c = f_t * c + i_t * c_tilda
        o_t = torch.sigmoid(self.output_gate(x))
        h = o_t * torch.tanh(c)
        y_t = self.fc_output(h)
        return y_t , h, c
    
    def get_loss_function(self):
        #loss function with softmax
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def get_optimizer(self, lr):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        seq = [starting_char]
        hidden = None
        cell = None
        # loop through sequence length
        for i in range(seq_len):
            x = torch.tensor([seq[-1]], dtype=torch.long)
            y, hidden, cell = self.forward(x, hidden, cell)
            y = y.squeeze()
            y = F.softmax(y/temp, dim=0)
            y = Categorical(y)
            y = y.sample()
            seq.append(y)
        return seq

def train(model, dataset, lr, out_seq_len, num_epochs):

    # code to initialize optimizer, loss function
    optimizer = model.get_optimizer(lr)
    loss_fn = model.get_loss_function()
    # loop through epochs
    
    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            # main loop code
            optimizer.zero_grad()
            output = model(in_seq)[0]
            loss = loss_fn(output, out_seq)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += 1
        # print info every X examples            
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")
            
        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly
        # randomly sample starting char index from vocab
        with torch.no_grad():
            for i in range(3):
                # starting_char_index = np.random.randint(0, len(dataset.vocab))
                starting_char = np.random.randint(0, len(dataset.vocab))
                print("Sample: {}".format("".join(model.sample_sequence(starting_char, out_seq_len))))
        
        n = 0
        running_loss = 0

    return model # return model optionally


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10 # one epoch is this # of examples
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path, seq_len, 4)
    model = CharRNN(dataset.n_chars, embedding_size, hidden_size)
    model = model.to(device)
    # code to train model
    
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)

def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path, seq_len, 4)
    model = CharLSTM(dataset.n_chars, embedding_size, hidden_size)
    model = model.to(device)
    # code to train model

    
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)


def fix_padding(batch_premises, batch_hypotheses):

    batch_premises = [torch.tensor(premise) for premise in batch_premises]
    batch_hypotheses = [torch.tensor(hypothesis) for hypothesis in batch_hypotheses]
    reversed_batch_premises = [torch.flip(premise, [0]) for premise in batch_premises]
    reversed_batch_hypotheses = [torch.flip(hypothesis, [0]) for hypothesis in batch_hypotheses]

    batch_premises = torch.nn.utils.rnn.pad_sequence(batch_premises, batch_first=True, padding_value=0)
    batch_hypotheses = torch.nn.utils.rnn.pad_sequence(batch_hypotheses, batch_first=True, padding_value=0)
    batch_premises_reverse = torch.nn.utils.rnn.pad_sequence(reversed_batch_premises, batch_first=True, padding_value=0)
    batch_hypotheses_reverse = torch.nn.utils.rnn.pad_sequence(reversed_batch_hypotheses, batch_first=True, padding_value=0)
    
    return batch_premises, batch_hypotheses, batch_premises_reverse, batch_hypotheses_reverse
    

def create_embedding_matrix(word_index, emb_dict, emb_dim):
    # return Tensor[word_indices, emb_dim]

    embedding_matrix = np.zeros((len(word_index) , emb_dim))
    for word, i in word_index.items():
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return torch.from_numpy(embedding_matrix).float()

def evaluate(model, dataloader, index_map):
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in dataloader:
            premises = sample['premise']
            hypotheses = sample['hypothesis']
            labels = sample['label']
            tokenized_premises = tokens_to_ix(tokenize(premises), index_map)
            tokenized_hypotheses = tokens_to_ix(tokenize(hypotheses), index_map)
            # use tokens_to_ix
            output = model(tokenized_premises, tokenized_hypotheses)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
        # uni-directional LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # define intermediate layers
        self.int_layer = nn.Linear(2*hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, a, b):
        # fix padding
        a, b, a_reverse, b_reverse = fix_padding(a, b)
        a = self.embedding_layer(a)
        b = self.embedding_layer(b)
        #use the final cell state of the LSTM as 
        # the representation of the sentence
        a_out, (hn, cn_a) = self.lstm(a)
        b_out, (hn, cn_b) = self.lstm(b)
        # concatenate the final cell states of the two sentences
        # and pass through a linear layer
        cn = torch.cat((cn_a, cn_b), dim=2)
        cn = cn.view(-1, 2*self.hidden_dim)
        cn = self.int_layer(cn)
        # relu activation
        cn = F.relu(cn)
        cn = self.out_layer(cn)
        return cn
    
        

class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm_forward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # define intermediate layers
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
        self.int_layer = nn.Linear(4*hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)


    def forward(self, a, b):
        # fix padding
        a, b, a_reverse, b_reverse = fix_padding(a, b)
        a = self.embedding_layer(a)
        b = self.embedding_layer(b)
        a_reverse = self.embedding_layer(a_reverse)
        b_reverse = self.embedding_layer(b_reverse)
        a_out, (hn, cn_a) = self.lstm_forward(a)
        b_out, (hn, cn_b) = self.lstm_forward(b)
        a_out_reverse, (hn, cn_a_reverse) = self.lstm_backward(a_reverse)
        b_out_reverse, (hn, cn_b_reverse) = self.lstm_backward(b_reverse)
        cn = torch.cat((cn_a, cn_a_reverse, cn_b, cn_b_reverse), dim=2)
        cn = cn.view(-1, 4*self.hidden_dim)
        cn = self.int_layer(cn)
        # relu activation
        cn = F.relu(cn)
        cn = self.out_layer(cn)
        return cn

def run_snli(model):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = {key: val.values for key, val in glove.T.items()}

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders
    dataloader_train = DataLoader(train_filtered, batch_size=32, shuffle=True)
    dataloader_valid = DataLoader(valid_filtered, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(test_filtered, batch_size=32, shuffle=True)

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    embedding_matrix = create_embedding_matrix(index_map, glove_embeddings, 100)
    # training code
    model = model(len(index_map), 100, 1, 3)
    model.embedding_layer.weight.data.copy_(embedding_matrix)
    
    # multilabel classification loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for sample in dataloader_train:
            premises = sample['premise']
            hypotheses = sample['hypothesis']            
            labels = sample['label']
            tokenized_premises = tokens_to_ix(tokenize(premises), index_map)
            tokenized_hypotheses = tokens_to_ix(tokenize(hypotheses), index_map)
            # use tokens_to_ix
            optimizer.zero_grad()
            outputs = model(tokenized_premises, tokenized_hypotheses)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print("Epoch: {}/{} Train Acc: {} Valid Acc: {}".format(epoch + 1, num_epochs, evaluate(model, dataloader_train, index_map), evaluate(model, dataloader_valid, index_map)))
    print("Test Acc: {}".format(evaluate(model, dataloader_test, index_map)))

def run_snli_lstm():
    model_class = UniLSTM
    run_snli(model_class)

def run_snli_bilstm():
    model_class = ShallowBiLSTM
    run_snli(model_class)

if __name__ == '__main__':
    run_char_rnn()
    run_char_lstm()
    run_snli_lstm()
    run_snli_bilstm()

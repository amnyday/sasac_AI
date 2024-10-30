import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

OOV = '[OOV]'
PAD = '[PAD]'

# hyperparameters
batch_size = 32

def pick_train_valid_test(train, valid, test):
    assert [train, valid, test] != [0, 0, 0]
    options = [train, valid, test]

    pick = random.choice([0, 1, 2])

    while options[pick] == 0:
        pick = random.choice([0, 1, 2])
    assert options[pick] != 0
    return pick

def split_train_valid_test(x, y, train_valid_test_ratio = (0.7, 0.15, 0.15)):
    # TensorDataset -> TensorDataset, TensorDataset, TensorDataset
    # x, y: list of data
    train_ratio, valid_ratio, test_ratio = train_valid_test_ratio
    y_label_dict = defaultdict(int)
    for y_data in y:
        y_label_dict[y_data.item()] += 1

    no_per_labels = {} # y_label별로 각각 train, valid, test

    for y_label, freq in y_label_dict.items():
        train_size, valid_size, test_size = int(freq * train_ratio), int(freq * valid_ratio), freq - int(freq * train_ratio) - int(freq * valid_ratio)
        no_per_labels[y_label] = [train_size, valid_size, test_size]

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []

    for x_data, y_data in zip(x, y):
        idx = pick_train_valid_test(*no_per_labels[y_data.item()])
        assert no_per_labels[y_data.item()][idx] > 0
        no_per_labels[y_data.item()][idx] -= 1

        if idx == 0:
            train_x.append(x_data)
            train_y.append(y_data)
        elif idx == 1:
            valid_x.append(x_data)
            valid_y.append(y_data)
        elif idx == 2:
            test_x.append(x_data)
            test_y.append(y_data)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def letter2tensor(letter, alphabets, oov = OOV):
    res = [0 for _ in range(len(alphabets))]

    if letter in alphabets:
        idx = alphabets.index(letter)
    else:
        idx = alphabets.index(oov)

    res[idx] = 1

    return torch.tensor(res)

def word2tensor(word, max_length, alphabets, pad = PAD, oov = OOV):
    # return torch.tensor with size (max_length, len(alphabets))
    res = torch.zeros(max_length, len(alphabets))

    for idx, char in enumerate(word):
        if idx < max_length:
            res[idx] = letter2tensor(char, alphabets, oov = oov)

    for i in range(max_length - len(word)):
        res[len(word) + i] = letter2tensor(pad, alphabets, oov = oov)

    return res

def determine_alphabets(data, pad = PAD, oov = OOV, threshold = 0.999):
    # data = list of [name, language_name]
    lst = []
    character_dict = defaultdict(int)

    for name, lang in data:
        for char in name:
            character_dict[char.lower()] += 1

    for k, v in character_dict.items():
        lst.append((k, v))

    lst = sorted(lst, key = lambda x:x[1], reverse = True)
    total_count = sum([e[1] for e in lst])
    s = 0

    alphabets = []

    for k, v in lst:
        s += v
        if s < threshold * total_count:
            alphabets.append(k)

    alphabets.append(pad)
    alphabets.append(oov)

    return alphabets

def determine_max_length(data, threshold = 0.99):
    lst = []
    name_length_dict = defaultdict(int)

    for name, lang in data:
         name_length_dict[len(name)] += 1

    for k, v in name_length_dict.items():
        lst.append((k, v))

    lst = sorted(lst, key = lambda x:x[1], reverse = True)
    total_count = sum([e[1] for e in lst])
    s = 0

    for k, v in lst:
        s += v
        if s > threshold * total_count:
            return k - 1
    # if not, return the maximum value in lst
    return max(lst, key = lambda x:x[0])[0]

def load_file():
    files = glob.glob('C:/Users/user/Desktop/중간실습/중간실습/names/*.txt')

    assert len(files) == 18

    data = []
    languages = []

    for file in files:
        with open(file) as f:
            names = f.read().strip().split('\n')
        lang = file.split('\\')[1].split('.')[0]

        if lang not in languages:
            languages.append(lang)

        for name in names:
            data.append([name, lang])

    return data, languages

def generate_dataset(batch_size = 32, pad = PAD, oov = OOV):
    data, languages = load_file()

    alphabets = determine_alphabets(data, pad = pad, oov = oov)
    max_length = determine_max_length(data)
    # print(alphabets, max_length)

    for idx, elem in enumerate(data):
        tmp = []
        for char in elem[0]:
            if char.lower() in alphabets:
                tmp.append(char.lower())
            else:
                tmp.append(oov)

        data[idx][0] = word2tensor(tmp, max_length, alphabets, pad = pad, oov = oov)
        data[idx][1] = languages.index(data[idx][1])

    x = torch.stack([e[0] for e in data])
    y = torch.stack([torch.tensor(e[1]) for e in data])

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    return dataloader, max_length, languages, alphabets

def tensor2word(t, alphabets):
    # t.shpae: max_length, len(alphabets)
    res = []
    for char_tensor in t:
        char = alphabets[int(torch.argmax(char_tensor).item())]
        res.append(char)

    return res

def idx2lang(idx, languages):
    return languages[idx]

# for batch_x, batch_y in dataset:
#     for i in range(batch_x.size(0)):
#         print(tensor2word(batch_x[i], alphabets), idx2lang(batch_y[i], languages))
#     break

import random
import pickle

def modify_dataset_for_ffn(dataset):
    x = []
    y = []

    for batch_x, batch_y in dataset:
        for i in range(batch_x.size(0)):
            x.append(batch_x[i].reshape((batch_x.size(1) * batch_x.size(2))))
            y.append(batch_y[i])

    train_x, train_y, valid_x, valid_y, test_x, test_y = split_train_valid_test(x, y)

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)
    valid_x = torch.stack(valid_x)
    valid_y = torch.stack(valid_y)
    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_dataloader, valid_dataloader, test_dataloader




def plot_loss_history(loss_history, other_loss_history = []):
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    if other_loss_history != []:
        plt.plot(range(1, len(other_loss_history) + 1), other_loss_history)
    plt.show()

# len(alphabets) * max_length * hidden_size + hidden_size * len(languages)
# 32 * 12 * 64 + 64 * 18 = 25000
#
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, alphabets, max_length):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(len(alphabets) * max_length, hidden_size)
        self.layer2 = nn.Linear(hidden_size, len(languages))

    def forward(self, x):
        # x: (batch_size, max_length, len(alphabets) : 32, 12, 57) -> (32, 12*57)
        x = x.view(x.size(0), -1)
        output = self.layer1(x)
        output = F.relu(output)
        output = self.layer2(output)
        output = F.log_softmax(output, dim = -1)

        return output # (batch_size, len(languages) : 32, 18)

    def train_model(self, train_data, valid_data, epochs = 100, learning_rate = 0.001, print_every = 1000):
        criterion = F.nll_loss
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        step = 0
        train_loss_history = []
        valid_loss_history = []

        train_log = {}

        for epoch in range(epochs):
            for x, y in train_data:
                step += 1
                y_pred = self(x)
                loss = criterion(y_pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mean_loss = torch.mean(loss).item()

                if step % print_every == 0 or step == 1:
                    train_loss_history.append(mean_loss)
                    valid_loss, valid_acc = self.evaluate(valid_data)
                    valid_loss_history.append(valid_loss)
                    print(f'[Epoch {epoch}, Step {step}] train loss: {mean_loss}, valid loss: {valid_loss}, valid_acc: {valid_acc}')
                    torch.save(self, f'checkpoints/feedforward_{step}.chkpts')
                    print(f'saved model to checkpoints/feedforward_{step}.chkpts')
                    train_log[f'checkpoints/feedforward_{step}.chkpts'] = [valid_loss, valid_acc]

        pickle.dump(train_log, open('checkpoints/train_log.dict', 'wb+'))

        return train_loss_history, valid_loss_history

    def evaluate(self, data):
        self.eval()
        criterion = F.nll_loss

        correct, total = 0, 0
        loss_list = []
        with torch.no_grad():
            for x, y in data:
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss_list.append(torch.mean(loss).item())
                correct += torch.sum((torch.argmax(y_pred, dim = 1) == y).float())
                total += y.size(0)
            return sum(loss_list) / len(loss_list), correct / total


dataset, max_length, languages, alphabets  = generate_dataset()
train_data, valid_data, test_data = modify_dataset_for_ffn(dataset)

model = FeedForwardNetwork(32, alphabets, max_length)
loss, acc = model.evaluate(train_data)

train_loss_history, valid_loss_history = model.train_model(train_data, valid_data)

plot_loss_history(train_loss_history, valid_loss_history)
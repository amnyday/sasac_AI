import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import random
import matplotlib.pyplot as plt

from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

OOV = '[OOV]'
PAD = '[PAD]'

# hyperparameters

def plot_history_loss(history_loss, other_history_loss = []):
    plt.plot(range(1, len(history_loss) +1), history_loss)
    if other_history_loss != []:
        plt.plot(range(1, len(other_history_loss) +1), other_history_loss)
    plt.show()

def charTotensor(char, alpha, oot = OOV):
    res = [0 for _ in range(len(alpha))]
    
    if char in alpha:
        idx = alpha.index(char)
    else:
        idx = alpha.index(oot)
        
    res[idx] = 1
    
    return torch.tensor(res)
    #알파벳글자수만큼 0부터26까지 res에 받아서 리스트로만들어놓고 sen으로받은문장이 alpha에 있으면 그 sen문장의인덱스를 알파벳에서 찾고
    #없으면 알파벳에서 oov의 인덱스를 찾고 res의 인덱스를1로한다. #문장을 받아서 알파벳에있으면 그알파벳인덱스위치에 1이들어간 텐서를 반환한다.
    
def wdTotensor(wd, max_len, alpha, pad = PAD, oov = OOV):
    wd_tensor = torch.zeros(max_len, len(alpha))
    
    for idx, char in enumerate(wd):
        if idx < max_len:
            wd_tensor[idx] = charTotensor(char, alpha, oov = oov)
        if idx < max_len:
            wd_tensor[idx] = charTotensor(char, alpha, oov=oov)
    for i in range(max_len - len(wd)):
        wd_tensor[len(wd) + i] = charTotensor(pad, alpha, oov = oov)
        
    return wd_tensor
    #알파벳글자만큼 제로벡터만들어놓고 단어인덱스랑 한글자씩 받아서 최대글자수보다 인덱스가 작으면         

def determine_alpha(data, pad = PAD, oov = OOV, threshold = 0.999):
    # data = list of [name, language_name]
    lst = []
    lst = []
    char_dict = defaultdict(int)
    
    for name, lang in data:
        for char in name:
            char_dict[char.lower()] += 1
    
    for k, v in char_dict.items():
        lst.append((k,v))

    lst = sorted(lst, key = lambda x:x[1], reverse = True)
    total_cnt = sum([e[1] for e in lst])
    sum = 0
    
    alpha = [] 

    for k, v in lst:
        sum += v
        if sum < threshold * total_cnt:
            alpha.append(k)
            
    alpha.append(pad)
    alpha.append(oov)

    return alpha

def determine_max_len(dt, threshold = 0.99):
    lst =[]
    name_len_dict = defaultdict(int)
    
    for name, contry in dt:
        name_len_dict[len(name)] +=1
            
    for k,v in name_len_dict.items():
        lst.append((k,v))
        
    lst = sorted(lst, key = lambda x:x[1], reverse = True)
    total_cnt = sum([e[1] for e in lst])
    s = 0
    
    for k, v in lst:
        s += v
        if s > threshold * total_cnt:
            return k - 1
    return max(lst, key = lambda x:x[0])[0]
    
def load_file():
    files = glob.glob('C:/Users/user/Desktop/중간실습/중간실습/names/*.txt')
    assert len(files) == 18
    
    dt = []
    contries = []
    
    for file in files:
        with open(file) as f:
            names = f.read().strip().split('\n')
        contry = file.split('\\')[1].split('.')[0]
        
        if contry not in contries:
            contries.append(contry)
            
        for name in names:
            dt.append([name, contry])
    
    return dt, contries

def split_train_valid_test(x, y, train_valid_test_rt = (0.7, 0.15, 0.15)):
    train_rt, valid_rt, test_rt = train_valid_test_rt
    y_label_dict = defaultdict(int)
    
    for y_dt in y:
        y


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



def generate_dt(batch_size = 32, pad = PAD, oov = OOV):
    dt, langs = load_file()
    
    alpha = determine_alpha(dt, pad= pad, oov=oov)
    max_len = determine_max_len(dt)
    
    for idx, elem in enumerate(dt):
        tmp = []
        for char in elem[0]:
            if char.lower() in alpha:
                tmp.append(char.lower())
            else:
                tmp.append(oov)
                
        dt[idx][0] = wdTotensor(tmp, max_len, alpha, pad = pad, oov = oov)
        dt[idx][1] = langs.index(dt[idx][1])
        

    x = [e[0] for e in dt]
    y = [torch.tensor(e[1]) for e in dt]

    train_x, train_y, valid_x, valid_y, test_x, test_y = split_
    
    
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

    return train_dataloader, valid_dataloader, test_dataloader, alphabets, max_length, languages


'''
def plot_loss_history(loss_history, other_loss_history = []):    
def letter2tensor(letter, alphabets, oov = OOV):
def word2tensor(word, max_length, alphabets, pad = PAD, oov = OOV):
def determine_alphabets(data, pad = PAD, oov = OOV, threshold = 0.999):
def determine_max_length(data, threshold = 0.99):

def load_file():

def generate_dataset(batch_size = 32, pad = PAD, oov = OOV):

def split_train_valid_test(x, y, train_valid_test_ratio = (0.7, 0.15, 0.15)):

def pick_train_valid_test(train, valid, test):

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, batch_first = True):

    def forward(self, x, hidden):

    def init_hidden(self):

    def train_model(self, train_data, valid_data, epochs = 100, learning_rate = 0.001, print_every = 1000):

    def evaluate(self, data):


rnn = RecurrentNeuralNetwork(128)
train_loss_history, valid_loss_history = rnn.train_model(train_dataset, valid_dataset)

plot_loss_history(train_loss_history, valid_loss_history)
'''
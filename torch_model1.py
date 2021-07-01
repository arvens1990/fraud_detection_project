import numpy as np
import pandas as pd
import pickle

import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 13326

EMBED_DIM = 100
# LEARNING_RATE = 0.000001
LEARNING_RATE = 0.000001
MOMENTUM = 0.05
WEIGHT_DECAY = 0.05
POOLING_MODE = "max"

EPOCHS = 150

class MIL(nn.Module):
    
    def __init__(self, input_size, pooling_mode="max", bias=True):
        super(MIL, self).__init__()

        self.pooling_mode = pooling_mode
        self.input_size = input_size
        
        # layers
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(.5)
        
        # pooling
        self.kernel = nn.Parameter(torch.randn((64,1)), requires_grad=True)
        self.kernel_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)  if bias else False
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = len(x)
        embedding = self.embed(torch.LongTensor(x[:,-19:-1]))
        embedding = embedding.view(batch_size, 18*EMBED_DIM)
        features = torch.tensor(x[:,:-19])
        x = torch.cat((features, embedding), dim=1)
        
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = x.matmul(self.kernel)
        if self.kernel_bias:
            x = x + self.kernel_bias
        x = self.sigmoid(x)

        if self.pooling_mode=="LSE":
            out = torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))[0]
            return out
        elif self.pooling_mode=="mean":
            out = torch.mean(x, dim=0, keepdim=True)[0]
            return out
        else:
            out = torch.max(x, dim=0, keepdim=True)[0]
            return out

def bag_loss(predicted, truth):
    loss = nn.BCELoss()
    truth = torch.tensor(truth).max().float()
    out = loss(predicted.squeeze(), truth)
    return out

def train_loop(X, mil, optimizer):
    # Setup
    mil.train()
    optimizer.zero_grad()
    
    # Metrics
    epoch_loss = 0
    epoch_con_mat = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0
        }
    
    # Loop over bags:
    for ibag, bag in enumerate(X):
        # forward
        y_pred = mil(bag)
        # loss
        loss = bag_loss(y_pred, y[ibag])
        # back-propogation
        loss.backward()
        optimizer.step()
        
        # tracking
        epoch_loss += loss.item()
        
        y_ibag = max(y[ibag])
        y_pred_item = round(y_pred.item()) # remember to look at this threshold as a hyperparam

        if y_pred_item == y_ibag:
            if y_ibag == 1:
                epoch_con_mat["tp"] +=1
            else:
                epoch_con_mat["tn"] +=1
        else:
            if y_ibag == 1:
                epoch_con_mat["fn"] +=1
            else:
                epoch_con_mat["fp"] +=1
    
    mean_loss = epoch_loss/len(X)
    accuracy = (epoch_con_mat["tp"]+epoch_con_mat["tn"])/(epoch_con_mat["tp"]+epoch_con_mat["fn"]+epoch_con_mat["tn"]+epoch_con_mat["fp"])
    pos_rate = epoch_con_mat["tp"]/(epoch_con_mat["tp"]+epoch_con_mat["fn"])

    return mean_loss, accuracy, pos_rate, epoch_con_mat

def valid_loop(X_test, mil):
    # Setup
    mil.eval()

    # Metrics
    epoch_v_loss = 0
    epoch_v_con_mat = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0
        }
    
    # Loop over bags:
    for ibag, bag in enumerate(X_test):
        # forward
        y_pred = mil(bag)
        # loss
        val_loss = bag_loss(y_pred, y_test[ibag])
        
        # tracking
        epoch_v_loss += val_loss.item()
        
        y_ibag = max(y[ibag])
        y_pred_item = round(y_pred.item())

        if y_pred_item == y_ibag:
            if y_ibag == 1:
                epoch_v_con_mat["tp"] +=1
            else:
                epoch_v_con_mat["tn"] +=1
        else:
            if y_ibag == 1:
                epoch_v_con_mat["fn"] +=1
            else:
                epoch_v_con_mat["fp"] +=1
    
    v_mean_loss = epoch_v_loss / len(X_test)
    v_accuracy = (epoch_v_con_mat["tp"]+epoch_v_con_mat["tn"])/(epoch_v_con_mat["tp"]+epoch_v_con_mat["fn"]+epoch_v_con_mat["tn"]+epoch_v_con_mat["fp"])
    v_pos_rate = epoch_v_con_mat["tp"]/(epoch_v_con_mat["tp"]+epoch_v_con_mat["fn"])
    
    return v_mean_loss, v_accuracy, v_pos_rate, epoch_v_con_mat

# RUN SCRIPT --------------------------------------------------------------------------------------

# LOAD DATA

load = "./datasets_train_val.pkl"
datasets = pickle.load(open(load, "rb"))

X = []
y = []

print("Preparing train data...")
for i, tup in enumerate(datasets[0]['train']):
    X.append(tup[0])
    y.append(tup[1])

print("Train data ready!")

X_test = []
y_test = []

print("Preparing test data...")
for i, tup in enumerate(datasets[0]['test']):
    X_test.append(tup[0])
    y_test.append(tup[1])

print("Test data ready!")

del datasets

shape = X[0].shape
print('First train datapoint shape: ', shape)

# LOAD MODEL

mil = MIL(input_size=shape[1]-19+18*EMBED_DIM, pooling_mode=POOLING_MODE, bias=True)
optimizer = torch.optim.SGD(mil.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

losses = []
accuracies = []
pos_rates = []
con_mats = []

val_losses = []
val_accuracies = []
val_pos_rates = []
val_con_mats = []

for epoch in range(EPOCHS+1):
    # # TRAIN ---------------------------------------------
    
    # # setup
    # mil.train()
    # optimizer.zero_grad()
    
    # # epoch metrics
    # epoch_loss = 0
    # epoch_con_mat = {
    #         "tp": 0,
    #         "tn": 0,
    #         "fp": 0,
    #         "fn": 0
    #     }
    
    # # loop over bags:
    # for ibag, bag in enumerate(X):
    #     # forward
    #     y_pred = mil(bag)
    #     # loss
    #     loss = bag_loss(y_pred, y[ibag])
    #     # back-propogation
    #     loss.backward()
    #     optimizer.step()
        
    #     # tracking
    #     epoch_loss += loss.item()
        
    #     y_ibag = max(y[ibag])
    #     y_pred_item = round(y_pred.item())

    #     if y_pred_item == y_ibag:
    #         if y_ibag == 1:
    #             epoch_con_mat["tp"] +=1
    #         else:
    #             epoch_con_mat["tn"] +=1
    #     else:
    #         if y_ibag == 1:
    #             epoch_con_mat["fn"] +=1
    #         else:
    #             epoch_con_mat["fp"] +=1
    
    # epoch_pos_rate = epoch_con_mat["tp"] / (epoch_con_mat["tp"]+epoch_con_mat["fn"])
    # epoch_accuracy = (epoch_con_mat["tp"]+epoch_con_mat["tn"]) / (epoch_con_mat["tp"]+epoch_con_mat["fn"]+epoch_con_mat["tn"]+epoch_con_mat["fp"])
    # epoch_loss = epoch_loss / len(X)

    mean_loss, accuracy, pos_rate, con_mat = train_loop(X, mil, optimizer)

    losses.append(mean_loss)
    pos_rates.append(pos_rate)
    accuracies.append(accuracy)
    con_mats.append(con_mat)

    # VALID ---------------------------------------------
    
    # #setup
    # mil.eval()

    # # epoch metrics
    # epoch_v_loss = 0
    # epoch_v_con_mat = {
    #         "tp": 0,
    #         "tn": 0,
    #         "fp": 0,
    #         "fn": 0
    #     }
    
    # # loop over bags:
    # for ibag, bag in enumerate(X_test):
    #     # forward
    #     y_pred = mil(bag)
    #     # loss
    #     val_loss = bag_loss(y_pred, y_test[ibag])
    #     # tracking
    #     epoch_v_loss += val_loss.item()
    #     y_ibag = max(y[ibag])
    #     y_pred_item = round(y_pred.item())

    #     if y_pred_item == y_ibag:
    #         if y_ibag == 1:
    #             epoch_v_con_mat["tp"] +=1
    #         else:
    #             epoch_v_con_mat["tn"] +=1
    #     else:
    #         if y_ibag == 1:
    #             epoch_v_con_mat["fn"] +=1
    #         else:
    #             epoch_v_con_mat["fp"] +=1
    
    # epoch_v_pos_rate = epoch_v_con_mat["tp"]/(epoch_v_con_mat["tp"]+epoch_v_con_mat["fn"])
    # epoch_v_accuracy = (epoch_v_con_mat["tp"]+epoch_v_con_mat["tn"])/(epoch_v_con_mat["tp"]+epoch_v_con_mat["fn"]+epoch_v_con_mat["tn"]+epoch_v_con_mat["fp"])
    
    v_mean_loss, v_accuracy, v_pos_rate, epoch_v_con_mat = valid_loop(X_test, mil)

    # val_pos_rates.append(epoch_v_pos_rate)
    # val_accuracies.append(epoch_v_accuracy)
    # val_con_mats.append(epoch_v_con_mat)

    # epoch_v_loss = epoch_v_loss / len(X_test)
    # val_losses.append(epoch_v_loss)

    print(f"Epoch {epoch}\{EPOCHS} \t Train Loss: {mean_loss:.4f};\tTrain Accuracy: {accuracy:.4f};\tTrain Posit: {pos_rate:.4f};\tTest Loss: {v_mean_loss:.4f}\tTest Accuracy: {v_accuracy:.4f};\tTest Posit: {v_pos_rate:.4f};")

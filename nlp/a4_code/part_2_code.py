
import random
from statistics import mode
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
import logging
import pandas as pd 
import os
import argparse
import logging
        
# ######################## PART 1: PROVIDED CODE ########################

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='distilbert-base-uncased')
args = parser.parse_args()

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

        
def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.
    
    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        dd = data_dict

        if len(dd["premise"]) != len(dd["hypothesis"]) or len(dd["premise"]) != len(
            dd["label"]
        ):
            raise AttributeError("Incorrect length in data_dict")

    def __len__(self):
        return len(self.data_dict["premise"])

    def __getitem__(self, idx):
        dd = self.data_dict
        return dd["premise"][idx], dd["hypothesis"][idx], dd["label"][idx]


def train_distilbert(model, loader, optimizer, sp, device):
    model.train()
    criterion = model.get_criterion()
    total_loss = 0.0

    for premise, hypothesis, target in tqdm(loader):
        optimizer.zero_grad()

        inputs = model.tokenize(premise, hypothesis).to(device)
        
        input_ids = inputs.pop('input_ids')  # Remove input_ids from batch
        embeds = model.get_distilbert().embeddings(input_ids)
        inputs['inputs_embeds'] = sp(embeds)
        inputs['attention_mask'] = pad_attention_mask(inputs.attention_mask, 5)

        if args.model == 'bert-base-uncased':
            inputs['token_type_ids'] = torch.cat([torch.zeros(inputs.token_type_ids.shape[0], 5).long().to(device), inputs.token_type_ids], dim=1)

        target = target.to(device, dtype=torch.float32)

        pred = model(inputs)


        loss = criterion(pred, target)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_distilbert(model, loader, device):
    model.eval()

    targets = []
    preds = []
    criterion = model.get_criterion()
    total_loss = 0.0

    for premise, hypothesis, target in loader:
        inputs = model.tokenize(premise, hypothesis).to(device)
        input_ids = inputs.pop('input_ids')
        embeds = model.get_distilbert().embeddings(input_ids)
        inputs['inputs_embeds'] = sp(embeds)
        inputs['attention_mask'] = pad_attention_mask(inputs.attention_mask, 5)
        
        if args.model == 'bert-base-uncased':
            inputs['token_type_ids'] = torch.cat([torch.zeros(inputs.token_type_ids.shape[0], 5).long().to(device), inputs.token_type_ids], dim=1)

        target = target.to(device, dtype=torch.float32)
        pred = model(inputs)
        loss = criterion(pred, target)
        preds.append(pred)
        targets.append(target)
        total_loss += loss.item()

    return torch.cat(preds), torch.cat(targets), total_loss / len(loader)


# ######################## PART 1: YOUR WORK STARTS HERE ########################
class CustomDistilBert(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: your work below
        self.distilbert = transformers.AutoModel.from_pretrained(args.model)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        
        if args.model == 'distilbert-base-uncased':
            print("Using Dim " + args.model)
            self.pred_layer = nn.Linear(self.distilbert.config.dim, 1)
            
        else:
            print("Using hiddent_size " + args.model)
            self.pred_layer = nn.Linear(self.distilbert.config.hidden_size, 1)
            
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
    
    # vvvvv DO NOT CHANGE BELOW THIS LINE vvvvv
    def get_distilbert(self):
        return self.distilbert

    def get_tokenizer(self):
        return self.tokenizer

    def get_pred_layer(self):
        return self.pred_layer

    def get_sigmoid(self):
        return self.sigmoid
    
    def get_criterion(self):
        return self.criterion
    # ^^^^^ DO NOT CHANGE ABOVE THIS LINE ^^^^^

    def assign_optimizer(self, **kwargs):
        # TODO: your work below
        optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        self.optimizer = optimizer

        return optimizer


    def slice_cls_hidden_state(
        self, x: transformers.modeling_outputs.BaseModelOutput
    ) -> torch.Tensor:
        x = x.last_hidden_state
        x = x[:, 0, :]
        return x
        

    def tokenize(
        self,
        premise: "list[str]",
        hypothesis: "list[str]",
        max_length: int = 128,
        truncation: bool = True,
        padding: bool = True,
    ):
    # A dictionary-like object that can be given to the model to make predictions.
        return self.tokenizer(premise, hypothesis, max_length=max_length, truncation=truncation, padding=padding, return_tensors="pt")

    def forward(self, inputs: transformers.BatchEncoding):
        inputs = self.distilbert(**inputs)
        inputs = self.slice_cls_hidden_state(inputs)
        inputs = self.pred_layer(inputs)
        output = self.sigmoid(inputs)
        return output.squeeze()
    
# ######################## PART 2: YOUR WORK HERE ########################
def freeze_params(model):
    model.requires_grad_(False)
    return model


def pad_attention_mask(mask, p):
    mask = torch.cat([torch.ones(mask.shape[0], p).to(device), mask], dim=1)
    return mask
    
class SoftPrompting(nn.Module):
    def __init__(self, p: int, e: int):
        super().__init__()
        self.p = p
        self.e = e
        
        self.prompts = torch.randn((p, e), requires_grad=True, device= device )
        
    def forward(self, embedded):
        prompts_embedded = self.prompts.unsqueeze(0).expand(embedded.shape[0], -1, -1)
        embedded = torch.cat([prompts_embedded, embedded], dim=1)
        return embedded

def combine_plot_validation_acc(models, acc_list):
    # all model should have different colors
    colors = ['r', 'g', 'b']
    for i in range(len(models)):
        plt.plot(acc_list[i], label = models[i], color = colors[i])
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.legend()
    if not os.path.exists('plots on part 2'):
        os.makedirs('plots on part 2')
    plt.savefig("plots on part 2/accuracy.png")
    plt.show()

def combine_plot_validation_loss(models, losses_list):
    # all model should have different colors
    colors = ['r', 'g', 'b']
    for i in range(len(models)):
        plt.plot(losses_list[i], label = models[i], color = colors[i])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    if not os.path.exists('plots on part 2'):
        os.makedirs('plots on part 2')
    plt.savefig("plots on part 2/losses.png")
    plt.show()
    
# ######################## PART 4: YOUR WORK HERE ########################


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score  # Make sure sklearn is installed
    from sklearn.metrics import accuracy_score
    random.seed(2022)
    torch.manual_seed(2022)
    logging_dir = "logs/"
    
    set_logger(os.path.join(logging_dir, args.model+' part_two.log'))
    logging.info("Logging info about part-2 below")
    
    # Parameters (you can change them)
    sample_size = None  # Change this if you want to take a subset of data for testing
    batch_size = 64
    n_epochs = 10
    num_words = 50000

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################### PART 2: TEST CODE ######################
    # Prefilled code showing you how to use the helper functions
    
    logging.info("Train dataset stats:")
    train_raw, valid_raw = load_datasets("data/nli")
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    print("=" * 80)
    logging.info("Running code for part-2 on " + args.model)
    print("-" * 80)

    train_loader = torch.utils.data.DataLoader(
        NLIDataset(train_raw), batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        NLIDataset(valid_raw), batch_size=batch_size, shuffle=False
    )

    model = CustomDistilBert().to(device)
    scores_trains = []
    scores_valids = []
    losses_trains = []
    losses_valids = []
    
    # ###################### PART 2: TEST CODE ######################
    freeze_params(model.get_distilbert()) # Now, model should have no trainable parameters

    # Create the softprompt module and the optimizer
    sp = SoftPrompting(p=5, e=model.get_distilbert().embeddings.word_embeddings.embedding_dim)
    def sp_optimizer(sp_params, lr=0.01, **kwargs):
        return torch.optim.Adam(sp_params, **kwargs)
    optimizer = sp_optimizer([sp.prompts])

    for epoch in range(n_epochs):
        loss = train_distilbert(model, train_loader, optimizer, sp, device = device)
        
        logging.info(f"Epoch {epoch} loss: {loss}")
        preds_train, targets_train, train_loss = eval_distilbert(model, train_loader, device = device)
        
        # evaluation on training set
        preds_train = preds_train.round()
        scores_train = accuracy_score(targets_train.cpu(), preds_train.cpu())
        scores_trains.append(scores_train)
        f1_train = f1_score(targets_train.cpu(), preds_train.cpu())
        losses_trains.append(train_loss)
        logging.info(f"Epoch {epoch} valid loss on training set: {train_loss}")
        logging.info(f"Epoch {epoch} f1 score on training set: {f1_train}")
        logging.info(f"Epoch {epoch} accuracy score on training set: {scores_train}")
        
        logging.info("\n")
     
        # evaluation on test set
        preds_valid, targets_valid, loss_vailds = eval_distilbert(model, valid_loader, device = device)
        preds_valid = preds_valid.round()
        scores_valid = accuracy_score(targets_valid.cpu(), preds_valid.cpu())
        scores_valids.append(scores_valid)
        f1_valid = f1_score(targets_valid.cpu(), preds_valid.cpu())
        
        losses_valids.append(loss_vailds)
        logging.info(f"Epoch {epoch} valid loss on validation set: {loss_vailds}")
        logging.info(f"Epoch {epoch} f1 score on validation set: {f1_valid}")
        logging.info(f"Epoch {epoch} accuracy score on validation set: {scores_valid}")
        
        # model saved
        logging.info("\n")
        logging.info("\n")
        logging.info("=" * 80)
    
    torch.save(model.state_dict(), args.model+' part_two_model.pt')
    logging.info("model is saved")



    # plot of the validation loss, and another of the validation accuracy over each of the 10 epochs with labels on different curves.
    # store the png in new folder called "plots on part 2"

    # check if the folder exists
    if not os.path.exists('plots on part 2'):
        os.makedirs('plots on part 2')

    plt.plot(losses_valids, label = "validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("plots on part 2/" + args.model + " losses.png")
    plt.show()

    plt.figure()
    plt.plot(scores_valids, label = "validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("plots on part 2/" + args.model + " accuracy.png")
    plt.show()

    # combine plot is constructed by reading the value from .log files of each model
    # Then calling combine_plot_validation_loss and combine_plot_validation_acc
    
    # losses_valids_bert = [0.6745318145706103, 0.6577232606135882, 0.6413594776621232, 0.6206689826571025,0.6014330289684809, 0.5871490647013371,0.5807616756512568, 0.5729822075137725,0.5643889236335571,0.5563389607346975 ]
    # losses_valids_roberta = [0.6888789958678759, 0.6858643837846242, 0.6772632111723607, 0.6655714574914712, 0.6561283377500681, 0.6417442021461633, 0.6322256831022409, 0.6194633142306254, 0.603768602013588, 0.5677209834639843]
    #losses_valids_distilbert = [0.6097664145322946, 0.5291866886501129, 0.47983859422115177, 0.4522954231271377, 0.42887316988064694, 0.42944545298814774, 0.40823584451125217, 0.3946322423334305, 0.390741795158157, 0.3852119008795573]
    # plot_validation_loss(["bert-base-uncased", "roberta-base", "distilbert-base-uncased"], [losses_valids_bert, losses_valids_roberta, losses_valids_distilbert])
    
    ## Ploting accuracy curve
    #roberta_acc = [0.5569850158922355, 0.5569850158922355, 0.5866505221734524, 0.6138943544725292, 0.6193431209323444, 0.6367489026789769, 0.6362948388073255, 0.6546087482972605, 0.68639321931285, 0.7454215226275163]
    #berta_acc = [0.5970939912214318, 0.6214620856667171, 0.6443166338731648, 0.6644467988497048, 0.6821552898441048, 0.6935068866353867, 0.6925987588920841, 0.7007719085818072, 0.70652338428939, 0.7140911154835781]
    #distil_bert_acc = [0.6865445739367337, 0.7478431966096565, 0.7781141213864083, 0.7979415771151809, 0.811109429393068, 0.8102013016497654, 0.8235205085515362, 0.8327531406084456, 0.8371424247010746, 0.8386559709399122 ]
    #plot_validation_acc(["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],[berta_acc, roberta_acc, distil_bert_acc])
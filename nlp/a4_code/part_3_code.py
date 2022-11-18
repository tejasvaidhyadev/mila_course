
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
from torch.utils.data import TensorDataset, DataLoader 
import os 
import logging

# ######################## PART 1: PROVIDED CODE ########################
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


def train_distilbert(model, loader, device):
    model.train()
    criterion = model.get_criterion()
    total_loss = 0.0

    for premise, hypothesis, target in tqdm(loader):
        optimizer.zero_grad()

        inputs = model.tokenize(premise, hypothesis).to(device)
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

    for premise, hypothesis, target in loader:
        preds.append(model(model.tokenize(premise, hypothesis).to(device)))

        targets.append(target)

    return torch.cat(preds), torch.cat(targets)


# ######################## PART 1: YOUR WORK STARTS HERE ########################
class CustomDistilBert(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: your work below
        self.distilbert = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.pred_layer = nn.Linear(self.distilbert.config.dim, 1)
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
    mask = torch.cat([torch.ones(mask.shape[0], p), mask], dim=1)
    return mask
    
class SoftPrompting(nn.Module):
    def __init__(self, p: int, e: int):
        super().__init__()
        self.p = p
        self.e = e
        
        self.prompts = torch.randn((p, e), requires_grad=True)
        
    def forward(self, embedded):
        prompts_embedded = self.prompts.unsqueeze(0).expand(embedded.shape[0], -1, -1)
        embedded = torch.cat([prompts_embedded, embedded], dim=1)
        return embedded
        ################## PART 3: YOUR WORK HERE ########################

def load_models_and_tokenizer(q_name, a_name, t_name, device='cpu'):
    q_enc = transformers.AutoModel.from_pretrained(q_name).to(device)
    a_enc = transformers.AutoModel.from_pretrained(a_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(t_name)
    return q_enc, a_enc, tokenizer

def tokenize_qa_batch(tokenizer, q_titles, q_bodies, answers, max_length=64) -> transformers.BatchEncoding:
    q_batch = tokenizer(q_titles, q_bodies, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    a_batch = tokenizer(answers, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    return q_batch, a_batch

def get_class_output(input_ids, attention_mask, token_type_ids, model):
    inputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    inputs = inputs.last_hidden_state
    inputs = inputs[:, 0, :]
    return inputs    

def inbatch_negative_sampling(Q: Tensor, P: Tensor, device: str = 'cpu') -> Tensor:
    sim = torch.matmul(Q, torch.transpose(P, 0, 1))
    # to device
    sim = sim.to(device)
    return sim

def contrastive_loss_criterion(S: Tensor, labels: Tensor = None, device: str = 'cpu'):
    # S is the similarity matrix from inbatch_negative_sampling
    score = S.view(S.shape[0], -1)
    if labels is None:
        # simply return a tensor with values such at Passage #0 corresponds to Question #0, P1 with Q1, etc.
        labels = torch.arange(S.shape[0]).to(device)
    labels = labels.view(-1)
    softmax_score = F.log_softmax(score, dim=1)
    loss = F.nll_loss(softmax_score, labels, reduction='mean')
    return loss
    

def get_topk_indices(Q, P, k: int = None):

    # dot-product similarity
    scores = torch.matmul(Q, P.T)
    if k is None:
        k = scores.shape[1]
    topk = torch.topk(scores, k=k, dim=1)
    # return indices of top-k and scores
    return topk.indices, topk.values
    
def select_by_indices(indices: Tensor, passages: 'list[str]') -> 'list[str]':
    selected = []
    for i in indices:
        selected.append([passages[j] for j in i])
    return selected


def embed_passages(passages: 'list[str]', model, tokenizer, device='cuda', max_length=512):
    passages = tokenizer(passages, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    passages = model(**passages.to(device))
    passages = passages.last_hidden_state
    # get the CLS token
    passages = passages[:, 0, :]
    return passages

def embed_questions(titles, bodies, model, tokenizer, device='cuda', max_length=512):
    questions = tokenizer(titles, bodies, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    questions = model(**questions.to(device))
    questions = questions.last_hidden_state
    # get the CLS token
    questions = questions[:, 0, :]
    return questions

def recall_at_k(retrieved_indices: 'list[list[int]]', true_indices: 'list[int]', k: int):
    recall = []
    for i, j in zip(retrieved_indices, true_indices):
        if j in i[:k]:
            recall.append(1)
        else:
            recall.append(0)
    # no np.mean 
    return sum(recall) / len(recall) 


def mean_reciprocal_rank(retrieved_indices: 'list[list[int]]', true_indices: 'list[int]'):

    mrr = []
    for i, j in zip(retrieved_indices, true_indices):
        if j in i:
            mrr.append(1 / (i.index(j) + 1))
        else:
            mrr.append(0)
    return sum(mrr) / len(mrr)


# ######################## PART 4: YOUR WORK HERE ########################
def get_dataloader(qa_data, tokenizer,  batch_size=64, shuffle= True):
    q_titles = qa_data.loc[:, 'QuestionTitle'].tolist()
    q_bodies = qa_data.loc[:, 'QuestionBody'].tolist()
    answers = qa_data.loc[:, 'Answer'].tolist()
    logging.info("Tokenization of whole dataset is completed")
    q_batch, a_batch = tokenize_qa_batch(tokenizer, q_titles, q_bodies, answers)
    # include token_type_ids
    dataset = TensorDataset(q_batch.input_ids, q_batch.attention_mask, q_batch.token_type_ids, a_batch.input_ids, a_batch.attention_mask, a_batch.token_type_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_true_indices(topk_indices, qa_data):
    true_indices = []
    for indices in topk_indices:
        true_indices.append(qa_data.loc[indices.cpu().numpy(), 'AnswerId'].tolist())
    return true_indices

def evalulation_loop(q_enc, a_enc, qa_data, k ,  device='cuda'):
    with torch.no_grad():
        q_titles = qa_data.loc[:, 'QuestionTitle'].tolist()
        q_bodies = qa_data.loc[:, 'QuestionBody'].tolist()
        passages = qa_data.loc[:, 'Answer'].tolist()
        true_indices = qa_data.loc[:, 'AnswerId'].tolist()
        P = embed_passages(passages, model=a_enc, tokenizer=tokenizer, device = device, max_length=128)
        Q = embed_questions(q_titles, q_bodies, model=q_enc, tokenizer=tokenizer, device = device, max_length=128)
        topk_indices, topk_scores = get_topk_indices(Q, P, k)
        topk_true_indices = get_true_indices(topk_indices, qa_data)
        recall, mrr = recall_at_k(topk_true_indices, true_indices, k), mean_reciprocal_rank(topk_true_indices, true_indices)
    return recall, mrr

                
if __name__ == "__main__":
    random.seed(2022)
    torch.manual_seed(2022)
    import pandas as pd 
    logging_dir = "logs/"
    
    set_logger(os.path.join(logging_dir, ' part_three.log'))
    logging.info("Logging info about part-3 below")

    # Parameters (you can change them)
    n_epochs = 40

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)
    # ###################### PART 3: TEST CODE ######################
    # Preliminary
    bsize = 128
    qa_data = dict(
        train = pd.read_csv('data/qa/train.csv'),
        valid = pd.read_csv('data/qa/validation.csv'),
        answers = pd.read_csv('data/qa/answers.csv'),
    )

    # Loading huggingface models and tokenizers    
    name = 'google/electra-small-discriminator'
    q_enc, a_enc, tokenizer = load_models_and_tokenizer(q_name=name, a_name=name, t_name=name)

    train_dataloader = get_dataloader(qa_data['train'], tokenizer, batch_size=bsize, shuffle=True)
    # valid_dataloader = get_dataloader(qa_data['valid'], tokenizer, batch_size=bsize, shuffle=False)
    logging.info("dataloader is created for training")
    
    optimizer = torch.optim.Adam(list(q_enc.parameters()) + list(a_enc.parameters()), lr=1e-5)
    recalls_train = []
    mrrs_train = []
    recalls_valid = []
    mrrs_valid = []

    # training loop
    for epoch in range(n_epochs):
        for batch in tqdm(train_dataloader):
            # unpack batch
            q_input_ids, q_attention_mask, q_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids = batch
            # forward pass
            q_enc.to(device)
            a_enc.to(device)

            # get class outputs
            q_enc.train()
            a_enc.train()
            
            q_enc.zero_grad()
            a_enc.zero_grad()
            
            q_out = get_class_output(q_input_ids.to(device), q_attention_mask.to(device), q_token_type_ids.to(device), q_enc)
            a_out = get_class_output(a_input_ids.to(device), a_attention_mask.to(device), a_token_type_ids.to(device), a_enc)

            # in-batch negative sampling
            S = inbatch_negative_sampling(q_out, a_out)
            loss = contrastive_loss_criterion(S)
            loss.backward()
            optimizer.step()
            
        logging.info(f'Epoch {epoch} Loss {loss.item()}')
        logging.info("running on eval set")
        
        q_enc.eval()
        a_enc.eval()

        # print epoch number
        logging.info(f'Epoch {epoch}')
        recall_train, mrr_train = evalulation_loop(q_enc, a_enc, qa_data['train'], 20,  device=device)
        recalls_train.append(recall_train)
        mrrs_train.append(mrr_train)
        logging.info(f'Train Recall@20: {recall_train}, MRR: {mrr_train}')

        recall_valid, mrr_valid = evalulation_loop(q_enc, a_enc, qa_data['valid'], 20, device=device)
        recalls_valid.append(recall_valid)
        mrrs_valid.append(mrr_valid)
        logging.info(f'Valid Recall@20: {recall_valid}, MRR: {mrr_valid}')
        import matplotlib.pyplot as plt
    # plot recalls_train and  recalls_valid 
    # Mark X-axis: epoch, Y-axis: recall
    plt.plot(recalls_train, label='train')
    plt.plot(recalls_valid, label='valid')
    # x-axis as epoch
    plt.xlabel('epoch')
    # y-axis as recall
    plt.ylabel('recall@20')
    plt.legend()
    plt.show()
    plt.savefig('recalls.png')

                
    sample = qa_data['valid'].sample(10)
    # get question title, body, true answer
    q_titles = sample['QuestionTitle'].tolist()
    q_bodies = sample['QuestionBody'].tolist()
    true_answers = sample['Answer'].tolist()
    # get true indices
    passages = qa_data['answers']['Answer'].dropna().tolist()

    scores_list = []
    batch_size_passages = 128
    # batching of passages to avoid out of memory error
    Q = embed_questions(q_titles, q_bodies, model=q_enc, tokenizer=tokenizer, device = device, max_length=128)
    for i in tqdm(range(0, len(passages), batch_size_passages)):
        batch_passages = passages[i:i+batch_size_passages]
        with torch.no_grad():
            P = embed_passages(batch_passages, model=a_enc, tokenizer=tokenizer, device = device, max_length=128)
        scores = torch.matmul(Q, P.T)
        scores_list.append(scores)
    scores = torch.cat(scores_list, dim=1)
    topk = torch.topk(scores, k=20, dim=1)
    topk_indices = topk.indices
    topk_scores = topk.values     

    selected_passages = select_by_indices(topk_indices, passages)
    predicted_answers = selected_passages
    # print the results
    for i in range(10):
        logging.info(f'Question title: {q_titles[i]}')
        logging.info('\n')
        logging.info(f'Question body: {q_bodies[i]}')
        logging.info('\n')        
        logging.info(f'True answer: {true_answers[i]}')
        logging.info('\n')        
        logging.info(f'Predicted answer: {predicted_answers[i]}')
        logging.info('------------------------')
        ## print predicted answers all passages after the next line
        logging.info('Predicted answers all passages:')
        for j in range(5):
            logging.info(f'{j+1}. {passages[topk_indices[i][j]]}')

        logging.info('------------------------')
        logging.info('------------------------')






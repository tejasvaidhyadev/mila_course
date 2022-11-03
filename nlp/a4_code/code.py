
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

# ######################## PART 1: PROVIDED CODE ########################

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

def get_class_output(model, batch):
    # Since this is similar to a previous question, it is left ungraded
    # TODO: your work below.
    pass

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


def embed_passages(passages: 'list[str]', model, tokenizer, device='cpu', max_length=512):
    passages = tokenizer(passages, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    passages = model(**passages)
    passages = passages.last_hidden_state
    # get the CLS token
    passages = passages[:, 0, :]
    return passages

def embed_questions(titles, bodies, model, tokenizer, device='cpu', max_length=512):
    questions = tokenizer(titles, bodies, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    questions = model(**questions)
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




if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import f1_score  # Make sure sklearn is installed

    random.seed(2022)
    torch.manual_seed(2022)

    # Parameters (you can change them)
    sample_size = 2500  # Change this if you want to take a subset of data for testing
    batch_size = 64
    n_epochs = 2
    num_words = 50000

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################### PART 1: TEST CODE ######################
    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data/nli")
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    full_text = (
        train_raw["premise"]
        + train_raw["hypothesis"]
        + valid_raw["premise"]
        + valid_raw["hypothesis"]
    )
    
    print("=" * 80)
    print("Running test code for part 1")
    print("-" * 80)

    train_loader = torch.utils.data.DataLoader(
        NLIDataset(train_raw), batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        NLIDataset(valid_raw), batch_size=batch_size, shuffle=False
    )

    model = CustomDistilBert().to(device)
    optimizer = model.assign_optimizer(lr=1e-4)

    for epoch in range(n_epochs):
        loss = train_distilbert(model, train_loader, device=device)

        preds, targets = eval_distilbert(model, valid_loader, device=device)
        preds = preds.round()

        score = f1_score(targets.cpu(), preds.cpu())
        print("Epoch:", epoch)
        print("Training loss:", loss)
        print("Validation F1 score:", score)
        print()

    # ###################### PART 2: TEST CODE ######################
    freeze_params(model.get_distilbert()) # Now, model should have no trainable parameters

    sp = SoftPrompting(p=5, e=model.get_distilbert().embeddings.word_embeddings.embedding_dim)
    batch = model.tokenize(
        ["This is a premise.", "This is another premise."],
        ["This is a hypothesis.", "This is another hypothesis."],
    )
    batch.input_embedded = sp(model.get_distilbert().embeddings(batch.input_ids))
    batch.attention_mask = pad_attention_mask(batch.attention_mask, 5)

    # ###################### PART 3: TEST CODE ######################
    # Preliminary
    bsize = 8
    qa_data = dict(
        train = pd.read_csv('data/qa/train.csv'),
        valid = pd.read_csv('data/qa/validation.csv'),
        answers = pd.read_csv('data/qa/answers.csv'),
    )

    q_titles = qa_data['train'].loc[:bsize-1, 'QuestionTitle'].tolist()
    q_bodies = qa_data['train'].loc[:bsize-1, 'QuestionBody'].tolist()
    answers = qa_data['train'].loc[:bsize-1, 'Answer'].tolist()

    # Loading huggingface models and tokenizers    
    name = 'google/electra-small-discriminator'
    q_enc, a_enc, tokenizer = load_models_and_tokenizer(q_name=name, a_name=name, t_name=name)
    

    # Tokenize batch and get class output
    q_batch, a_batch = tokenize_qa_batch(tokenizer, q_titles, q_bodies, answers)

    q_out = get_class_output(q_enc, q_batch)
    a_out = get_class_output(a_enc, a_batch)

    # Implement in-batch negative sampling
    S = inbatch_negative_sampling(q_out, a_out)

    # Implement contrastive loss
    loss = contrastive_loss_criterion(S)
    # or
    # > loss = contrastive_loss_criterion(S, labels=...)

    # Implement functions to run retrieval on list of passages
    titles = q_titles
    bodies = q_bodies
    passages = answers + answers
    Q = embed_questions(titles, bodies, model=q_enc, tokenizer=tokenizer, max_length=16)
    P = embed_passages(passages, model=a_enc, tokenizer=tokenizer, max_length=16)

    indices, scores = get_topk_indices(Q, P, k=5)
    selected = select_by_indices(indices, passages)

    # Implement evaluation metrics
    retrieved_indices = [[1, 2, 12, 4], [30, 11, 14, 2], [16, 22, 3, 5]]
    true_indices = [1, 2, 3]

    print("Recall@k:", recall_at_k(retrieved_indices, true_indices, k=3))

    print("MRR:", mean_reciprocal_rank(retrieved_indices, true_indices))
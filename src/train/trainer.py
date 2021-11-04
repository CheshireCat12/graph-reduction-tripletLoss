from pathlib import Path
from os.path import join

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from torch_geometric import seed_everything
from src.models.graph_u_net import GraphUNet
import networkx as nx
import numpy as np
import shutil
import torch
import torch_geometric
import torch_geometric.utils as tg_utils
import xml.etree.ElementTree as ET
import xml.dom.minidom as md

from collections import defaultdict, namedtuple
from networkx.readwrite.graphml import write_graphml_lxml
from os.path import join
from pathlib import Path
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple
import json


def train(model: torch.nn.Module,
          train_loader: torch_geometric.data.DataLoader,
          optimizer,
          criterion) -> None:
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out, *_ = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model: torch.nn.Module,
         test_loader: torch_geometric.data.DataLoader) -> float:
    model.eval()

    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        out, *_ = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.


def save_model(model: torch.nn.Module, folder: str, filename: str) -> None:
    model.eval()
    Path(folder).mkdir(parents=True, exist_ok=True)
    filename = join(folder, filename)
    torch.save(model.state_dict(), filename)


def optimize(model,
             train_loader,
             val_loader,
             test_loader,
             criterion,
             num_epochs: int,
             seed: int,
             stats: dict,
             args) -> None:

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    stats[seed]['train_acc'] = []
    stats[seed]['val_acc'] = []
    stats[seed]['best_val_acc'] = float('-inf')
    stats[seed]['best_test_acc'] = float('-inf')

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, criterion)

        stats[seed]['train_acc'].append(test(model, train_loader))
        stats[seed]['val_acc'].append(test(model, val_loader))

        if stats[seed]['val_acc'][epoch] > stats[seed]['best_val_acc']:
            stats[seed]['best_val_acc'] = stats[seed]['val_acc'][epoch]
            stats[seed]['best_test_acc'] = test(model, test_loader)

            folder, filename = join(args.folder_results, 'trained_models/'), f'trained_gnn_{seed}.pt'
            save_model(model, folder, filename)

    print(f'Best val acc: {stats[seed]["best_val_acc"]:.2f}')
    print(f'Best test acc: {stats[seed]["best_test_acc"]:.2f}')

def start_training(args, seeds, dataset_):
    # Load Dataset
    # dataset_ = TUDataset(root=args.folder_data,
    #                      name=args.dataset)
    dataset_size = len(dataset_)

    perc_train = args.percentage_train / 100
    perc_val = (1 - perc_train) / 2
    train_size = int(dataset_size * perc_train)
    val_size = int(dataset_size * perc_val)

    criterion = torch.nn.CrossEntropyLoss()

    # Split train & test set
    # seed_everything(43)
    # seeds = np.random.randint(2000, size=args.num_seeds)

    dataset = dataset_.copy()
    stats = {'parameters': vars(args)}

    for idx, seed in enumerate(seeds):
        print(f'Train seed: {seed} [{idx+1}/{len(seeds)}]')
        seed_everything(seed)
        dataset = dataset.shuffle()

        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size+val_size]
        test_dataset = dataset[train_size+val_size:]

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = GraphUNet(in_channels=dataset.num_node_features,
                          hidden_channels=64,
                          dim_gr_embedding=32,
                          out_channels=dataset.num_classes,
                          depth=1)

        stats[int(seed)] = {}

        optimize(model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 test_loader=test_loader,
                 criterion=criterion,
                 num_epochs=args.num_epochs,
                 seed=seed,
                 stats=stats)

        with open(join(args.folder_results, f'stats_gnn_training.json'), 'w') as f:
            json.dump(stats, f, indent=4)

import xml.dom.minidom as md
import xml.etree.ElementTree as ET
from collections import defaultdict
from os.path import join
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import torch
import torch_geometric
from networkx.readwrite.graphml import write_graphml_lxml
from torch_geometric import seed_everything
from torch_geometric.data import Data, DataLoader
import torch_geometric.utils as tg_utils


def reduce_graph(batch: torch_geometric.data.Batch,
                 trained_model: torch.nn.Module) -> torch_geometric.data.Data:
    """
    Reduce the given graph.
    First it is extracted from the batch, then reduce with the trained model.

    Args:
        batch (torch_geometric): Graph to reduce
        trained_model (torch.nn.Module): trained model used to reduce the graph.

    Returns:
        torch_geometric.Data: The reduced graph
    """
    _, reduced_graph_x, reduced_graph_edge_index, _ = trained_model(batch.x,
                                                                    batch.edge_index,
                                                                    batch.batch)

    reduced_graph = Data(x=reduced_graph_x,
                         edge_index=reduced_graph_edge_index)

    return reduced_graph


def convert_2_nx(reduced_graph: torch_geometric.data.Data) -> nx.Graph:
    """
    Convert the graph from torch_geometric.Data to nx with the correct
    formatting of the node features.

    Args:
        reduced_graph (torch_geometric.Data): the graph to convert

    Returns:
        nx.Graph: the converted graph
    """
    nx_graph = tg_utils.to_networkx(reduced_graph,
                                    node_attrs=['x'],
                                    to_undirected=True)

    # Convert the node feature vector into string.
    # The write_graphml() does not support lists/vectors.
    for node, d in nx_graph.nodes(data=True):
        for k, v in d.items():
            if isinstance(v, float):
                print('A float has to be changed is float', )
                v = [v] * reduced_graph.x.size(1)
            d[k] = str(v)

    return nx_graph


def save_graph(nx_reduced_graph: nx.Graph, idx: int, folder: str) -> None:
    """
    Write the graph in the given folder under 'gr_<idx>.graphml' filename.

    Args:
        graph (nx.Graph): graph to save
        idx (int): idx of the graph (used in the filename 'gr_<idx>.graphml')
        folder (str): folder where to solve the graph

    Returns:
        None
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    filename = join(folder, f'gr_{idx}.graphml')
    write_graphml_lxml(nx_reduced_graph,
                       filename,
                       infer_numeric_types=True)


def parse_class_to_xml(name_set: str,
                       classes: List[Tuple[int, int]],
                       folder: str) -> None:
    """
    Parse the graph classes into xml file

    Args:
        name_set (str): name of the set to save (e.g., train, val, test)
        classes (List[Tuple[int, int]]):
            list containing the idx and class tuple for each graph
        folder (str): folder where to save the classes

    Returns:
        None
    """
    graph_collection = ET.Element('GraphCollection')

    finger_prints = ET.SubElement(graph_collection, 'fingerprints')

    for idx_graph, class_ in classes:
        print_ = ET.SubElement(finger_prints, 'print')
        print_.set('file', f'gr_{idx_graph}.graphml')
        print_.set('class', str(class_))

    b_xml = ET.tostring(graph_collection).decode()
    newxml = md.parseString(b_xml)

    Path(folder).mkdir(parents=True, exist_ok=True)
    filename = join(folder, f'{name_set}.cxl')
    with open(filename, mode='w') as f:
        f.write(newxml.toprettyxml(indent=' ', newl='\n'))


def save_classes(graph_classes: defaultdict, folder: str) -> None:
    """
    Save the corresponding classes for each graph.

    Args:
        graph_classes (defaultdict):
            Dict containing the set of data as key and
             the value is the list of tuple containing
             the idx of the graph and its corresponding class.
        folder (str): folder where to save the classes

    Returns:
        None
    """
    for name_set, idx_classes in graph_classes.items():
        parse_class_to_xml(name_set, idx_classes, folder)

from src.models.graph_u_net import GraphUNet

def start_converting(args, dataset_, seeds):

    dataset_size = len(dataset_)

    perc_train = args.percentage_train / 100
    perc_val = (1 - perc_train) / 2
    train_size = int(dataset_size * perc_train)
    val_size = int(dataset_size * perc_val)

    dataset = dataset_.copy()
    for idx, seed in enumerate(seeds):
        print(f'Convert seed: {seed} [{idx + 1}/{len(seeds)}]')

        seed_everything(seed)
        dataset = dataset.shuffle()
        convert_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        trained_model = GraphUNet(in_channels=dataset.num_node_features,
                                  hidden_channels=64,
                                  dim_gr_embedding=32,
                                  out_channels=dataset.num_classes,
                                  depth=1)
        trained_model.load_state_dict(torch.load(join(args.folder_results, f'trained_models/trained_gnn_{seed}.pt')))
        trained_model.eval()

        folder = f'{args.folder_results}/{seed}/data'
        graph_classes = defaultdict(list)

        for idx, batch in enumerate(convert_loader):
            reduced_graph = reduce_graph(batch, trained_model)
            reduced_graph = Data(x=reduced_graph.x, edge_index=reduced_graph.edge_index)
            nx_reduced_graph = convert_2_nx(reduced_graph)

            save_graph(nx_reduced_graph, idx, folder)

            # Get the class value
            current_graph_classes = int(batch.y.data[0])

            # Split the dataset into train, val, and test sets
            if idx < train_size:
                graph_classes['train'].append((idx, current_graph_classes))
            elif idx < train_size + val_size:
                graph_classes['validation'].append((idx, current_graph_classes))
            else:
                graph_classes['test'].append((idx, current_graph_classes))

        save_classes(graph_classes, folder)

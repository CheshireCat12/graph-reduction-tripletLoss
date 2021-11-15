##########
# Import #
##########
from argparse import ArgumentParser, Namespace

import numpy as np
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset

from src.train.trainer import start_training
from src.graph_converter import start_converting

def main(args):
    dataset_ = TUDataset(root=args.folder_data,
                         name=args.dataset)

    seed_everything(43)
    seeds = np.random.randint(2000, size=args.num_seeds)

    if args.train:
        start_training(args, dataset_, seeds)

    start_converting(args, dataset_, seeds)


if __name__ == '__main__':
    parser = ArgumentParser(description='Train GNN')

    # Parameters of the experiment
    parser.add_argument('--train',
                        action='store_true',
                        help='Choose the experiment to run.')

    parser.add_argument('--num-epochs',
                        required=True, type=int,
                        help='Number of epochs')
    parser.add_argument('--percentage-train',
                        required=True, type=int,
                        help='Percentage of data used in the training set.\n'
                             'The remaining data are split in 2 equivalent size set.')
    parser.add_argument('--dataset',
                        required=True, type=str,
                        help='Name of the dataset')
    parser.add_argument('--num-seeds',
                        required=True, type=int,
                        help='Number of seeds to generate.')
    parser.add_argument('--specific-seed',
                        type=int, default=False,
                        help='Train GNN with specific seed.')


    parser.add_argument('--dim-hidden-vec',
                        required=True, type=int,
                        help='Give the dimension of hidden vector dim')
    parser.add_argument('--dim-gr-embedding',
                        required=True, type=int,
                        help='Give the vector size of the node features of the reduced graphs')


    parser.add_argument('--depth',
                        required=True, type=int,
                        help='Depth of the model, defines the size of reduced graphs')
    parser.add_argument('--augment-depth-by-step',
                        action='store_true',
                        help='Defines if the depth is settled once at the beginning or'
                             'if the model is trained multiple times with increasing depth')
    parser.add_argument('--freeze-parameters',
                        action='store_true',
                        help='Freeze the trained parameters before increasing the depth')

    parser.add_argument('--folder-data',
                        type=str,
                        help='Folder where to save the graph datasets.')
    parser.add_argument('--folder-results',
                        type=str,
                        help='Folder where to save the stats, the trained models, the reduced graphs')
    parser.add_argument('--name-experiment', type=str, required=True,
                        help='Specify the experiment name under which to save the experiment')

    args = parser.parse_args()

    if not args.folder_data:
        args.folder_data = f'./data/{args.dataset}/'
    if not args.folder_results:
        args.folder_results = f'./results/{args.dataset}/{args.name_experiment}/'

    # Merge the command line parameters with the constant parameters
    # arguments = Namespace(**vars(args),
    #                       **{'folder_data': folder_data,
    #                          'folder_results': folder_results})
    main(args)

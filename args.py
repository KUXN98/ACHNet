import argparse


def get_args():
    parser = argparse.ArgumentParser(description='code for ACHNet')
    parser.add_argument('--dataset', default='cifar-100-IF100', type=str,
                        help='Dataset name.')
    parser.add_argument('--root',
                        default=None,
                        type=str,
                        help='Path of dataset')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size.(default: 8)')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate.(default: 1e-5)')
    parser.add_argument('--code_length', default='32, 64, 96', type=str,
                        help='Binary hash code length.(default: 32,64,96)')
    parser.add_argument('--feature_dim', default=2000, type=int,
                        help='number of classes.(default: 2000)')
    parser.add_argument('--num_classes', default=100, type=int,
                        help='number of classes.(default: 100)')
    parser.add_argument('--max_iter', default=100, type=int,
                        help='Number of iterations.(default: 300)')
    parser.add_argument('--num_workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=4, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--lamb', default=0.2, type=float,
                        help='Hyper-parameter: balance between CE loss and contrasive loss.')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')
    parser.add_argument('--evaluate-interval', default=4, type=int,
                        help='Evaluation interval.(default: 4)')

    args = parser.parse_args()

    return args


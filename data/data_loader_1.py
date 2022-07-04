from PIL import ImageFile
from data import cifar100, imagenet

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_workers(int): Number of loading data threads.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """

    if dataset == 'cifar-100-IF1':
        root = root + '/cifar-100-IF1'
        train_dataloader, query_dataloader, retrieval_dataloader = cifar100.load_data(root,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )

    elif dataset == 'cifar-100-IF50':
        root = root + '/cifar-100-IF50'
        train_dataloader, query_dataloader, retrieval_dataloader = cifar100.load_data(root,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'cifar-100-IF100':
        root = root + '/cifar-100-IF100'
        train_dataloader, query_dataloader, retrieval_dataloader = cifar100.load_data(root,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )

    elif dataset == 'imagenet-100-IF1':
        train_dataloader, query_dataloader, retrieval_dataloader = imagenet.load_data(dataset,
                                                                                      root,
                                                                                      batch_size,
                                                                                      num_workers,
                                                                                      )

    elif dataset == 'imagenet-100-IF50':
        train_dataloader, query_dataloader, retrieval_dataloader = imagenet.load_data(dataset,
                                                                                      root,
                                                                                      batch_size,
                                                                                      num_workers,
                                                                                      )
    elif dataset == 'imagenet-100-IF100':
        train_dataloader, query_dataloader, retrieval_dataloader = imagenet.load_data(dataset,
                                                                                      root,
                                                                                      batch_size,
                                                                                      num_workers,
                                                                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return train_dataloader, query_dataloader, retrieval_dataloader



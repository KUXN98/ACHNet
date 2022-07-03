import os
import torch
import numpy as np
import random
import torch.optim as optim

from args import get_args
from loguru import logger
from network import resnet
from data import imb_data_loader
from tools import model_train, loss


def main(args):

    # Get device
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(args.gpu)
    else:
        args.device = torch.device("cpu")

    # Set random seed
    if args.seed is not None:
        torch.backends.cudnn.benchmark = True
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

    logger.add('logs/{}_code_{}_gamma_{}_batch_size_{}.log'.format(
        args.dataset,
        args.code_length,
        args.lamb,
        args.batch_size,
    ),
        rotation='500 MB',
        level='INFO',
    )
    logger.info(args)

    # Build dataset
    dataset = args.dataset.split('-')[0]

    train_loader, query_loader, retrieval_loader = imb_data_loader.load_data(args.dataset,
                                                                             args.root,
                                                                             args.batch_size,
                                                                             args.num_workers
                                                                             )

    print('dataset loading end')

    # Print class-samples number
    class_samples = torch.Tensor(np.zeros(args.num_classes))
    for _, targets, _ in train_loader:
        class_samples += torch.sum(targets, dim=0)
    print('class sample number:{}'.format(class_samples))

    args.code_length = list(map(int, args.code_length.split(',')))

    for length in args.code_length:
        length = int(length)

        # Build network

        model = resnet.load_model(args.feature_dim, length, args.num_classes)
        model.to(args.device)

        if dataset == 'cifar':
            optimizer = optim.RMSprop(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4,
            )
        elif dataset == 'imagenet':
            feature_params = []
            hashing_params = []
            for p_name, p in model.named_parameters():
                if p_name.startswith('features'):
                    feature_params += [p]
                else:
                    hashing_params += [p]

            optimizer = optim.RMSprop([
                {'params': feature_params, 'lr': 0.1*args.lr, 'weight_decay': 5e-4},
                {'params': hashing_params, 'lr': 10*args.lr, 'weight_decay': 5e-4}]
            )
        else:
            optimizer = None
            print('dataset is not right')
            exit()

        criterion_ce = loss.CELoss().to(args.device)
        criterion_con = loss.CenConLoss().to(args.device)

        # Training
        checkpoint = model_train.train(args,
                                       length,
                                       model,
                                       optimizer,
                                       train_loader,
                                       query_loader,
                                       retrieval_loader,
                                       criterion_ce,
                                       criterion_con
                                       )
        logger.info('[code_length:{}][map:{:.4f}]'.format(length, checkpoint['map']))

        # Save checkpoint
        torch.save(
            checkpoint,
            os.path.join('checkpoints', '{}_model_{}_code_{}_beta_{}_gamma_{}_map_{:.4f}_batchsize_{}_maxIter_{}.pt'.format(
                args.dataset,
                args.arch,
                length,
                args.beta,
                args.gamma,
                checkpoint['map'],
                args.batch_size,
                args.max_iter)
                         )
            )


if __name__ == '__main__':
    main(get_args())

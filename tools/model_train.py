import time

from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from tools.utils import *
from tools.evaluate import mean_average_precision


def train(args, length, model, optimizer, train_loader, query_loader,
          retrieval_loader, criterion_ce, criterion_con):
    """
    Training network.

    Args:
        args: args.
        length: code length.
        model: resnet34.
        optimizer: optimizer.
        train_loader: train data loader.
        query_loader: query data loader.
        retrieval_loader: retrieval data loader.
        criterion_ce: CE loss.
        criterion_con: Cen_con loss.
        class_sample: Class sample number.

    Returns:
        checkpoint: Checkpoint of network.
    """
    scheduler = CosineAnnealingLR(
        optimizer,
        args.max_iter,
        args.lr / 100,
    )

    # Initialization
    running_loss = 0.
    best_map = 0.

    # Training

    for it in range(args.max_iter):

        generate_hash_center_start = time.time()
        hash_center = generate_hash_center(model, train_loader, args.device, length)
        generate_hash_center_end = time.time()
        print('iter[{}/{}] generate_hash_center time{:.3f}'.format(it, args.max_iter,
                                                                   generate_hash_center_end - generate_hash_center_start))

        model.train()
        tic = time.time()
        iter_num = 0
        epoch_loss = 0
        loss_con_num = 0

        for data, targets, index in train_loader:
            iter_num += 1

            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)

            optimizer.zero_grad()

            hash_center = hash_center.to('cuda')
            
            hashcodes, assignments, direct_feature = model(data)

            loss_ce = criterion_ce(assignments, targets)

            loss_con = criterion_con(hashcodes, hash_center, targets)
            loss = args.lamb * loss_ce + loss_con

            running_loss = running_loss + loss.item()
            epoch_loss = epoch_loss + loss.item()
            loss_con_num = loss_con_num + loss_con

            loss.backward()
            optimizer.step()

        # update step
        scheduler.step()
        training_time = time.time() - tic

        print('iter[{}/{}] train time{:.3f} loss{:.3f} con_loss{:.3f}'.format(it, args.max_iter, training_time, epoch_loss,
                                                                              loss_con_num))

        # Evaluate
        if it % args.evaluate_interval == args.evaluate_interval - 1:

            start = time.time()
            query_code, query_assignment, label_q = generate_code(model,
                                                                  query_loader,
                                                                  length,
                                                                  args.num_classes,
                                                                  args.device,
                                                                  0
                                                                  )
            retrieval_code, retrieval_assignment, _ = generate_code(model,
                                                                    retrieval_loader,
                                                                    length,
                                                                    args.num_classes,
                                                                    args.device,
                                                                    1
                                                                    )

            correct = get_correct_num(query_assignment, label_q)

            class_correct_num, class_sample_num = np.array(sample_num_per_class(query_assignment, label_q,
                                                                                100, 0), dtype=int)

            train_total_correct_class = class_correct_num
            train_total_sample_class = class_sample_num

            train_accuracy_pre_class = np.array(correct_per_class(train_total_correct_class,
                                                                  train_total_sample_class, 100))
            train_three = [train_accuracy_pre_class[:33].mean(), train_accuracy_pre_class[33:66].mean(),
                           train_accuracy_pre_class[66:].mean()]
            print('epoch = {}, total test accuracy = {:.3f}, class accuracy = {}'.
                  format(it, correct / 10000, train_three))

            query_targets = query_loader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_loader.dataset.get_onehot_targets()

            # Compute map
            mAP, mAP_class = mean_average_precision(
                query_code.to(args.device),
                retrieval_code.to(args.device),
                query_targets.to(args.device),
                retrieval_targets.to(args.device),
                args.device,
                args.topk,
            )

            # Log
            logger.info('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][time:{:.2f}][map class:{}]'.format(
                it + 1,
                args.max_iter,
                running_loss / args.evaluate_interval,
                mAP,
                training_time,
                mAP_class
            ))
            running_loss = 0.

            # Checkpoint
            if best_map < mAP:
                best_map = mAP

                checkpoint = {
                    'network': model.state_dict(),
                    'qB': query_code.cpu(),
                    'rB': retrieval_code.cpu(),
                    'qL': query_targets.cpu(),
                    'rL': retrieval_targets.cpu(),
                    'qAssignment': query_assignment.cpu(),
                    'rAssignment': retrieval_assignment.cpu(),
                    'map': best_map,
                    'hash_center': hash_center.cpu(),
                    'gamma': args.gamma,
                }
            end = time.time()
            print('evaluate time = {:3f}'.format(end - start))

    return checkpoint

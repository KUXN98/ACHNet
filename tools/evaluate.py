import torch


def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=-1,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_ap = 0.0
    mean_h, mean_m, mean_t = 0.0, 0.0, 0.0
    query_h, query_m, query_t = 0.0, 0.0, 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float()

        # Map of head, middle and tail class
        label = torch.where(query_labels[i, :] == 1)[0]
        num = label // 33
        mean_ap += (score / index).mean()
        if num == 0:
            mean_h += (score / index).mean()
            query_h += 1
        elif num == 1:
            mean_m += (score / index).mean()
            query_m += 1
        else:
            mean_t += (score / index).mean()
            query_t += 1

    mean_ap = mean_ap / num_query
    mean_ap_h = mean_h / query_h
    mean_ap_m = mean_m / query_m
    mean_ap_t = mean_t / query_t
    torch.cuda.empty_cache()
    mean_ap_class = [mean_ap_h, mean_ap_m, mean_ap_t]

    return mean_ap, mean_ap_class

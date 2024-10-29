import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    data_sample = dataset_items[0]
    for key in data_sample.keys():
        if type(data_sample[key]) is torch.Tensor:
            result_batch[key] = torch.vstack(
                [elem[key] for elem in dataset_items]
            )
        else:
            result_batch[key] = [elem[key] for elem in dataset_items]

    return result_batch

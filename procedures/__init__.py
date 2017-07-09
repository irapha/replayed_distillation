
def get(procedure):
    if procedure == 'train':
        from . import train
        return train
    elif procedure == 'compute_stats':
        from . import compute_stats
        return compute_stats
    elif procedure == 'optimize_dataset':
        from . import optimize_dataset
        return optimize_dataset
    elif procedure == 'distill':
        from . import distill
        return distill
    else:
        raise NotImplemented('This procedure not implemented yet')

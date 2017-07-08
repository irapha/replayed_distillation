
def get(procedure):
    if procedure == 'train':
        from . import train
        return train
    elif procedure == 'distill':
        from . import distill
        return distill
    elif procedure == 'compute_stats':
        from . import compute_stats
        return compute_stats
    else:
        raise NotImplemented('This procedure not implemented yet')

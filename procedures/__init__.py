
def get(procedure):
    if procedure == 'train':
        from . import train
        return train
    elif procedure == 'distill':
        from . import distill
        return distill
    else:
        raise NotImplemented('This procedure not implemented yet')


def get(procedure):
    if procedure == 'train':
        from . import train
        return train
    else:
        raise NotImplemented('This procedure not implemented yet')

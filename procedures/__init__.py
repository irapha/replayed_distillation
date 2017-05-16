
def get(procedure):
    if procedure == 'train':
        from . import train
        return train
    if procedure == 'train_conv':
        from . import train_conv
        return train_conv
    elif procedure == 'distill':
        from . import distill
        return distill
    elif procedure == 'distill_conv':
        from . import distill_conv
        return distill_conv
    elif procedure == 'replay':
        from . import replay
        return replay
    elif procedure == 'replay_conv':
        from . import replay_conv
        return replay_conv
    else:
        raise NotImplemented('This procedure not implemented yet')


def get(model_name):
    if model_name == 'lenet':
        from . import lenet5 as d
        return d
    if model_name == 'hinton1200':
        from . import hinton1200 as d
        return d
    elif model_name == 'hinton800':
        from . import hinton800 as d
        return d
    else:
        raise NotImplemented('This model not implemented yet')

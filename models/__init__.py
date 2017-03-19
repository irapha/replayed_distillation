
def get(model_name):
    if model_name == 'hinton1200':
        from . import hinton1200 as d
        return d
    else:
        raise NotImplemented('This model not implemented yet')

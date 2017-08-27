
def get(model_name):
    if model_name == 'hinton1200':
        from . import hinton1200 as d
        return d
    elif model_name == 'hinton800':
        from . import hinton800 as d
        return d
    elif model_name == 'lenet':
        from . import lenet as d
        return d
    elif model_name == 'lenet_half':
        from . import lenet_half as d
        return d
    elif model_name == 'vgg19':
        from . import vgg19 as d
        return d
    elif model_name == 'vgg16':
        from . import vgg16 as d
        return d
    elif model_name == 'alex':
        from . import alex as d
        return d
    else:
        raise NotImplemented('This model not implemented yet')

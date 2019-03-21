import json
from tensorboardX import SummaryWriter
def setup_tensorboard(location):
    writer = SummaryWriter(location)
    return writer
def check_params(config_file):
    params = json.load(open(config_file))
    config={}
    if len(params)!=3:
        raise ValueError("Wrong configuration file")
    if not [*params] != ['model_params', 'train_params', 'infer_params']:
        raise ValueError("Wrong configuration file")
    model_params = params["model_params"]
    train_params = params["train_params"]
    infer_params = params["infer_params"]
    for param in params:
        for obj in param:
            config[param][obj]=eval(param[obj])
    return config
def get_params(config,param,default):
    if param not in config or config[param] is None:
        return default
    else:
        return config[param]
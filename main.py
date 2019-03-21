import sys
import argparse
import torch
from data import Dataset
from model import ConvLSTM
from utils import check_params, get_params, setup_tensorboard
from trainer import train, train_distributed
args = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str,default="sample_configs/config.json")
parser.add_argument("-m","--mode",type=str,default="train")
parser.add_argument("--distributed",type=bool,default=False)
args = parser.parse_known_args(args)
def main(config_file,mode,distributed):
    config = check_params(config_file)
    if mode in ["Train","train"]:
        train_dataset = Dataset(config["train_params"]["input_path"],config["train_params"]["imsize"])
        if distributed:
            import horovod as hvd
            hvd.init()
            if hvd.rank()==0:
                writer = setup_tensorboard(get_params(config["train_params"],"tensorboard_location","./summary/"))
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(),
                                                                            rank=hvd.rank())

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train_params"]["batch_size"],
                                                       sampler=train_sampler, shuffle=True)
            model = ConvLSTM(**config["model_params"])
            optimizer = hvd.DistributedOptimizer(model.optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            train_distributed(model,train_loader,optimizer,config,writer)
        else:
            writer = setup_tensorboard(get_params(config["train_params"], "tensorboard_location", "./summary/"))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train_params"]["batch_size"],
                                                     shuffle = True)
            model = ConvLSTM(**config["model_params"])
            train(model,train_loader,model.optimizer,config,writer)
    elif mode in ["infer","Infer"]:
        model = ConvLSTM(**config["model_params"])
        model.load_state_dict(config["infer_params"]["model_save_path"])
        output_file = open(config["infer_params"]["output_path"])

if __name__=="__main__":
    config_file = args.config_file
    mode = args.mode
    distributed = args.distributed
    main(config_file,mode,distributed)
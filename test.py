from dataset_loader import DatasetLoader
from mt5_en_pt_model import MT5Model
from IO_utils import parse_arguments

import sys
import torch

DATASET_PATH = './data/emtions-en-pt/dataset'

def main():
        args = parse_arguments(sys.argv[1:])
        DATA_FRAC = args['data_frac'] if 'data_frac' in args.keys() else 0.5
        CHECKPOINT_PATH = args['checkpoint_path'] if 'checkpoint_path' in args.keys() else None
        
        assert CHECKPOINT_PATH is not None, "Error: Can't test model without checkpoint_path param"
        
        dataset = DatasetLoader.load_dataset(DATASET_PATH, DATA_FRAC)
        # Load the model state from checkpoint
        mt5Model = MT5Model()
        checkpoint = torch.load('./checkpoints/dimap-mt5-en-pt_checkpoints/checkpoint-15500/pytorch_model.bin')
        mt5Model.load_state_dict(checkpoint)

if __name__ == "__main__":
    main()
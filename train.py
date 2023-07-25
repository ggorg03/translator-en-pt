from dataset_loader import DatasetLoader
from mt5_en_pt_model import MT5Model
from IO_utils import parse_arguments

import sys

DATASET_PATH = './data/emtions-en-pt/dataset'

def main():
        args = parse_arguments(sys.argv[1:])
        EPOCHS = args['epochs'] if 'epochs' in args.keys() else 3
        DATA_FRAC = args['data_frac'] if 'data_frac' in args.keys() else 0.5
        
        dataset = DatasetLoader.load_dataset(DATASET_PATH, DATA_FRAC)
    
        mt5Model = MT5Model(dataset, EPOCHS)
        mt5Model.train()
        
        # SIMPLE TEST
        text1 = "I like cats"
        text2 = "I like dogs"
        print(f'{text1} -> {mt5Model.infer(text1)}')
        print(f'{text2} -> {mt5Model.infer(text2)}')

if __name__ == "__main__":
    main()
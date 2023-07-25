from datasets import Dataset, DatasetDict

class DatasetLoader:
    @staticmethod
    def load_dataset(path:str, frac_size=1.0) -> DatasetDict:
        dataSet = DatasetDict.load_from_disk(path)
        return DatasetLoader.prepare_dataset(dataSet, frac_size)

    @staticmethod
    def prepare_dataset(ds:DatasetDict, frac_size) -> DatasetDict:
        # frac_size must be a value between 0 and 1
        if 0.0 > frac_size > 1.0:
            frac_size = 1.0
        # selecting just id, en, pt columns
        ds_train = ds['train']
        df_train = ds_train.to_pandas()[['id', 'en', 'pt']]
        # shuffle data and getting a sample
        df_train = df_train.sample(frac=frac_size, random_state=42).reset_index(drop=True)
        ds["train"] = Dataset.from_pandas(df_train)
        #spliting data into train, test and validation -> train 80% | test 10% | val 10%
        ds = DatasetLoader.split_column_dataset(
            ds, 
            col='train', 
            new_col='test',
            frac_size=0.2
        )
        ds = DatasetLoader.split_column_dataset(
            ds,
            col='test',
            new_col='validation',
            frac_size=0.5
        )

        return ds

    @staticmethod
    def split_column_dataset(ds:DatasetDict, col, new_col, frac_size:float=0.1):
        new_ds = ds[col].train_test_split(test_size=frac_size)
        ds[col] = new_ds['train']
        ds[new_col] = new_ds['test']
        return ds
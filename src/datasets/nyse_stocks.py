from .dataset import Dataset


class StocksDataset(Dataset):
    def __init__(self, name, file_name):
        super(StocksDataset, self).__init__(name, file_name)

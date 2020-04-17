from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import RandomSampler, BatchSampler
from torch import stack, device, LongTensor, transpose
from torch import cuda
from numpy import arange

class DataStreamer(IterableDataset):
    
    def __init__(self, data=None, X=None, Y=None, batch_size=None):
        self.cuda0 = device('cuda:0')
        if data is not None:
            self.data = data
            print("Collating data...")
            self.X = stack([dpoint[0] for dpoint in self.data], 1).to(device=self.cuda0)
            self.Y = stack([dpoint[1] for dpoint in self.data], 1).to(device=self.cuda0)
            print("...done.")
        elif X is not None and Y is not None:
            self.data = None
            self.X = X
            self.Y = Y
        else:
            raise Exception("No data provided!")
        #
        self.batch_size = batch_size

    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        if self.batch_size is None:
            self.batch_size = 1
        #
        N = self.X.size()[1]
        self.random_sampler = RandomSampler(arange(N), replacement=False, num_samples=None)
        # print(f"N: {N}, N_transformed: {len(list(self.random_sampler))}")
        if self.data is not None:
            self.random_sampler = RandomSampler(self.data, replacement=False, num_samples=None)
        #
        self.batch_sampler = BatchSampler(
            self.random_sampler,
            self.batch_size,
            False
        )
        self.batch_iterator = iter(self.batch_sampler)
        return self
    
    def __next__(self):
        next_batch_idx = next(self.batch_iterator)
        perm = cuda.LongTensor(next_batch_idx)
        return (self.X[:, perm], self.Y[:, perm])

    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
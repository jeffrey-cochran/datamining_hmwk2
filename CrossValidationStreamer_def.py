from DataStreamer_def import DataStreamer
from torch.utils.data import RandomSampler, BatchSampler
from torch import cuda

class CrossValidationStreamer(object):

    def __init__(self, data_streamer, k):
        self.k = k
        self.data_streamer = data_streamer
        return

    def __iter__(self):
        N = self.data_streamer.X.size()[1]
        batch_size = int(N / self.k)
        self.random_sampler = RandomSampler(
            self.data_streamer.data, 
            replacement=False, 
            num_samples=None
        )
        self.batch_sampler = BatchSampler(
            self.random_sampler,
            batch_size,
            False
        )
        self.batches = list(self.batch_sampler)
        self.training_batches = iter([ 
            [
                idx for j in range(self.k)  if j != i for idx in self.batches[j] 
            ] for i in range(self.k) 
        ])
        self.test_batches = iter(self.batches)
        return self

    def __next__(self):
        train_perm = cuda.LongTensor(next(self.training_batches))
        test_perm = cuda.LongTensor(next(self.test_batches))
        return (
            DataStreamer(X=self.data_streamer.X[:, train_perm], Y=self.data_streamer.Y[:, train_perm]),
            DataStreamer(X=self.data_streamer.X[:, test_perm], Y=self.data_streamer.Y[:, test_perm])
        )

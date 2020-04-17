import torch
#
class ReshapeImage(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        return

    def __call__(self, sample):
        return sample.reshape(784)


class OneHOT(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        return

    def __call__(self, label):
        one_hot = torch.zeros(10)
        one_hot[label] = 1.
        return one_hot

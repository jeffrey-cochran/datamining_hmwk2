from constants import DECREASE, COSINE
from numpy import cos, pi

class DynamicStepSize(object):

    def __init__(
        self, 
        initial_step_size, 
        method=DECREASE, 
        decrease_factor=0.999,
        step_size_range=(0.0001, 0.01),
        period=100
    ):
        self.current_step_size = initial_step_size
        self.next = self.next_DECREASE if method==DECREASE else self.next_COSINE
        self.decrease_factor = decrease_factor
        self.step_size_min = step_size_range[0]
        self.step_size_max = step_size_range[1]
        self.iter_count = 0
        self.period = float(period)
        return

    def next_DECREASE(self):
        self.current_step_size = self.current_step_size * self.decrease_factor
        return self.current_step_size

    def next_COSINE(self):
        self.iter_count += 1
        self.current_step_size = self.decrease_factor**self.iter_count * (
            self.step_size_min + 0.5 * (self.step_size_max - self.step_size_min) * (
                1 + cos( self.iter_count * pi / self.period)
            ) 
        )
        return self.current_step_size
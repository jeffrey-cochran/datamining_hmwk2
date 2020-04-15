from plot_utils import Series
import matplotlib.pyplot as plt
from constants import TRAIN, TEST
#
#
file_names = [
    "results/problem_1/step_1e-02_tol_1e-03.csv",
    "results/problem_1/step_1e-03_tol_1e-03.csv",
    "results/problem_1/step_2e-03_tol_1e-03.csv",
    "results/problem_1/step_3e-03_tol_1e-03.csv",
    "results/problem_1/step_6e-03_tol_1e-03.csv"
]

labels = [
    "log(tau) = -2",
    "log(tau) = -2.25",
    "log(tau) = -2.5",
    "log(tau) = -2.75",
    "log(tau) = -3"
]

training_fig, training_ax = plt.subplots()
testing_fig, testing_ax = plt.subplots()

for i in range(5):
    #
    # Plot training data
    training_data = Series()
    training_data.load_from_file(
        file_names[i],
        labels[i],
        TRAIN
    )
    training_data.plot(training_ax)
    #
    # Plot testing data
    testing_data = Series()
    testing_data.load_from_file(
        file_names[i],
        labels[i],
        TEST
    )
    testing_data.plot(testing_ax)

training_fig.x_label("Hello")
plt.legend()
plt.show()


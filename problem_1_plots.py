from plot_utils import Series
import matplotlib.pyplot as plt
from matplotlib import rc
from constants import TRAIN, TEST

#
# ==============================================
# plot comparison
# ==============================================
#
colors =[
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple"
]
file_names = [
    "results/problem_1/step_1e-02_tol_1e-03.csv",
    "results/problem_1/step_6e-03_tol_1e-03.csv",
    "results/problem_1/step_3e-03_tol_1e-03.csv",
    "results/problem_1/step_2e-03_tol_1e-03.csv",
    "results/problem_1/step_1e-03_tol_1e-03.csv"
]

labels = [
    r"$log(\tau) = -2$",
    r"$log(\tau) = -2.25$",
    r"$log(\tau) = -2.5$",
    r"$log(\tau) = -2.75$",
    r"$log(\tau) = -3$"
]

training_fig, training_ax = plt.subplots()
# testing_fig, testing_ax = plt.subplots()

for i in range(5):
    #
    # Plot training data
    training_data = Series()
    training_data.load_from_file(
        file_names[i],
        labels[i],
        TRAIN
    )
    training_data.plot(training_ax, color=colors[i])
    #
    # Plot testing data
    testing_data = Series()
    testing_data.load_from_file(
        file_names[i],
        None,
        TEST
    )
    testing_data.plot(training_ax, linestyle="dashed", linewidth=2, color=colors[i])

#
# Set display params
training_ax.legend()
training_ax.set_xlabel("Iterations Over Full Dataset")
training_ax.set_ylabel("Mean Squared Error")

plt.show()


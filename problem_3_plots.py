from plot_utils import Series
import matplotlib.pyplot as plt
from matplotlib import rc
from constants import TRAIN, TEST

#
# ==============================================
# plot batch comparison
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

#
# ==============================================
# plot step size comparison
# ==============================================
#
# num_files = 4
# file_names = [
#     "results/problem_3/a/step_5e-03_batch_100_ridgeCoeff_0.01.csv",
#     "results/problem_3/a/step_5e-03_batch_100_ridgeCoeff_0.1.csv",
#     "results/problem_3/a/step_5e-03_batch_100_ridgeCoeff_1.0.csv",
#     "results/problem_3/a/step_5e-03_batch_100_ridgeCoeff_10.0.csv"
# ]

# labels = [
#     r"$\lambda = 0.01$",
#     r"$\lambda = 0.1$",
#     r"$\lambda = 1.0$",
#     r"$\lambda = 10.0$"
# ]

# training_fig, training_ax = plt.subplots()
# testing_fig, testing_ax = plt.subplots()

# for i in range(num_files):
#     #
#     # Plot training data
#     training_data = Series()
#     training_data.load_from_file(
#         file_names[i],
#         labels[i],
#         TRAIN
#     )
#     training_data.plot(training_ax, color=colors[i])
#     #
#     # Plot testing data
#     testing_data = Series()
#     testing_data.load_from_file(
#         file_names[i],
#         None,
#         TEST
#     )
#     testing_data.plot(training_ax, linestyle="dashed", linewidth=2, color=colors[i])
# training_ax.legend()
# testing_ax.legend()
# training_ax.set_xlabel("Iterations Over Full Dataset")
# testing_ax.set_xlabel("Iterations Over Full Dataset")
# training_ax.set_ylabel("Mean Squared Error")
# testing_ax.set_ylabel("Mean Squared Error")

num_files = 2
file_names = [
    "results/problem_3/c/batch_size_1e+02_step_size_1e-03_ridge_coeff_1e-02.csv",
    "results/problem_3/c/batch_size_6e+04_step_size_1e-03_ridge_coeff_1e-02.csv"
]

labels = [
    r"SGD w/ Batch Size 100",
    r"Gradient Descent"
]

training_fig, training_ax = plt.subplots()
testing_fig, testing_ax = plt.subplots()

for i in range(num_files):
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
training_ax.legend()
testing_ax.legend()
training_ax.set_xlabel("Iterations Over Full Dataset")
testing_ax.set_xlabel("Iterations Over Full Dataset")
training_ax.set_ylabel("Mean Squared Error")
testing_ax.set_ylabel("Mean Squared Error")
#
#
plt.show()
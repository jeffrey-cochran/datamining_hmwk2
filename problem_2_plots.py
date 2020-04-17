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
# num_files = 6
# file_names = [
#     "results/problem_2/batches/step_5e-03_batch_10.csv",
#     "results/problem_2/batches/step_5e-03_batch_50.csv",
#     "results/problem_2/batches/step_5e-03_batch_100.csv",
#     "results/problem_2/batches/step_5e-03_batch_500.csv",
#     "results/problem_2/batches/step_5e-03_batch_1000.csv",
#     "results/problem_2/batches/step_5e-03_batch_5000.csv",
# ]

# labels = [
#     r"$Batch Size = 10$",
#     r"$Batch Size = 50$",
#     r"$Batch Size = 100$",
#     r"$Batch Size = 500$",
#     r"$Batch Size = 1000$",
#     r"$Batch Size = 5000$",
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
#     testing_data.plot(training_ax, linestyle="dashed", linewidth="2", color=colors[i])

# #
# # Set display params
# training_ax.legend()
# testing_ax.legend()
# training_ax.set_xlabel("Iterations Over Full Dataset")
# testing_ax.set_xlabel("Iterations Over Full Dataset")
# training_ax.set_ylabel("Mean Squared Error")
# testing_ax.set_ylabel("Mean Squared Error")
# plt.show()
#
# ==============================================
# plot step size comparison
# ==============================================
#
num_files = 5
file_names = [
    "results/problem_2/steps/step_1e-02_batch_100.csv",
    "results/problem_2/steps/step_5e-03_batch_100.csv",
    "results/problem_2/steps/step_1e-03_batch_100.csv",
    "results/problem_2/steps/step_5e-04_batch_100.csv",
    "results/problem_2/steps/step_1e-04_batch_100.csv"
]

labels = [
    r"$Step Size = 1e-2$",
    r"$Step Size = 5e-3$",
    r"$Step Size = 1e-3$",
    r"$Step Size = 5e-4$",
    r"$Step Size = 1e-4$"
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

plt.show()
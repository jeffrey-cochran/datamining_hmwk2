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
num_files = 3
file_names = [
    "results/problem_4/const_step_size.csv",
    "results/problem_4/decreasing_step_size.csv",
    "results/problem_4/cosine_step_size.csv"
]

labels = [
    r"$Constant \,\, \tau$",
    r"$Decreasing \,\, \tau$",
    r"$Cosine \,\, Annealed \,\, \tau$"
]

training_fig, training_ax = plt.subplots()

for i in range(num_files):
    #
    # Plot training data
    training_data = Series()
    training_data.load_from_file(
        file_names[i],
        labels[i],
        TRAIN
    )
    training_data.plot(training_ax, color=colors[i], use_log_y=False)
    #
    # Plot testing data
    testing_data = Series()
    testing_data.load_from_file(
        file_names[i],
        None,
        TEST
    )
    testing_data.plot(training_ax, linestyle="dashed", linewidth=2, color=colors[i], use_log_y=False)
training_ax.legend()
training_ax.set_xlabel("Iterations Over Full Dataset")
training_ax.set_ylabel("Mean Squared Error")

plt.show()
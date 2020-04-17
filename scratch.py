import csv

file_base = "results/problem_2/batches/step_5e-03_batch_"
file_ext = ".csv"
batch_sizes = ["10", "50", "100", "500", "1000", "5000"]

read_file_names = [file_base + batch_size + file_ext for batch_size in batch_sizes]
write_file_names = [file_base + batch_size + "_w" + file_ext for batch_size in batch_sizes]


for i in range(6): 
    with open(read_file_names[i], "w", newline="") as f_write:
        csv_writer = csv.writer(f_write, delimiter=",")
        with open(write_file_names[i], newline="") as f_read:
            csv_reader = csv.reader(f_read, delimiter=",")
            for row in csv_reader:
                csv_writer.writerow(row)
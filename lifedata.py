from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split
from neural_net_UCI_data import parse_line
from neural_net_UCI_data import normalize

with open("heart_failure_clinical_records_dataset.csv", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# print(training_data)
td = normalize(training_data)
# print(td)

train, test = train_test_split(td)

nn = NeuralNet(13, 3, 1)
nn.train(train, iters=10000, print_interval=1000, learning_rate=0.2)

for i in nn.test_with_expected(test):
    difference = round(abs(i[1][0] - i[2][0]), 3)
    print(f"desired: {i[1]}, actual: {i[2]} diff: {difference}")

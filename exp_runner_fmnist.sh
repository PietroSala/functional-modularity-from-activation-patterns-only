#! /bin/bash

source .venv/bin/activate
mkdir -p output/
screen -dmS fashion_mnist_a; screen -S fashion_mnist_a -X stuff "python experiments/run_experiment.py --name fashion_mnist_a 2>&1 > output/logs_fashion_mnist_a.txt &\n"
screen -dmS fashion_mnist_b; screen -S fashion_mnist_b -X stuff "python experiments/run_experiment.py --name fashion_mnist_b 2>&1 > output/logs_fashion_mnist_b.txt &\n"
screen -dmS fashion_mnist_c; screen -S fashion_mnist_c -X stuff "python experiments/run_experiment.py --name fashion_mnist_c 2>&1 > output/logs_fashion_mnist_c.txt &\n"
screen -dmS fashion_mnist_d; screen -S fashion_mnist_d -X stuff "python experiments/run_experiment.py --name fashion_mnist_d 2>&1 > output/logs_fashion_mnist_d.txt &\n"



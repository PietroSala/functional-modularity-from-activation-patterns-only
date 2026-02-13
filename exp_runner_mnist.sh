#! /bin/bash
    
# Example of running experiment_v0.py with different experiment templates
source .venv/bin/activate
mkdir -p output/
screen -dmS mnist_a; screen -S mnist_a -X stuff "python experiments/run_experiment.py --name mnist_a 2>&1 > output/logs_mnist_a.txt &\n"
screen -dmS mnist_b; screen -S mnist_b -X stuff "python experiments/run_experiment.py --name mnist_b 2>&1 > output/logs_mnist_b.txt &\n"
screen -dmS mnist_c; screen -S mnist_c -X stuff "python experiments/run_experiment.py --name mnist_c 2>&1 > output/logs_mnist_c.txt &\n"
screen -dmS mnist_d; screen -S mnist_d -X stuff "python experiments/run_experiment.py --name mnist_d 2>&1 > output/logs_mnist_d.txt &\n"

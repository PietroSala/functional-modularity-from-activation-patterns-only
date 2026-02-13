# Define your metrics
from ax.core import Metric


class Accuracy(Metric):
    def __init__(self, name, properties = None):
        super().__init__(name, False, properties)
    
    def fetch_trial_data(self, trial):
        print(trial)
        # Your code to evaluate validation accuracy for the trial's arm parameters
        pass

class Modularity(Metric):
    def __init__(self, name, properties = None):
        super().__init__(name, False, properties)
    
    def fetch_trial_data(self, trial):
        print(trial)
        # Your code to evaluate model size or another competing metric
        pass

class ModelSize(Metric):
    def __init__(self, name, properties = None):
        super().__init__(name, True, properties)
    
    def fetch_trial_data(self, trial):
        print(trial)
        # Your code to evaluate model size or another competing metric
        pass
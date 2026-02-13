import json
import os
from ax.core import OutcomeConstraint
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.early_stopping.strategies import threshold
from ax.service.managed_loop import optimize
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.core.objective import Objective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.metric import Metric
from ax.core.objective import MultiObjective
from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep
from ax.adapter.registry import Models, Generators
import traceback



from concurrent import futures
from dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.metamodel.metamodelconfig import MetaModelConfig, MetaModelTemplate, TemplateFashionMnist
from modularnet.metamodel.metrics import Accuracy
from modularnet.modularspace import ModularSpace
from modularnet.modularstep import ModularStep



from skimage.util import noise
from utils.logger import Logger
from utils.utils import seed_everything, session_path

def test():
    template = TemplateFashionMnist("test_experiment")
    exp = Experiment("test_experiment", template)
    exp.run()


class Experiment:
    OUTPUT_DIR = 'output/'
    TRIAL_DIR = 'trials/'
    CONFIG_DIR = 'config/'
    LOG_DIR = 'logs/'
    REPORT_DIR = 'reports/'
    

    def __init__(self, name, template:MetaModelTemplate, seed = None, debug=False, n_trials=10, parallel_trials=1, draft=False, device='cuda', save_video_space=False, save_history=False):
        self.client = AxClient()

        self.signature = template.signature()
        #self.base_path = session_path(name, uid=self.signature, force=True)
        self.seed = seed if seed is not None else 42
        self.name = name
        self.dataset = None
        self.template = template
        self.results = []
        self.best_results = None
        self.trial_num = 0
        self.logger = Logger(self.path_logs(), filename=self.name)
        self.debug = debug
        self.n_trials = n_trials
        self.objectives = None
        self.parallel_trials = parallel_trials
        self.draft = draft
        self.device = device
        self.save_video_space = save_video_space
        self.save_history = save_history


    def makedirs(self, path):
        path = path.strip()

        if os.path.isfile(path):
            path = os.path.dirname(path)
        elif not os.path.exists(path) and not path.endswith('/'):
            path = os.path.dirname(path)
            
        os.makedirs( path, exist_ok=True)

    def path_home(self,rel_path=None, makedirs=True):
        exp_uid = self.name + "_" + self.signature # cache by template ( any change => new template => new signature => new experiment )
        path = os.path.join(Experiment.OUTPUT_DIR, exp_uid )
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path

    def path_trial(self, trial_num, rel_path=None, makedirs=True):
        path = self.path_home(f'{Experiment.TRIAL_DIR}/t{trial_num}/')
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path

    def path_config(self,rel_path=None, makedirs=True):
        path = self.path_home(Experiment.CONFIG_DIR)
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path
    
    def path_logs(self,rel_path=None, makedirs=True):
        path = self.path_home(Experiment.LOG_DIR)
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path
    
    def path_report(self,rel_path=None, makedirs=True):
        path = self.path_home(Experiment.REPORT_DIR)
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path

    def path_snapshot(self):
        return self.path_config('ax_client_snapshot.json')  

    def save(self):
        path_snapshot = self.path_snapshot()
        self.client.save_to_json_file( path_snapshot )

    def load(self):
        path_snapshot = self.path_snapshot()
        if not os.path.exists( path_snapshot ): return
        self.client = AxClient.load_from_json_file( path_snapshot )

    def init_client(self):
        params = self.template.get_params()
        
        
        min_acc = self.template.params.get('nas_min_acc', None) 
        if min_acc is None: 
            print("Warning: accuracy metric enabled but no minimum accuracy specified: using default 0.8")
            min_acc = 0.8
        else:
            min_acc = float(min_acc['value'])
        
        objectives = {
            'val_acc': ObjectiveProperties(minimize=False, threshold=min_acc), 
        }

        if self.template.params.get('exp_metric_modularity', False):
            min_mod_score = self.template.params.get('nas_min_mod_score', None)
            if min_mod_score is None: 
                print("Warning: modularity metric enabled but no minimum modularity score specified: using default 0.2")
                min_mod_score = 0.2
            else:
                min_mod_score = float(min_mod_score['value'])
            objectives['modularity_score'] = ObjectiveProperties(minimize=False, threshold=min_mod_score) 


        self.client.create_experiment( 
            name=self.name, 
            parameters=params,
            objectives=objectives,
        )
        self.load()
       

    def result_analysis(self):
        print("Result Analysis")
        self.init_client()
        cards = self.client.compute_analyses()
        
        flat_cards = []
        flat_cards.extend([card.flatten() for card in cards])
        flat_cards = flat_cards[0]
        for card in flat_cards:
            card_name = card.name.replace(" ", "_").lower() 
            basedir = self.path_report(card_name + '/')
            print(basedir)

            filename_txt = os.path.join(basedir, f"{card_name}.txt")
            with open(filename_txt, 'w') as f: f.write(f"{card.name}\n\n{card.title}\n\n{card.subtitle}")
            
            filename_content = os.path.join(basedir, f"{card_name}.json")
            with open(filename_content, 'w') as f: f.write(card.blob)

            if hasattr(card, 'get_figure'):
                filename_fig = os.path.join(basedir, f"{card_name}.jpg")
                fig = card.get_figure()
                img = fig.to_image(format="jpg")
                with open(filename_fig, 'wb') as f: f.write(img)
                
            


    def run(self):
        self.init_client()
        self.trial_num = len( self.client.experiment.trials )

        results = []
        for _ in range(self.n_trials):
            trials, stop = self.client.get_next_trials(max_trials=self.parallel_trials)
            for trial_index, parameters in trials.items():
                
                print("#"*80)
                print(f"Trial {trial_index} with parameters:")
                print("#"*80)
                print(json.dumps(parameters, indent=2))
                
                try:
                    metric, model = self.train_evaluate(parameters, trial_index)
                    results.append(metric)
                    self.client.complete_trial(trial_index=trial_index, raw_data=metric)
                    model.save()
                    self.save_results()
                except Exception as e:
                    print(f"[ERROR] Exception during trial {trial_index}: {e}")
                    print(traceback.format_exc())
                    


            if stop: break



    def save_results(self):
        self.best_results = self.client.get_pareto_optimal_parameters()
        print(self.best_results)
        content = json.dumps(self.best_results, indent=2, sort_keys=True)
        with open( self.path_home('pareto_optimal_results.json'), 'w') as f: f.write(content)
        self.save()
        
    def train_evaluate(self, options, trial_num):
        # Code to load the experiment
        config = MetaModelConfig(options)

        seed_everything(self.seed)
        name = f'{self.name}_t{trial_num}'
        path_trial = self.path_trial(trial_num)
        
        # Create model first to determine device
        self.model = MetaModel(self.name, config, device=self.device)

        if self.debug: self.model.print()

        device  = self.model.device
        # Pass model device to dataset
        self.dataset = DataProvider.get_dataprovider(config, device=device)
        
        space_size = config.modular_space_size
        # Pass model device to space
        self.space = ModularSpace( name, self.model, space_size, useActivations=True, useLayers=True, base_path=path_trial, device=device, history=self.save_history )
        
        steps = ModularStep(config, name, path_trial, self.dataset, self.model, self.space)

        steps.space.rescale_factor = 0.9
        steps.space.modularizing = config.exp_modularizer

                
        steps.step_run(draft = self.draft, save_video=self.save_video_space)

        metrics = {k:v[-1] for k,v in steps.metrics.items()} 
        metrics['model_size'] = self.model.get_model_size()
        print(metrics)
        return metrics, self.model
    
    def report(self):
        pass



if __name__ == "__main__": test()







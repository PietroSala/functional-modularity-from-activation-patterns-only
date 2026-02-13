
import argparse
from modularnet.metamodel.experiment import Experiment

from modularnet.metamodel.metamodelconfig import TemplateFashionMnist



    

def main():
    argparser = argparse.ArgumentParser(description="Run ModularNet Experiment")
    argparser.add_argument('--name', type=str, default='mnist_test_d', choices=TemplateFashionMnist.VARIANTS, help='Experiment template to use')
    args = argparser.parse_args()
    
    template = TemplateFashionMnist.get_experiment_variant(args.name)
    exp = Experiment(args.name, template, parallel_trials=1, debug=False, n_trials=50, draft=False, device='cuda')
    exp.run()


if __name__ == "__main__": 
    main()
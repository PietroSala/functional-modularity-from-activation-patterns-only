
import argparse
from modularnet.metamodel.experiment import Experiment

from modularnet.metamodel.metamodelconfig import TemplateFashionMnist
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from SALib.analyze import sobol
from sklearn.ensemble import RandomForestRegressor

def main():
    argparser = argparse.ArgumentParser(description="Run ModularNet Experiment")
    argparser.add_argument('--name', type=str, default='mnist_a', choices=TemplateFashionMnist.VARIANTS, help='Experiment template to use')
    args = argparser.parse_args()
    
    template = TemplateFashionMnist.get_experiment_variant(args.name)
    exp = Experiment(args.name, template, parallel_trials=1, debug=True, n_trials=50, draft=False, device='cuda')
    
    exp.result_analysis()

    reporter = AxReporter(exp)
    reporter.run()

class AxReporter:
    def __init__(self, exp:Experiment, report_folder: str = None):
        report_folder = exp.path_report() if report_folder is None else report_folder
        self.exp = exp
        self.ax_client = exp.client
        self.ax_exp = self.ax_client.experiment
        self.folder = report_folder
        os.makedirs(self.folder, exist_ok=True)
        self.parameters = [p.name for p in self.ax_exp.search_space.parameters]
        # Ensure self.metrics gives all metric names you care about
        self.metrics = [self.ax_exp.optimization_config.objective.metric.name]  # extend as needed
    
    def _extract_trials_to_df(self):
        # Replace with your robust DataFrame extraction logic!
        rows = []
        for trial in self.ax_exp.trials.values():
            if trial.status.is_completed:
                row = trial.arm.parameters.copy()
                # Get metrics
                for m in self.metrics:
                    try:
                        val = trial.fetch_data().df.query("metric_name==@m")['mean']
                        if not val.empty:
                            row[m] = val.values[0]
                    except Exception: pass
                rows.append(row)
        return pd.DataFrame(rows)

    def plot_parallel_coordinates(self):
        df = self._extract_trials_to_df()
        fig = px.parallel_coordinates(df, dimensions=df.columns, color=df[self.metrics[0]])
        fig.write_html(os.path.join(self.folder, "parallel_coordinates.html"))
        fig.write_image(os.path.join(self.folder, "parallel_coordinates.png"))
        fig.show()

    def plot_correlation_heatmap(self):
        df = self._extract_trials_to_df()
        corr = df.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, "correlation_heatmap.png"))
        plt.show()

    def plot_optimization_trace(self):
        metric = self.metrics[0]
        df = self._extract_trials_to_df()
        best_so_far = df[metric].cummax()
        plt.figure(figsize=(8,5))
        plt.plot(best_so_far, label='Best so far')
        plt.xlabel("Trial")
        plt.ylabel(f"Best {metric}")
        plt.title("Optimization Trace")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, "optimization_trace.png"))
        plt.show()

    def plot_hyperparameter_importance(self):
        df = self._extract_trials_to_df()
        X = df[self.parameters]
        y = df[self.metrics[0]]
        model = RandomForestRegressor().fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances.sort_values().plot(kind="barh", figsize=(7,6), title="Hyperparameter Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, "hyperparam_importances.png"))
        plt.show()

    def plot_sobol_sensitivity(self):
        df = self._extract_trials_to_df()
        params = self.parameters
        bounds = [[float(df[p].min()), float(df[p].max())] for p in params]
        problem = {'num_vars': len(params), 'names': params, 'bounds': bounds}
        Y = df[self.metrics[0]].values
        Si = sobol.analyze(problem, Y, print_to_console=False)
        resdf = pd.DataFrame({"Parameter": params, "S1": Si['S1'], "ST": Si['ST']})
        resdf.set_index("Parameter")[["S1", "ST"]].plot(kind="bar", figsize=(7,6), title="Sobol Sensitivity")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, "sobol_sensitivity.png"))
        plt.show()
        resdf.to_csv(os.path.join(self.folder, "sobol_sensitivity.csv"), index=False)

    def save_metric_summary_table(self):
        df = self._extract_trials_to_df()
        stats = df.describe()
        stats.to_csv(os.path.join(self.folder, "metric_summary.csv"))
        print(stats)

    def plot_parallel_coords_highlight(self, top_n=5):
        df = self._extract_trials_to_df()
        metric = self.metrics[0]
        top = df.nlargest(top_n, metric)
        bottom = df.nsmallest(top_n, metric)
        combined = pd.concat([top, bottom])
        fig = px.parallel_coordinates(combined, dimensions=combined.columns, color=combined[metric])
        fig.write_html(os.path.join(self.folder, "parallel_coords_highlight.html"))
        fig.write_image(os.path.join(self.folder, "parallel_coords_highlight.png"))
        fig.show()

    def plot_pareto_front(self):
        if len(self.metrics) >= 2:
            df = self._extract_trials_to_df()
            fig = px.scatter(df, x=self.metrics[0], y=self.metrics[1], color=df[self.metrics[1]])
            fig.update_layout(title="Pareto Front")
            fig.write_html(os.path.join(self.folder, "pareto_front.html"))
            fig.write_image(os.path.join(self.folder, "pareto_front.png"))
            fig.show()

    def plot_trial_status_bar(self):
        statuses = [t.status.name for t in self.ax_exp.trials.values()]
        counts = pd.Series(statuses).value_counts()
        counts.plot(kind="bar", figsize=(7,5), title="Trial Status Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, "trial_status_bar.png"))
        plt.show()

    def plot_metric_box_grouped(self, group_by: str):
        df = self._extract_trials_to_df()
        metric = self.metrics[0]
        plt.figure(figsize=(7,5))
        sns.boxplot(x=df[group_by], y=df[metric])
        plt.title(f"{metric} Distribution by {group_by}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, f"{metric}_by_{group_by}_box.png"))
        plt.show()

    def run(self):
        self.plot_parallel_coordinates()
        self.plot_correlation_heatmap()
        self.plot_optimization_trace()
        self.plot_hyperparameter_importance()
        self.plot_sobol_sensitivity()
        self.save_metric_summary_table()
        self.plot_parallel_coords_highlight()
        self.plot_pareto_front()
        self.plot_trial_status_bar()
        # Add group metric box plots for the most interesting hyperparameters:
        for param in self.parameters:
            self.plot_metric_box_grouped(param)



if __name__ == "__main__": 
    main()
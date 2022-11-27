import os
from plotnine import *
import pandas as pd
import requests
import re


def main():
    for name, benchmark_data_path in load_benchmarks():
        df = pd.read_csv(benchmark_data_path)
        df['Population size'] = pd.Categorical(df['population_size'])
        print(f"Plotting {name}")
        print(df)
        plt = ggplot(df, aes(x="Iterations", y="Markets visited")) + \
            scale_x_log10(name="Iterations per run") +\
            geom_jitter(width=0.5, height=0.2) +\
            geom_smooth(span=0.75)# geom_jitter(height=0.05) +\
        print(plt)
        # run(name, market_data_path, durations_data_path,
        #     save_runs=True, show_route=False, show_plot_runs=False, save_plot_runs=True)


def load_benchmarks():
    base_path = os.path.join("data", "runs")
    dataset_names = list()
    benchmark_data_paths = list()

    for file in os.listdir(base_path):
        stem, ext = os.path.splitext(file)

        dataset_names.append(stem)
        benchmark_data_paths.append(os.path.join(base_path, file))
    return zip(dataset_names, benchmark_data_paths)


if __name__ == '__main__':
    main()

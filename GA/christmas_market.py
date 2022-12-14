import datetime
from multiprocessing import Pool
from dataclasses import dataclass
import signal
import threading
from time import sleep
from typing import Optional, Sequence, Tuple
from sko.GA import GA_TSP
from nptime import nptime as time
from datetime import timedelta
import pandas as pd
import os
from os import PathLike
import humanfriendly
import numpy as np
import matplotlib.pyplot as plt

TIME_PER_MARKET = timedelta(minutes=30).seconds
NEED_FULL_STAY_AT_LAST_MARKET = False
NAME_INDEX = 0
OPENING_INDEX = 1
CLOSING_INDEX = 2
PRINT_ITERATIONS_AND_RUNS = False


def parse_time_as_seconds(s: str) -> int:
    h, m = map(int, s.strip().split(":"))
    return m * 60 + h * 3600


def format_time(secs: int) -> str:
    hours = np.floor(secs / 3600)
    secs = np.floor(secs / 60) - hours * 60
    return f"{hours:02.0f}:{secs:02.0f}"


def format_duration(secs: int) -> str:
    return humanfriendly.format_timespan(timedelta(seconds=float(secs)))


def load_markets(path: PathLike) -> pd.DataFrame:
    @dataclass
    class Market:
        name: str
        opening: int
        closing: int

    df = pd.read_csv(path)
    markets = list()
    for i in range(len(df)):
        name = df.loc[i, "name"]
        opening = parse_time_as_seconds(df.loc[i, "opening"])
        closing = parse_time_as_seconds(df.loc[i, "closing"])
        markets.append(Market(name, opening, closing))
    df = pd.DataFrame(markets)
    df.set_index("name", drop=False, inplace=True)
    return df


def load_distance_matrix(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.set_index("name", inplace=True)
    return df


@dataclass
class PerformRunsResult:
    average_markets_visited: float
    average_time_wasted: float
    best_route_markets_visited: int
    best_route_time_wasted: int
    best_route: Sequence[int]


def perform_runs(markets, durations, max_iter: int, num_runs: int) -> PerformRunsResult:
    first_opening_time = min(markets[:, OPENING_INDEX])
    last_closing_time = max(markets[:, CLOSING_INDEX])

    def christmas_market(p) -> Tuple[int, int]:
        start_time = markets[p[0], OPENING_INDEX]
        current_time = start_time
        markets_visited = 0

        # walk along route
        for i in range(len(p)):
            last_exit_time = current_time

            current_market = p[i]
            opening_time = markets[current_market, OPENING_INDEX]
            closing_time = markets[current_market, CLOSING_INDEX]

            if i > 0:
                # walk to next destination
                last_market = p[i-1]
                walking_time = durations[last_market, current_market]
                current_time += walking_time

            # wait until market is opening
            if current_time < opening_time:
                current_time = opening_time

            if current_time >= closing_time:
                current_time = last_exit_time
                break

            # stay for 30 mins or until market is closing
            current_time += TIME_PER_MARKET
            if current_time > closing_time:
                if NEED_FULL_STAY_AT_LAST_MARKET:
                    break
                else:
                    current_time = closing_time
                    markets_visited += 1
                    break

            markets_visited += 1

        end_time = current_time

        time_wasted = (last_closing_time - first_opening_time) - \
            TIME_PER_MARKET * markets_visited

        return markets_visited, time_wasted

    ga = GA_TSP(func=lambda p: christmas_market(
        p)[1], n_dim=len(markets), size_pop=50, prob_mut=1)

    average_markets_visited = 0.0
    average_time_wasted = 0.0
    best_route_markets_visited = 0
    best_route_time_wasted = 0
    best_route = []

    for i in range(num_runs):
        if PRINT_ITERATIONS_AND_RUNS:
            print(f"Iterations: {max_iter}, Run {i+1}/{num_runs}")
        route, _ = ga.run(max_iter)
        current_markets_visited, current_time_wasted = christmas_market(
            route)
        average_markets_visited += current_markets_visited
        average_time_wasted += current_time_wasted
        if current_markets_visited > best_route_markets_visited:
            best_route_markets_visited = current_markets_visited
            best_route_time_wasted = current_time_wasted
            best_route = route

    average_markets_visited /= num_runs
    average_time_wasted /= num_runs

    return PerformRunsResult(average_markets_visited, average_time_wasted, best_route_markets_visited, best_route_time_wasted, best_route)


def print_route(route, markets, durations):
    current_time = markets[route[0], OPENING_INDEX]
    print(f"1. {format_time(current_time)} - {markets[route[0], NAME_INDEX]}")
    for i in range(1, len(route)):
        opening_time = markets[route[i], OPENING_INDEX]
        closing_time = markets[route[i], CLOSING_INDEX]

        walking_time = durations[route[i - 1], route[i]]

        current_time += walking_time

        if current_time < opening_time:
            current_time = opening_time

        if current_time > closing_time:
            break

        if current_time + TIME_PER_MARKET > closing_time and NEED_FULL_STAY_AT_LAST_MARKET:
            break

        print(f"\tWalking {format_duration(walking_time)}")
        print(
            f"{i+1}. {format_time(current_time)} - {markets[route[i], NAME_INDEX]}")

        current_time += TIME_PER_MARKET

        if current_time > closing_time:
            break


def plot_runs(max_iters: Sequence[int], markets_visited: Sequence[float], show: bool, save_path: Optional[str] = None):
    # plot results
    fig, ax1 = plt.subplots()
    ax1.set_title("Time-constrained travelling Christmas market visitor")
    # fig.tight_layout()

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Markets visited")
    ax1.semilogx(max_iters, markets_visited)

    # ax2 = ax1.twinx()
    # ax2.set_xlabel("Iterations")
    # ax2.set_ylabel("Time wasted")
    # ax2.semilogx(max_iters, time_wasted
    # )
    if save_path is not None:
        plt.savefig(save_path)
        
    if show:
        plt.show()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run(dataset_name: str, market_data_path: PathLike, durations_data_path: PathLike,
        save_runs: bool, print_route: bool, save_plot_runs: bool, show_plot_runs: bool):
    # Load market list and distance matrix
    markets = np.array(load_markets(market_data_path))
    durations = np.array(load_distance_matrix(
        durations_data_path), dtype=np.int32)

    # run algorithm with different number of iterations
    max_iters = np.geomspace(1, 4000, num=50, dtype=np.int32)
    num_runs_per_iter = np.full_like(max_iters, 10)

    start = datetime.datetime.now()

    with Pool(os.cpu_count()-1, init_worker) as executor:
        results = executor.starmap_async(perform_runs, zip(
            np.full((max_iters.shape[0], markets.shape[0],
                    markets.shape[1]), markets, dtype="object"),
            np.full((max_iters.shape[0], durations.shape[0],
                    durations.shape[1]), durations, dtype="object"),
            reversed(max_iters),
            num_runs_per_iter
        ), chunksize=1)
        try:
            while not results.ready():
                sleep(0.5)
        except KeyboardInterrupt:
            executor.terminate()
            executor.join()
            exit()
        else:
            executor.close()
            executor.join()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    #     results = executor.map(perform_runs, max_iters, num_runs_per_iter)
    # results = map(perform_runs, max_iters, num_runs_per_iter)
    results = list(reversed(list(results.get())))

    end = datetime.datetime.now()
    print(f"Ran for {format_duration((end-start).seconds)}")

    best = max(results, key=lambda x: x.best_route_markets_visited)
    markets_visited = np.array([t.average_markets_visited for t in results])

    if save_runs:
        # save results to csv file
        df = pd.DataFrame(np.transpose(np.vstack((max_iters, markets_visited))), columns=[
            "Iterations", "Markets visited"])
        os.makedirs(os.path.join("data", "runs"), exist_ok=True)
        df.to_csv(os.path.join("data", "runs", f"{dataset_name}_runs.csv"), header=True, index=False)

    if show_plot_runs or save_plot_runs:
        figure_path = None
        if save_plot_runs:
            os.makedirs(os.path.join("data", "figures"), exist_ok=True)
            figure_path = os.path.join("data", "figures", f"{dataset_name}_figure.png")
            
        # plot best visited markets against iterations
        plot_runs(max_iters, markets_visited, show_plot_runs, figure_path)

    if print_route:
        # print best route
        best_route = best.best_route
        assert(np.unique(best_route).shape == best_route.shape)
        print(f"Best route visites {np.ceil(best_route.shape[0])} markets")
        print_route(best_route, markets, durations)


def main_single():
    market_data_path = os.path.join("data", "tcmt_data.csv")
    durations_data_path = os.path.join("data", "tcmt_durations.csv")
    run("tcmt_default", market_data_path, durations_data_path,
        save_runs=True, print_route=True, show_plot_runs=True, save_plot_runs=True)

def main_batch():
    base_path = os.path.join("data", "tcmt_instances")
    dataset_names = list()
    market_data_paths = list()
    durations_data_paths = list()
    
    for file in os.listdir(base_path):
        if "durations" in file:
            continue
        
        stem, ext = os.path.splitext(file)

        dataset_names.append(stem)
        market_data_paths.append(os.path.join(base_path, file))
        durations_data_paths.append(os.path.join(base_path, f"{stem}_durations.csv"))
    
    for name, market_data_path, durations_data_path in zip(dataset_names, market_data_paths, durations_data_paths):
        print(f"Processing {name}")
        run(name, market_data_path, durations_data_path,
            save_runs=True, print_route=False, show_plot_runs=False, save_plot_runs=True)

def main():
    main_batch()

if __name__ == '__main__':
    main()

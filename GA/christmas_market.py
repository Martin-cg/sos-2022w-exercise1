from multiprocessing import Pool
from dataclasses import dataclass
from typing import Tuple
from sko.GA import GA_TSP
from nptime import nptime as time
from datetime import timedelta;
import pandas as pd;
import os
import humanfriendly
import numpy as np
import matplotlib.pyplot as plt
from object_pool import ObjectPool

TIME_PER_MARKET = timedelta(minutes=30).seconds
NEED_FULL_STAY_AT_LAST_MARKET = False

def parse_time(s: str) -> time:
    h, m = map(int, s.strip().split(":"))
    return time(hour=h, minute=m)

def format_time(secs: int) -> str:
    hours = np.floor(secs / 3600)
    secs = np.floor(secs / 60) - hours * 60
    return f"{hours:02.0f}:{secs:02.0f}"

def format_duration(secs: int) -> str:
    return humanfriendly.format_timespan(timedelta(seconds=float(secs)))

def load_markets() -> pd.DataFrame:
    @dataclass
    class Market:
        name: str
        opening: int
        closing: int

    df = pd.read_csv(os.path.join("data", "tcmt_data.csv"))
    markets = list()
    for i in range(len(df)):
        name = df.loc[i, "name"]
        opening = parse_time(df.loc[i, "opening"])
        opening = opening.to_timedelta().seconds
        closing = parse_time(df.loc[i, "closing"])
        closing = closing.to_timedelta().seconds
        markets.append(Market(name, opening, closing))
    df = pd.DataFrame(markets)
    df.set_index("name", drop=False, inplace=True)
    return df

def load_distance_matrix() -> pd.DataFrame:
    df = pd.read_csv(os.path.join("data", "tcmt_durations.csv"))
    df.set_index("name", inplace=True)
    # df = df.applymap(lambda s: timedelta(seconds=s))
    return df

def main():
    # Load market list and distance matrix
    markets = np.array(load_markets())
    durations = np.array(load_distance_matrix(), dtype=np.int32)

    name_loc = 0
    opening_loc = 1
    closing_loc = 2

    first_opening_time = min(markets[:, opening_loc])
    last_closing_time = max(markets[:, closing_loc])

    def christmas_market(p) -> Tuple[int, int]:
        start_time = markets[p[0], opening_loc]
        current_time = start_time
        markets_visited = 0
        
        # walk along route
        for i in range(len(p)):
            last_exit_time = current_time
            
            current_market = p[i]
            opening_time = markets[current_market, opening_loc]
            closing_time = markets[current_market, closing_loc]
            
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
        
        time_wasted = (last_closing_time - first_opening_time) - TIME_PER_MARKET * markets_visited
        
        return markets_visited, time_wasted

    def create_solver() -> GA_TSP:
        print("create_solver()")
        ga = GA_TSP(func=lambda p: christmas_market(p)[1], n_dim=len(markets), size_pop=50, prob_mut=1)
        return ga

    pool = ObjectPool(create_solver)
    
    # run algorithm with different number of iterations
    max_iters = np.geomspace(1, 1000, num=5, dtype=np.int32)
    markets_visited = np.zeros_like(max_iters, dtype=np.float32)
    time_wasted = np.zeros_like(max_iters, dtype=np.float32)
    num_runs_per_iter = np.full_like(max_iters, 5)

    def perform_runs(max_iter: int, num_runs: int) -> Tuple[float, float]:
        avg_markets_visited = 0.0
        avg_time_wasted = 0.0

        with pool.item() as ga:
            for i in range(num_runs):
                print(f"Iterations: {max_iter}, Run {i+1}/{num_runs}")
                route, _ = ga.run(max_iter)
                current_market_visited, current_time_wasted = christmas_market(route)
                avg_markets_visited += current_market_visited
                avg_time_wasted += current_time_wasted
            
        avg_markets_visited /= num_runs
        avg_time_wasted /= num_runs
        
        return (avg_markets_visited, avg_time_wasted)

    import datetime
    start = datetime.datetime.now()
    # with Pool(5) as executor:
    #     results = executor.map(perform_runs, zip(max_iters, num_runs_per_iter))
    # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    #     results = executor.map(perform_runs, max_iters, num_runs_per_iter)
    results = map(perform_runs, max_iters, num_runs_per_iter)
    results = list(results)
    end = datetime.datetime.now()
    print(end-start)
    best = max(results, key=lambda x: x[0])
    markets_visited = [t[0] for t in results]
    time_wasted = [t[1] for t in results]

    # save results to csv file
    df = pd.DataFrame(np.transpose(np.vstack((max_iters, markets_visited, time_wasted))), columns=[
                    "Iterations", "Markets visited", "Time wasted"])
    df.to_csv(os.path.join("data", "runs.csv"), header=True, index=False)


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
    # ax2.semilogx(max_iters, time_wasted)

    plt.show()


    # print best route
    best_route = best[1]
    print(f"Best route: {best_route}")
    assert(np.unique(best_route).shape == best_route.shape)

    current_time = markets[best_route[0], opening_loc]
    print(f"1. {format_time(current_time)} - {markets[best_route[0], name_loc]}")
    for i in range(1, len(best_route)):
        opening_time = markets[best_route[i], opening_loc]
        closing_time = markets[best_route[i], closing_loc]
        
        walking_time = durations[best_route[i - 1], best_route[i]]
        
        current_time += walking_time
        
        if current_time < opening_time:
            current_time = opening_time

        if current_time > closing_time:
            break

        if current_time + TIME_PER_MARKET > closing_time and NEED_FULL_STAY_AT_LAST_MARKET:
            break
        
        print(f"\tWalking {format_duration(walking_time)}")
        print(
            f"{i+1}. {format_time(current_time)} - {markets[best_route[i], name_loc]}")

        current_time += TIME_PER_MARKET
        
        if current_time > closing_time:
            break

if __name__ == '__main__':
    main()

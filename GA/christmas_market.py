from dataclasses import dataclass
from sko.GA import GA_TSP
from nptime import nptime as time
from datetime import timedelta;
import pandas as pd;
import os
import humanfriendly
import numpy as np
import matplotlib.pyplot as plt

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

# Load market list and distance matrix
markets = np.array(load_markets())
durations = np.array(load_distance_matrix(), dtype=np.int32)

name_loc = 0
opening_loc = 1
closing_loc = 2

first_opening_time = min(markets[:, opening_loc])
last_closing_time = max(markets[:, closing_loc])

def christmas_market(p):
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


# define genetic algorithm
ga = GA_TSP(func=lambda p: christmas_market(p)[1], n_dim=len(markets), size_pop=50, prob_mut=1)


# run algorithm with different number of iterations
max_iters = np.geomspace(1, 4000, num=50, dtype=np.int32)
markets_visited = np.zeros_like(max_iters, dtype=np.int32)

best = (0, [])
for i, max_iter in enumerate(max_iters):
    print(max_iter)
    route, _ = ga.run(max_iter)
    markets_visited[i] = christmas_market(route)[0]
    if markets_visited[i] > best[0]:
        best = (markets_visited[i], route)


# save results to csv file
df = pd.DataFrame(np.transpose(np.vstack((max_iters, markets_visited))), columns=[
                  "Iterations", "Markets visited"])
df.to_csv(os.path.join("data", "runs.csv"), header=True, index=False)


# plot results
plt.figure()
plt.semilogx(max_iters, markets_visited)
plt.title("Time-constrained travelling Christmas market visitor")
plt.xlabel("Iterations")
plt.ylabel("Markets visited")
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


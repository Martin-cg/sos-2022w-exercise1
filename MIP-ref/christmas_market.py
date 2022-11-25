from dataclasses import dataclass
from sko.GA import GA_TSP
from sko.tools import func_transformer
from nptime import nptime as time
from datetime import timedelta
import pandas as pd
import os
import humanfriendly
import numpy as np
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp
from ortools.linear_solver import pywraplp
import sys
import json
import os
import re

TIME_PER_MARKET = timedelta(minutes=30).seconds
NEED_FULL_STAY_AT_LAST_MARKET = True


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

    df = pd.read_csv(os.path.join("../data", "tcmt_data.csv"))
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
    df = pd.read_csv(os.path.join("../data", "tcmt_durations.csv"))
    df.set_index("name", inplace=True)
    # df = df.applymap(lambda s: timedelta(seconds=s))
    return df


# Load market list and distance matrix
markets = np.array(load_markets())
durations = np.array(load_distance_matrix(), dtype=np.int32)
n_markets = len(markets)

name_loc = 0
opening_loc = 1
closing_loc = 2

first_opening_time = min(markets[:, opening_loc])
last_closing_time = max(markets[:, closing_loc])

model = cp_model.CpModel()
# manager = pywrapcp.RoutingIndexManager(n_markets, 1, 0)
# routing = pywrapcp.RoutingModel(manager)


# TODO: Modelling idea
# constrain the model to visit as many markets as possible
# say visiting time of a market must be more than visiting time of the prior one + 30 min + waytime
# allow visits only during open hours

x = {}
for i in range(n_markets):
    for j in range(n_markets):

        x[(i, j)] = model.NewBoolVar('x_i%ij%i' % (i, j))

u = {}
for i in range(n_markets):
    u[i] = model.NewIntVar(0, n_markets - 1, 'u_i%i' % i)

model.AddAllDifferent(u[i] for i in range(n_markets))





for i in range(n_markets):
    model.AddElement()
# for i in range(n_markets):
#     model.Add(sum(u[i] == u[j] for j in range(n_markets)) == 1)
#
#
# for i in range(n_markets):
#     model.Add(sum(x[(i, j)] for j in range(n_markets)) == 1)
# for j in range(n_markets):
#     model.Add(sum(x[(i, j)] for i in range(n_markets)) == 1)
# for i in range(n_markets):
#     model.Add(x[(i, i)] == 0)

# # Constraint (2)
# # The total size of the tasks each worker takes on is at most capacity of agent.
# for market in range(n_agents):
#     solver.Add(
#         sum(usage[market][task] * x[market, task]
#             for task in range(n_tasks)) <= capacity[market])
#
# # Constraint (3)
# # Each task is assigned to exactly one agent.
# for j in range(n_tasks):
#     solver.Add(solver.Sum([x[i, j] for i in range(n_agents)]) == 1)

# Objective (1)
# Add all agent/tasks combination to objective.
objective_terms = []
print(u)
model.Minimize(sum(durations[i][j] for k in range(1, n_markets) for i in range(n_markets) for j in range(n_markets) if
                   u[k] == i and u[k - 1] == j))

model.Minimize(model.Sum(objective_terms))
model.set_time_limit(int(60) * 1000)

status = model.Solve()
if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    result = []
    for market in range(n_markets):
        result.append(market)
    print(result, end=";")
    print(f'{model.Objective().Value()}', end=";")
    print(f'{model.WallTime() / 1000}', end=";")
    print(status == cp_model.OPTIMAL)
    print(x[(3, 2)])
else:
    print('No solution found.')

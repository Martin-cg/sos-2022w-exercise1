import numpy as np
from scipy import spatial
from sko.GA import GA_TSP
from sko.operators import ranking, selection, crossover, mutation
import pandas as pd
from datetime import datetime

market_time = 30 * 60


def main():
    pts = pd.read_csv("../data/tcmt_coordinates.csv")
    pts['opening'] = pts['opening'].apply(time_to_secs)
    pts['closing'] = pts['closing'].apply(time_to_secs)
    # num_points, points_coordinate, duration_matrix, cal_total_distance = function_for_TSP(50)
    num_points = len(pts)
    duration_matrix = pd.read_csv("../data/tcmt_durations.csv").drop('name', axis=1).to_numpy()
    start_time = pts['opening'].min()
    ending_time = pts['closing'].max()

    def time_spend_waiting(routine):
        current_time = start_time
        waiting_time = 0
        num_pts, = routine.shape

        for i in range(num_pts):
            m = routine[i]
            if i > 0:  # walk to next destination
                walking_time = duration_matrix[routine[i - 1]][m]
                current_time += walking_time
                waiting_time += walking_time
            if current_time < pts['opening'][m]:  # wait until market is opening
                waiting_time += pts['opening'][m] - current_time
                current_time = pts['opening'][m]
            if current_time < pts['closing'][m]:  # stay for 30 mins or until market is closing
                current_time += min(market_time, max(pts['closing'][m] - current_time, 0))
            if current_time >= ending_time:  # go home
                break
        return waiting_time

    # return
    # ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=100, prob_mut=1)
    ga_tsp = GA_TSP(func=time_spend_waiting, n_dim=num_points, size_pop=50, max_iter=1000, prob_mut=1)

    ga_tsp.register('selection', selection.selection_tournament, tourn_size=3). \
        register('mutation', mutation.mutation_TSP_1). \
        register('crossover', crossover.crossover_2point_bit). \
        register('ranking', ranking.ranking_linear)

    best_points, best_distance = ga_tsp.run()

    print('best routine:', best_points, 'time spent walking/waiting:', best_distance / 3600)
    total_walking_time = pts['opening'][best_points[0]] - start_time
    current_time = pts['opening'][best_points[0]]
    print("1. {0}, time of arrival: {1}, walking time {2}, total walking time {3}".format(
        pts['name'][best_points[0]],
        secs_to_time(start_time + total_walking_time),
        secs_to_time(total_walking_time), secs_to_time(total_walking_time)))
    for i in range(1, len(best_points)):
        m = best_points[i]
        walking_time = duration_matrix[best_points[i - 1]][m]
        total_walking_time += walking_time
        current_time += walking_time + min(market_time, max(pts['closing'][m] - current_time, 0))
        print('{0}. {1}, time of arrival: {2}, walking time {3}, total walking time {4}'
              .format(i + 1,
                      pts['name'][m],
                      secs_to_time(current_time),
                      secs_to_time(walking_time),
                      secs_to_time(total_walking_time)))
        if current_time >= ending_time:
            break


def secs_to_time(secs):
    hours = np.floor(secs / 3600)
    return "{0:02.0f}:{1:02.0f}".format(hours, np.floor(secs / 60) - hours * 60)


def time_to_secs(timestring):
    time = datetime.strptime(timestring.strip(), '%H:%M')
    return time.second + time.minute * 60 + time.hour * 3600

# This is copied from the original TSP example
def function_for_TSP(num_points, seed=None):
    if seed:
        np.random.seed(seed=seed)

    # generate coordinate of points randomly
    points_coordinate = np.random.rand(num_points, 2)
    distance_matrix = spatial.distance.cdist(
        points_coordinate, points_coordinate, metric='euclidean')

    # print('distance_matrix is: \n', distance_matrix)

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    return num_points, points_coordinate, distance_matrix, cal_total_distance


if __name__ == '__main__':
    main()

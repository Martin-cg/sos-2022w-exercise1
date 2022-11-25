import json

import pandas as pd
import requests
import numpy as np
import re

osrm_server_walking = "https://api.openrouteservice.org/v2/matrix/foot-walking"
api_key = "5b3ce3597851110001cf6248ebba63aede984b07be14be05a0453ffe"  # i requested this, pls don't go crazy


def main(num,opening,closing):
    np.random.seed(0)
    opening_max_delay = 5
    closing_max_delay = 5
    thirty_min_chance = 0.25
    df = pd.DataFrame(data={'name': ["market {0}".format(n) for n in range(num)],
                            'link': None,
                            'opening': [
                                "{0:02.0f}:{1:02.0f}".format(np.floor(opening + opening_max_delay * np.random.rand()),
                                                             30 * (np.random.rand() > thirty_min_chance)) for _ in
                                range(num)],
                            'closing': [
                                "{0:02.0f}:{1:02.0f}".format(np.floor(closing - closing_max_delay * np.random.rand()),
                                                             30 * (np.random.rand() > thirty_min_chance)) for _ in
                                range(num)],
                            'coords': None,
                            'lat': 48.21 + np.random.randn(num) * 0.05,
                            'lng': 16.36 + np.random.randn(num) * 0.05})
    df.to_csv('../data/tcmt_instances/tcmt_{0}_{1}-{2}.csv'.format(num, opening, closing), index_label=False)
    locations = []
    for i in range(len(df)):
        locations.append([df.loc[i, "lng"], df.loc[i, "lat"]])

    # res = requests.post(osrm_server_walking,
    #                     data=str({'locations': locations}).replace("'", '"'),  # otherwise api doesn't accept the json
    #                     headers={'Authorization': api_key, 'Content-Type': 'application/json'})
    # durations = json.loads(res.text)['durations']
    durations = get_big_distance_matrices(locations)
    print(durations)
    input()
    dur_df = pd.DataFrame(durations, columns=df["name"], index=df["name"])
    print(dur_df)
    dur_df.to_csv('../data/tcmt_instances/tcmt_{0}_{1}-{2}_durations.csv'.format(num, opening, closing), index_label=False)


def get_big_distance_matrices(locations):
    nrow = int(np.floor(2500/len(locations)))
    durations = np.empty((0, len(locations)))
    for i in range(int(np.ceil(len(locations)/nrow))):
        res = requests.post(osrm_server_walking,
                            data=str({'locations': locations,
                                      'sources': list(range(i*nrow, min((i+1)*nrow, len(locations))))})
                            .replace("'", '"'),
                            headers={'Authorization': api_key, 'Content-Type': 'application/json'})
        print(res)
        print(res.text)
        durations = np.append(durations, json.loads(res.text)['durations'],axis = 0)
    return durations

if __name__ == '__main__':
    pass
    #dont
    # for i in [100, 200]:
    #     main(i, 9, 23)
    # for i in [5, 10, 20, 50, 100, 200]:
    #     main(i, 9, 20 + i/5)

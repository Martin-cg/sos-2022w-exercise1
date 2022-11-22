import pandas
import pandas as pd
import requests
import json

osrm_server_walking = "https://api.openrouteservice.org/v2/matrix/foot-walking"
api_key = "5b3ce3597851110001cf6248ebba63aede984b07be14be05a0453ffe"  # i requested this, pls don't go crazy


def main():
    df = pd.read_csv('../data/tcmt_coordinates.csv')
    locations = []
    for i in range(len(df)):
        locations.append([df.loc[i, "lng"], df.loc[i, "lat"]])

    res = requests.post(osrm_server_walking,
                        data=str({'locations': locations}).replace("'", '"'),  # otherwise api doesn't accept the json
                        headers={'Authorization': api_key, 'Content-Type': 'application/json'})
    durations = json.loads(res.text)['durations']
    dur_df = pandas.DataFrame(durations, columns=df["name"], index=df["name"])
    dur_df.to_csv('../data/tcmt_durations.csv')
    print(dur_df)


if __name__ == '__main__':
    main()

import pandas as pd
import requests
import re

coordinates = re.compile(r"\@?(48\.[0-9]+),\+?(16\.[0-9]+)")


def main():
    df = pd.read_csv('../data/tcmt_data.csv')
    df['coords'] = df.apply(lambda row: get_coords(row.link), axis=1)
    df['lat'] = df.apply(lambda row: row.coords[0], axis=1)
    df['lng'] = df.apply(lambda row: row.coords[1], axis=1)
    df.to_csv('../data/tcmt_coordinates.csv', index_label=False)
    print(df)


# when following the link, at some point the url contains the coordinates
# for g.page links, the page itself contains the coordinates instead
def get_coords(link):
    headers = {'Cookie': 'CONSENT=YES+cb.20211212-16-p1.de+FX+896',  # so the consent page doesn't come up
               'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0'}  # so google
    # isn't annoying
    while not coordinates.match(link):
        r = requests.get(link, allow_redirects=False, cookies={}, headers=headers)
        if 'location' in r.headers:
            link = r.headers['location']
        else:
            return coordinates.findall(r.text)[0]

    return coordinates.findall(r.headers['location'])[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

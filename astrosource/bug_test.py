import numpy as np
from pathlib import Path

BASE_PATH = Path('/Users/egomez/Downloads/weirdperiodissue/')
INPUT_PATHS = ['autovar_12weekcourse_2019b_ip','iperiodtest']

def read_sort_phot(path):
    doerp = np.genfromtxt(BASE_PATH / path / 'stdComps.csv', delimiter=',')
    dr = [d[0] for d in  doerp]
    dr.sort()
    return dr

def run():
    phot = []
    for path in INPUT_PATHS:
        phot.append(read_sort_phot(path))
    print(f'Length autovar: {len(phot[0])} astrosource: {len(phot[1])}')
    for x in zip(phot[0],phot[1]):
        print(x[0] / x[1])


if __name__ == '__main__':
    run()

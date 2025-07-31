from prometheus_client import start_http_server, Gauge
import numpy as np
import time

def open_server():
    # Opens port 9091 so Prometheus can scrape
    port = 9091
    start_http_server(port)
    print('Started HTTP server on port {}'.format(port))

def seek_end(f):
    # seeks to the end of a file
    f.seek(0,2)

def read_latest(f, count):
    # reads the next line out of a file (array size count)
    l = None
    while True:
        l = f.readline()
        if l:
            return np.fromstring(l, sep=' ', count=count)

open_server()

filepaths = [ './data/state_data_errors.txt', './data/state_data_inputs.txt', './data/state_data_setpoints.txt', './data/state_data_states.txt', './data/state_data_auxtemps.txt', './data/state_data_G.txt' ]
counts = [ 12, 6, 6, 12, 2, 12 ]
files = [ open(f, 'r') for f in filepaths ]

side_labels = ['back','front','left','right','top','bottom']

state_labels = [i + ' energy' for i in side_labels] + [i + ' temperature' for i in side_labels] + [ i + ' energy error' for i in side_labels ] + [ i + ' temperature error' for i in side_labels ]

state_gauge = Gauge('state', 'State information', [ 'index' ])

while True:
    ss = read_latest(files[3], counts[3])
    errs = read_latest(files[0], counts[0])
    inputs = read_latest(files[1], counts[1])
    auxtemps = read_latest(files[4], counts[4])
    G = read_latest(files[5], counts[5])
    for s in range(len(ss)):
        state_gauge.labels(state_labels[s]).set(ss[s])
    for s in range(len(errs)):
        state_gauge.labels(state_labels[s+len(ss)]).set(errs[s])
    for s in range(len(inputs)):
        state_gauge.labels(side_labels[s] + ' input').set(inputs[s])
    for s in range(len(auxtemps)):
        state_gauge.labels(['right grad', 'left grad'][s]).set(auxtemps[s])
    for s in range(len(G)):
        state_gauge.labels('G' + str(s)).set(G[s])
    seek_end(files[3])
    seek_end(files[0])
    seek_end(files[1])
    seek_end(files[4])
    seek_end(files[5])
    print(time.ctime(time.time()))


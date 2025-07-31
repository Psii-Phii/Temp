import numpy as np
import traceback
from time import time as gettime

datatypes = [ 'times', 'states', 'inputs', 'setpoints', 'errors', 'auxtemps', 'G' ]

def get_data():
    '''Loads all states, inputs, and setpoints in a file, timestamped. returns (times, states, inputs, setpoints, errors) tuple.'''
    data = { key: None for key in datatypes }
    filepaths = { d: './data/state_data_{}.txt'.format(d) for d in datatypes }

    # open in bytes (for np.save), append
    try:
        # get total size of all files together
        total_read_length = 0
        def get_length(filepath):
            with open(filepath, 'br') as file:
                file.seek(0, 2)
                l = file.tell()
                file.close()
            return l

        for i in datatypes:
            total_read_length += get_length(filepaths[i])

        print('Total data length: {:.3f}GB'.format(total_read_length/1.0e9))
        
        total_time = 0.0
        read_length = 0
        def load_data(datatype, read_length, total_time):
            st = gettime()
            print('Loading {}...'.format(datatype))
            with open(filepaths[datatype], 'br') as file:
                data[datatype] = np.loadtxt(file)
                file.seek(0, 2)
                read_length += file.tell()
                file.close()
            total_time += gettime()-st
            print('Took {:.1f}s. ETA remaining: {:.2f}s'.format(gettime()-st, total_time/(read_length/(total_read_length-read_length)) if total_read_length != read_length else 0.0))
            return read_length, total_time

        for i in datatypes:
            read_length, total_time = load_data(i, read_length, total_time)

        print('Loading complete!')
        print('Took {:.1f}s (avg. rate of {:.2f}MB/s)'.format(total_time, (total_read_length/1.0e6)/total_time))
        
        return data
    except:
        print('Failed to read data: ' + traceback.format_exc())
        return data

def plot(title, plot_index, xaxis, data, label, style='-', index=111, axis_average=False):
    fig=plt.figure(num=plot_index)
    ax = fig.add_subplot(index)
    ax.plot(xaxis, data, style, label=label, linewidth=0.5)
    if(axis_average):
        avg = np.average(data)
        sd = np.std(data)
        ax.axis([None,None,avg-sd*2.0, avg+sd*2.0])
    ax.set_title(title)
    ax.legend(loc='upper left')

# actual plotting time
# get data
data = get_data()

l = min([len(i) for i in data.values()])
print('Minimum length of arrays is {}. Lengths are {}, {}, {}, {}, {}'.format(l, len(data['times']), len(data['states']), len(data['inputs']), len(data['setpoints']), len(data['errors'])))
for i in data.keys():
    data[i] = data[i][-l:]

duration = data['times'][-1]-data['times'][0] # seconds

low_time = input('How far back to plot? (s, no input for all) Duration is {:n}s '.format(duration))
if low_time == '':
    low_time = 0
else:
    low_time = int(low_time)
    l = low_time
    for i in data.keys():
        data[i] = data[i][-l:]
print('Importing slow imports')

#slow imports
import matplotlib as mpl
import matplotlib.style as mplstyle
mpl.use('TKAgg', force=True)
mplstyle.use('fast')
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates

convert = input('Convert times to pretty MPL dates?')
if convert == 'y':
    print('Converting times')
    plt.rcParams['timezone'] = 'America/Toronto'
    data['times'] = (mpldates.num2date(data['times']/(24*3600)))

num_states = data['states'].shape[1]
num_inputs = num_states//2

np.set_printoptions(precision=3, linewidth=np.inf)
labels = ['Ba','F','L','R','T','B']
temp_avgs = np.average(data['states'][:,num_inputs:], axis=0)
temp_stds = np.std(data['states'][:,num_inputs:], axis=0)
print('Avg. Temps:\n{}\nStd Devs:\n{}'.format(temp_avgs, temp_stds))
avgs = np.average(data['errors'], axis=0)
stds = np.std(data['errors'], axis=0)
print('Avg. errors:\n{}\nStd Devs:\n{}\nSD Proportions (%):\n{}'.format(avgs, stds, np.abs(stds/avgs*100.0)))
print('Plotting temps')
plot('Temperatures', 1, data['times'], np.append(data['states'][:,num_inputs:], data['auxtemps'], axis=1), labels + ['GR', 'GL'], index=312, axis_average=True)
print('Plotting energies')
plot('Heater Energies', 2, data['times'], data['states'][:,:num_inputs], labels, axis_average=True)
print('Plotting inputs')
plot('Inputs', 1, data['times'], np.c_[data['inputs'], np.sum(data['inputs'], axis=1)[:,np.newaxis]], labels + ['Total'], index=313)
print('Plotting setpoints')
plot('Setpoints', 1, data['times'], data['setpoints'], labels, index=311)
print('Plotting errors')
plot('Model Errors', 3, data['times'], data['errors'][:,:num_inputs], ['E' + i for i in labels], '--', index=212)
plot('Model Errors', 3, data['times'], data['errors'][:,num_inputs:], ['T' + i for i in labels], '--', index=211)

plt.show()

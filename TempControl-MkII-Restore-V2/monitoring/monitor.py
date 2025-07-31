import time
import matplotlib as mpl
mpl.use('TKAgg', force=True)
import matplotlib.pyplot as plt
import matplotlib.dates as mpld
import numpy as np

file_errors = open('./data/state_data_errors.txt', 'r')
file_inputs = open('./data/state_data_inputs.txt', 'r')
file_setpoints = open('./data/state_data_setpoints.txt', 'r')
file_states = open('./data/state_data_states.txt', 'r')
file_times = open('./data/state_data_times.txt', 'r')

fig_obs = plt.figure('Observables')
fig_obs_axes = fig_obs.subplots(nrows=3,ncols=1, sharex=True)
fig_obs_axes[0].set_title('Setpoints (C)')
fig_obs_axes[1].set_title('Temperatures (C)')
fig_obs_axes[2].set_title('Inputs (W)')

len_recorded = 10
num_inputs = 6
errors = np.zeros((len_recorded, 2*num_inputs))
inputs = np.zeros((len_recorded, num_inputs))
setpoints = np.zeros((len_recorded, num_inputs))
states = np.zeros((len_recorded, 2*num_inputs))
times = np.array([[i] for i in range(len_recorded)])

def seek_end():
    file_errors.seek(0,2)
    file_inputs.seek(0,2)
    file_setpoints.seek(0,2)
    file_states.seek(0,2)
    file_times.seek(0,2)

def load_data_from_file(f, c):
    l = f.readline()
    if l:
        return np.fromstring(l, sep=' ', count=c)
    else:
        return None

def get_data():
    errors_ = None
    inputs_ = None
    setpoints_ = None
    states_ = None
    times_ = None
    
    while(1):
        needs_loop = False
        if errors_ is None:
            errors_ = load_data_from_file(file_errors, errors.shape[1])
            needs_loop = True
        if inputs_ is None:
            inputs_ = load_data_from_file(file_inputs, inputs.shape[1])
            needs_loop = True
        if setpoints_ is None:
            setpoints_ = load_data_from_file(file_setpoints, setpoints.shape[1])
            needs_loop = True
        if states_ is None:
            states_ = load_data_from_file(file_states, states.shape[1])
            needs_loop = True
        if times_ is None:
            times_ = load_data_from_file(file_times, times.shape[1])
            needs_loop = True
        if not(needs_loop):
            return (errors_, inputs_, setpoints_, states_, times_)

def add_to_plot(errs, ins, sps, sts, ts):

    errors[:len_recorded-1] = errors[1:]
    inputs[:len_recorded-1] = inputs[1:]
    setpoints[:len_recorded-1] = setpoints[1:]
    states[:len_recorded-1] = states[1:]
    times[:len_recorded-1] = times[1:]
    errors[-1] = errs
    inputs[-1] = ins
    setpoints[-1] = sps
    states[-1] = sts
    times[-1] = ts

    start_time = 0
    for i in times:
        if i > 0.01 or start_time == len_recorded:
            break
        start_time += 1
    
    for i in range(len(line0)):
        line0[i].set_ydata(setpoints[:,i])
    for i in range(len(line1)):
        line1[i].set_ydata(states[:,num_inputs+i])
    for i in range(len(line2)):
        line2[i].set_ydata(inputs[:,i])
    
    fig_obs_axes[0].set_ylim([np.min(setpoints)-0.5, np.max(setpoints)+0.5])
    fig_obs_axes[1].set_ylim([np.min(states[:,num_inputs:])-0.5, np.max(states[:,num_inputs:])+0.5])
    fig_obs_axes[2].set_ylim([np.min(inputs)-0.5, np.max(inputs)+0.5])
    plt.gcf().canvas.draw()
    plt.gcf().canvas.start_event_loop(0.1)



plt.ion()
plt.show()

line0 = fig_obs_axes[0].plot(times, setpoints)
line1 = fig_obs_axes[1].plot(times, states[:,num_inputs:])
line2 = fig_obs_axes[2].plot(times, inputs)

while(1):
    es, ins, sps, ss, ts = get_data()
    print(time.ctime(ts[0]))
    seek_end()
    add_to_plot(es, ins, sps, ss, ts)

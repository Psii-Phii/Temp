'''Take a large amount of data, and process it. The data will contain temperatures, as well as heater outputs.'''

import numpy as np
import scipy as sp
import parse
from parse import compile
np.set_printoptions(precision=4,linewidth=np.inf)

################### PREDEFINITIONS

heater_max_powers = [12**2/(4.7+1.33)**2 * 4.7] * 5 + [12**2/(40.0/9 + 1.33)**2 * (40/9)]
ambient_temperature = 21.5

data_path = "./identification/profiling_data/"
save_path = "./identified_system/"

################### PRES END

heater_power = float(input('Heater power? (% PWM) '))

heater_times_temps = {} # a dictionary with keys of heater index, and values of [np.array of temperatures, times]
heater_times_temps_cooling = {} # a dictionary with keys of heater index, and values of [np.array of temperatures, times] (for the cooling portion)
num_inputs = 0
num_states = 0

def closest_in_array(arr, x):
    '''Returns the index of the closest value to x in the array arr.'''
    closest_i = 0
    closest_dist = abs(arr[0] - x)
    for i in range(len(arr)):
        dist = abs(arr[i] - x)
        if dist < closest_dist:
            closest_i = i
            closest_dist = dist
    return closest_i

def load_data():
    global heater_times_temps
    global heater_times_temps_cooling
    global num_inputs
    global num_states
    
    compiled_parser = parse.compile("time: {}, temperatures: [{}], heater: {}")
    with open(data_path+'test_data.txt', 'r') as file:
        lines = file.read().split('\n')
    for i in lines:
        data = compiled_parser.parse(i)
        if data is None:
            continue
        time = float(data[0])
        heater = int(data[2])
        if heater_times_temps.get(heater) == None:
            heater_times_temps[heater] = []
        heater_times_temps[heater].append(([ float(t) for t in data[1].split(',') ], time))

    compiled_parser = parse.compile("C time: {}, temperatures: [{}], heater: {}")
    with open(data_path+'test_data.txt', 'r') as file:
        lines = file.read().split('\n')
    for i in lines:
        data = compiled_parser.parse(i)
        if data is None:
            continue
        time = float(data[0])
        heater = int(data[2])
        if heater_times_temps_cooling.get(heater) == None:
            heater_times_temps_cooling[heater] = []
        heater_times_temps_cooling[heater].append(([ float(t) for t in data[1].split(',') ], time))

    num_inputs = len(heater_times_temps) # ASSUME that there is one sensor for every heater
    num_states = num_inputs*2

def plot_data():
    global tcs
    global heater_times_temps
    import matplotlib.pyplot as plt
    labels=['Ba','F','L','R','T','Bo']
    colors=['r','g','b','y','m','c']
    for heater in heater_times_temps:
        plt.figure('Heater ' + str(heater))
        temps = [[] for i in range(len(heater_times_temps))]
        times = []
        for i in heater_times_temps[heater][:-1]:
            for j in range(len(temps)):
                temps[j].append(i[0][j])
            times.append(i[1])

        average_width = 60
        avg_temps = [ [ sum(temps[i][j-average_width:j+average_width])/(2*average_width+1) for j in range(average_width, len(temps[i])-average_width) ] for i in range(len(temps)) ]
        ders = [ [ t[i+1] - t[i] for i in range(len(t)-1) ] for t in avg_temps ]
        avg_ders = [sum(d[:5])/5 for d in ders]
        expected_values = [[avg_temps[i][0] for t in range(len(avg_temps[i]))] for i in range(len(avg_temps))]#[ [avg_temps[i][0] + (t*avg_ders[i]) for t in range(len(avg_temps[i]))] for i in range(len(avg_temps)) ]

        for i in range(len(avg_temps)):
            plt.plot(times[average_width:-average_width], avg_temps[i], colors[i], label=labels[i])
        
        plt.legend()
    plt.show()

def get_leakage(sensor_index):
    '''Gets the cooling rate for a sensor. Uses its associated heater for greatest "resolution"'''
    global heater_times_temps_cooling

    heater_index = sensor_index
    
    temperatures = [ ts[0][sensor_index] for ts in heater_times_temps_cooling[heater_index] ] # get only this sensor's temperature data that has to do with the heater indicated
    def gain_func(x, leakage, maxgain):
        leakage = np.abs(leakage)
        gain_prop = np.exp(-x*leakage)
        return maxgain*gain_prop
    def error_func(args, xvals, yvals, maxgain, true_deadtime=None):
        if(true_deadtime is not(None)):
            args[0] = true_deadtime
        errors = np.abs(gain_func(xvals, *args, maxgain) - yvals)*xvals
        return np.sum(errors * np.abs(xvals - xvals[len(xvals)//2]))

    cutoff = 1
    if(sensor_index == 3):
        cutoff = -3600
    newgains = np.array(temperatures) - temperatures[-cutoff]
    maxgain = max(newgains)
    xvals = np.arange(0.0, len(newgains), 1.0)
    popt = sp.optimize.minimize(error_func, [ 1.0/1000 ], args=(xvals, newgains, maxgain)).x
    leakage = -np.abs(popt[0])

    import matplotlib.pyplot as plt
    plt.plot(xvals, newgains)
    plt.plot(xvals, gain_func(xvals, *popt, maxgain))
    plt.show()

    # avg_leakage
    return leakage

def get_time_constant(heater_index, sensor_index, c_set=4.0):
    '''Gets the gain/(input*t.c) and time constant from one heater to a sensor'''
    global heater_times_temps
    
    # Note: the time constant is the amount of time from dead time to 63.2% gain.
    average_width = 30 # samples
    sample_length = 1 # second
    # Calculate derivatives (smooth)
    temperatures = [ ts[0][sensor_index] for ts in heater_times_temps[heater_index] ] # get only this sensor's temperature data that has to do with the heater indicated
    if(len(temperatures) < average_width):
        return (1.0, 0.0)
    temperatures = temperatures[:-10]
    average_temperatures = [ sum(temperatures[i-average_width:i+average_width])/(2*average_width+1) for i in range(average_width, len(temperatures)-average_width) ]
    derivatives = [ (average_temperatures[i+1] - average_temperatures[i])/sample_length for i in range(len(average_temperatures)-1) ]
    avg_derivatives = [ sum(derivatives[i-average_width:i+average_width])/(2*average_width+1) for i in range(average_width, len(derivatives)-average_width) ]

    # find maximum derivative, and how long it takes to get there (deadtime)
    max_derivative_index = avg_derivatives.index(max(avg_derivatives))
    max_derivative_temperature = average_temperatures[max_derivative_index+average_width]
    max_derivative_slope = max(avg_derivatives)
    min_gain = average_temperatures[0] # no need to search everywhere.
    
    deadtime = (max_derivative_index+average_width*2)*sample_length - (max_derivative_temperature - min_gain) / max_derivative_slope # seconds
    
    # just use SciPy to fit an exp curve.
    cutoff = 1
    newgains = np.array(temperatures[:-cutoff]) - temperatures[0]

    def gain_func(x, deadtime, tau, Geq):
        return np.piecewise(x, [x > np.abs(deadtime), x<=np.abs(deadtime)], [lambda x: Geq*(1.0 - np.exp(-(x-np.abs(deadtime))/np.abs(tau))), 0.0])
    
    def error_func(args, xvals, yvals, true_deadtime=None):
        if(true_deadtime is not(None)):
            args[0] = true_deadtime
        errors = np.abs(gain_func(xvals, *args) - yvals)*xvals
        return np.sum(errors)

    xvals = np.arange(0.0, len(newgains), 1.0)
    popt, pcov = sp.optimize.curve_fit(gain_func, xvals, newgains, p0=[len(newgains)/4, 1.0, 1.0])
    popt = sp.optimize.minimize(error_func, [ deadtime, 1000.0, 2.0 ], args=(xvals[:int(deadtime)*4], newgains[:int(deadtime)*4])).x
    deadtime = np.abs(popt[0])
    popt = sp.optimize.minimize(error_func, popt, args=(xvals, newgains, deadtime)).x
    time_constant = popt[1]
    eq_gain = popt[2]
    if(time_constant > 0.0):
        print(heater_index, sensor_index, time_constant, eq_gain, deadtime)
        import matplotlib.pyplot as plt
        print(popt)
        plt.plot(xvals, newgains)
        plt.plot(xvals, gain_func(xvals, *popt))
        plt.show()

    if time_constant <= 0.0:
        print('Zero or negative time constant at H'+str(heater_index)+', S'+ str(sensor_index)+'. Forcing positivity')
        time_constant = -time_constant

    wattage = (heater_power*heater_power*1.0e-4) * heater_max_powers[heater_index]
    return (time_constant, deadtime, eq_gain, wattage)

def get_system_coupling_strength(heater_index, A_matrix):
    # adds together the effects of a heater to the rest of the system in quadrature
    elements = np.array(A_matrix[num_inputs:,heater_index])
    self_coupling = elements[heater_index]
    coupling = np.sum(elements)/self_coupling
    return coupling

def load_tcs_gains():
    global tcs
    global gains
    global heat_capacities
    global omega
    global leakages
    tcs = np.zeros((num_inputs,num_inputs))
    inputs_w = np.zeros(num_inputs)
    heat_capacities = np.zeros(num_inputs)
    gains = np.zeros((num_inputs,num_inputs))
    heater_gains = np.zeros((num_inputs, num_inputs)) # G/(I*deadtime * tc)
    pocket_gains = np.zeros((num_inputs, num_inputs)) # G/tc

    omega = np.zeros((2,2,num_inputs,num_inputs))
    leakages = np.zeros(num_inputs)
    for i in range(num_inputs):
        leakages[i] = get_leakage(i)
        for j in range(num_inputs):
            tc, dt, eq_gain, input_w = get_time_constant(i,j)
            gains[i,j] = eq_gain
            tcs[i,j] = tc
            if(i==j):
                heat_capacities[i] = input_w/np.abs(leakages[i]) * (1.0 - np.exp(leakages[i] * dt))
                inputs_w[i] = input_w
                print('Eq. Energy:', input_w/leakages[i], heat_capacities[i], input_w, leakages[i])
    for i in range(num_inputs):
        for j in range(num_inputs):
            heater_gains[i,j] = 1.0 / tcs[i,j] #gains[i,j] / (inputs_w[i] * tcs[i,j] - heat_capacities[i] / leakages[i] * gains[i,j])# should be total time?
            pocket_gains[i,j] = 1.0 / tcs[i,j] #gains[i,j] / tcs[i,j]
    for i in range(num_inputs):
        out_of_norm = []
        for j in range(num_inputs):
            if(i != j):
                proportion = tcs[i,j]/tcs[i,i]
                if(proportion > 10.0):
                    out_of_norm += [j]
                    print('TC {} at ({},{}) out of norm'.format(tcs[i,j], i, j))
                    tcs[i,j] = 0.0
                    gains[i,j] = 0.0
        index = 0
        for j in out_of_norm:
            avg = (sum(tcs[i,:]) - tcs[i,i])/(num_inputs-1-len(out_of_norm)+index)
            avg_gains = (sum(gains[i,:]) - gains[i,i])/(num_inputs-1-len(out_of_norm)+index)
            index+=1
            print('Found that TC at ({},{}) is out of norm. Replacing TC/Gain with {}/{}'.format(i,j,avg,avg_gains))
            tcs[i,j] = avg
            gains[i,j] = avg_gains

    for i in range(num_inputs):
        heat_cap = heat_capacities[i] # J/K
        input_w = inputs_w[i]
        leakage_coef = leakages[i]
        print('Heat capacity: {}J/K'.format(heat_cap))
        
        # construct Hi
        omega[0,0,i,i] += leakage_coef
        print('Leakage: {}'.format(leakage_coef))
        
        for j in range(num_inputs):
            tc = tcs[i,j]
            opptc = tcs[j,i]
            avg_tc = (tc + opptc)/2.0
            other_heat_cap = heat_capacities[j]
            
            # Hj contribution
            omega[0,0,i,j] += (heat_cap/other_heat_cap) / avg_tc
            omega[0,0,i,i] -= 1.0/avg_tc

            # Tj contribution
            omega[0,1,i,j] += 2.0 * heat_cap / opptc
            omega[0,0,i,i] -= 2.0 / opptc
            omega[0,1,i,i] -= 2.0 * heat_cap / opptc

        # construct Ti
        omega[1,1,i,i] += leakage_coef/heat_cap

        for j in range(num_inputs):
            tc = tcs[i,j]
            opptc = tcs[j,i]
            avg_tc = (tc+opptc)/2.0
            other_heat_cap = heat_capacities[j]

            # Hj contribution
            omega[1,0,i,j] += 1.0/(heat_cap * tc)
            omega[1,1,i,i] -= 1.0/tc

            # Tj contribution
            omega[1,1,i,j] += 1.0/avg_tc
            omega[1,1,i,i] -= 1.0/avg_tc

# load data and time constants
load_data()
if input('Plot?')=='y':
    plot_data()
# get time constant matrix
load_tcs_gains()

print('Time constant matrix (each row corresponds to a heater)')
print(tcs)
print('Eq. Gain/Input ratio matrix (row corresponds to heater)')
print(gains)
print('Omega (0,0), (1,0), (0,1), (1,1)')
print(omega[0,0], omega[1,0], omega[0,1], omega[1,1])

A = np.zeros((num_states, num_states))
B = np.zeros((num_states, num_inputs))
C = np.zeros((num_inputs, num_states))
D = np.zeros((num_inputs, num_inputs))
G = np.zeros(num_states)

A[:num_inputs,:num_inputs] = omega[0,0]
A[num_inputs:,:num_inputs] = omega[1,0]
A[:num_inputs,num_inputs:] = omega[0,1]
A[num_inputs:,num_inputs:] = omega[1,1]
# leakage (box losing energy)
for i in range(num_inputs,num_inputs*2):
    G[i] -= np.sum(A[i,:]) * ambient_temperature

A += np.identity(A.shape[0]) # need to add the identity matrix because the equation is not dx/dt = Ax+Bu, but rather x[t+1] = Ax[t]+Bu


# construct B
#ezpz
for i in range(0, num_inputs):
    B[i,i] = 1.0#1/tcs[i][i]

# construct C
for i in range(num_inputs, num_inputs*2):
    C[i-num_inputs,i] = 1

couplings = np.zeros(num_inputs)
for i in range(0, 6):
    couplings[i] = get_system_coupling_strength(i, A)

# construct D (zeros)
#done

print('Matrices (A, B, C, D, G):')
print(A)
print(B)
print(C)
print(D)
print(G)

print('Heater-rest of system couplings (cost multipliers):')
print(couplings)

input(f'Input to save to files to {save_path}')

# write all the A,B,C,D to a file!
np.savetxt(save_path+'ssparams_A.out', A)
np.savetxt(save_path+'ssparams_B.out', B)
np.savetxt(save_path+'ssparams_C.out', C)
np.savetxt(save_path+'ssparams_D.out', D)
np.savetxt(save_path+'ssparams_G.out', G)

print('Make sure to update the first N entries of G as the controller deems fit')


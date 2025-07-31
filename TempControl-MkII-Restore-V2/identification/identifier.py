import numpy as np
np.set_printoptions(precision=3, linewidth=np.inf)
import scipy as sp
import matplotlib.pyplot as plt
import pathos.multiprocessing as mp

calls = 0
def callback(intermediate_result: sp.optimize.OptimizeResult):
    global calls
    if(calls % 1000 == 0):
        print('Callback: fun={:.2f}'.format(intermediate_result.fun))
    calls += 1

def direct_callback(xk):
    global calls
    if(calls % 1000 == 0):
        print('Callback: x =',xk)
    calls += 1

def basin_callback(x, fun, a):
    print(f'Callback, fun={fun}')

#def calculate_fft_temperatures(args, temperatures, inputs):
#    N = temperatures.shape[0]
#    k_pp = np.reshape(args[:N*N], (N,N))
#    k_pp -= k_pp * np.identity(N)
#    l = np.array(args[N*N:N*N+N])
#    c = np.array(args[N*N+N:N*N+2*N])
#    k_hp = np.array(args[N*N+2*N:N*N+3*N])
#    ckhp = c*k_hp # elementwise
#    freq = np.array([1,2,3,4,5,6])/3600.0
#
#    # timestep
#    timestep = np.exp(-1j * freq[None,:])-1 # for discrete time, e^(-i*omega) - 1, not omega*i. For small omega, this simplifies to omega*i (sinx=x, cosx=1)
#    
#    #k_omegas
#    k_omegas = np.sum(k_pp, axis=1)
#    
#    # numerator
#    weighted_temps = np.sum(k_pp[:,:,None] * temperatures[:,None,:], axis=1)
#    input_term = inputs / (timestep + ckhp[:,None])
#    ambient_term = 21.0 * l[:,None]
#    numerator = weighted_temps + input_term + ambient_term
#    # frequency is second axis, temperature index is first
#
#    # denom
#    fac = 1/(timestep/k_hp[:,None] + c[:,None])
#    denominator = timestep + k_omegas[:,None] + k_hp[:,None] + l[:,None] - ckhp[:,None] * fac
#
#    calculated_ts = numerator/denominator
#
#    return calculated_ts

def calculate_fft_temperatures(args, temperatures, inputs):
    N = temperatures.shape[0]
    h_pp = np.reshape(args[:N*N], (N,N))
    h_pp -= h_pp * np.identity(N)
    L = np.array(args[N*N:N*N+N])
    c = np.array(args[N*N+N:N*N+2*N])
    h_hp = np.array(args[N*N+2*N:N*N+3*N])
    #freq = np.array([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])/3600.0
    #freq = np.array([0,1,2,3,4,5,6])/3600.0
    #freq = np.array([1,2,3,4,5,6])/3600.0
    freq = np.fft.fftfreq(temperatures.shape[1])

    delta = np.exp(-1j * freq[None,:]) - 1 # for discrete time. Basically just i*omega.
    
    h_sum = np.sum(h_pp, axis=1) # TODO: force symmetric?
    
    # numerator
    weighted_temps = np.sum(h_pp[:,:,None] * temperatures[:,None,:], axis=1)
    input_term = (h_hp[:,None] * inputs/c[:,None]) * 1/(delta + h_hp[:,None])
    ambient_term = np.where(freq[None,:] == 0, 21.0 * L[:,None], 0.0) # delta function
    numerator = weighted_temps + input_term + ambient_term
    # frequency is second axis, temperature index is first

    # denom
    denominator = delta + h_sum[:,None] + h_hp[:,None] + L[:,None] - (h_hp[:,None])**2 * 1/(delta + h_hp[:,None])

    calculated_ts = numerator/denominator

    return calculated_ts

def error_func(args, temperatures, inputs):
    '''args=(k_pp[i,j] for all i,j (N*N, i==j=>0), such that they can be reshaped into ndarray[i,j], then l^i, c~^i, and k_hp^i). temperatures=fourier-space temperature profiles of all sensors, indexed [N,T] where T is the # of frequency samples'''
    calculated_ts = calculate_fft_temperatures(args, temperatures, inputs)
    differences = temperatures - calculated_ts
    
    num_elements = differences.shape[0]*differences.shape[1]

    diff_square = np.square(np.real(differences)) + np.square(np.imag(differences))
    abs_expectation = np.abs(temperatures)

    return np.sum(diff_square/abs_expectation)

def fit_data(args = None, plot=False):
    T=7200
    temperatures = np.loadtxt('./identification/profiling_data/testrun_temps.txt')[-T:]
    inputs = np.loadtxt('./identification/profiling_data/testrun_inputs.txt')[-T:]
    temperatures = np.swapaxes(temperatures, 1, 0)[:6]
    inputs = np.swapaxes(inputs, 1, 0)

    freq = np.fft.fftfreq(temperatures.shape[1])
    
    lowpass_range = 0.03
    lowpass_filter = np.exp(-(freq/lowpass_range)**2)

    temp_fft = np.fft.fft(temperatures, axis=1) * lowpass_filter # fft along each time stream (axis 1) # TODO: Does forward norm make sense? It's a normalization by 1/n...
    input_fft = np.fft.fft(inputs * np.indices((T,)).flatten(), axis=1)
    for i in range(6):
        input_fft[i,:] = 0
        input_fft[i,i] = 100
        input_fft[i,-i] = 100
    
    temp_fft = temp_fft
    input_fft = input_fft
    freq = freq

    N = temperatures.shape[0]

    #TODO: If fitting isn't good, play around with the bounds
    bounds = []
    for i in range(N*N):
        # k_pp
        bounds += [[1e-10, 1e-1]]
    for i in range(N):
        # l
        bounds += [[1e-10, 1e-1]]
    for i in range(N):
        # c
        bounds += [[1e2, 1e3]]
    for i in range(N):
        # k_hp
        bounds += [[1e-10, 1e-1]]

    if (args is None):
        x0 = np.ones(N*N+3*N) * 1e-5
    else:
        x0 = args
    
    print('Starting fit')
    ret = sp.optimize.minimize(error_func, args=(temp_fft, input_fft), bounds=bounds, method='Nelder-Mead', x0=x0, callback=callback)

    h_pp = np.reshape(ret.x[:N*N], (N,N))
    h_pp -= np.identity(N)*h_pp
    L = np.array(ret.x[N*N:N*N+N])
    c = np.array(ret.x[N*N+N:N*N+2*N])
    h_hp = np.array(ret.x[N*N+2*N:N*N+3*N])

    if(plot):
        print(f'Processed:\n\nh_pp:\n{h_pp}')
        print(f'L:\n{L}')
        print(f'c:\n{c}')
        print(f'h_hp:\n{h_hp}')
        for i in range(6):
            plt.plot(freq, np.real(temp_fft[i,:]), 'k', label='real data')
            plt.plot(freq, np.real(calculate_fft_temperatures(ret.x, temp_fft, input_fft)[i,:]), 'k--', label='real sim')
            plt.plot(freq, np.imag(temp_fft[i,:]), 'b', label='imag data')
            plt.plot(freq, np.imag(calculate_fft_temperatures(ret.x, temp_fft, input_fft)[i,:]), 'b--', label='imag sim')
            plt.legend()
            plt.show()

    return h_pp, L, c, h_hp, ret.x

def construct_matrices(h_pp, L, c, h_hp):
    N=len(c)
    omega = np.zeros(shape=(2,2,N,N))

    for i in range(N):
        for j in range(N):
            if(i == j):
                continue
            # first & second quadrants - pocket temperatures
            omega[0,0,i,j] += h_pp[i,j]
            omega[0,0,i,i] -= h_pp[i,j]
        omega[0,1,i,i] += h_hp[i]
        omega[0,0,i,i] -= h_hp[i] + L[i]
    omega[0,0] += np.identity(N)

    for i in range(N): # third and fourth quadrants
        omega[1,1,i,i] -= h_hp[i] + L[i]
        omega[1,0,i,i] += h_hp[i]
    omega[1,1] += np.identity(N)


    A = np.zeros(shape=(N*2,N*2))
    A[:N,:N] = omega[0,0]
    A[:N,N:] = omega[0,1]
    A[N:,:N] = omega[1,0]
    A[N:,N:] = omega[1,1]

    B = np.zeros(shape=(2*N, N))
    B += np.eye(12, 6, k=-6)
    for i in range(N):
        B[i+N,i] /= c[i]
    C = np.zeros(shape=(N, 2*N))
    C += np.eye(6, 12)
    D = np.zeros(shape=(N, N))

    G = np.zeros(12)
    G[:6] = L * 21 # approx room temp
    G[6:] = L * 21

    return A, B, C, D, G

load_from_file = True

if(load_from_file == False):
    h_pp, L, c, h_hp, args = fit_data(plot=False)

    m = 1
    for i in range(m): # this does actually reduce the cost! Significantly!
        h_pp, L, c, h_hp, args = fit_data(args=args, plot=(i==m-1))

    np.savetxt('./identification/profiling_data/h_pp.out', h_pp)
    np.savetxt('./identification/profiling_data/L.out', L)
    np.savetxt('./identification/profiling_data/c.out', c)
    np.savetxt('./identification/profiling_data/h_hp.out', h_hp)
    np.savetxt('./identification/profiling_data/args.out', args) # backup all args, unformatted
else:
    h_pp = np.loadtxt('./identification/profiling_data/h_pp.out')
    L =    np.loadtxt('./identification/profiling_data/L.out')
    c =    np.loadtxt('./identification/profiling_data/c.out')
    h_hp = np.loadtxt('./identification/profiling_data/h_hp.out')
    args = np.loadtxt('./identification/profiling_data/args.out')

A, B, C, D, G = construct_matrices(h_pp, L, c, h_hp)
print(A)

T = 3600*6*6
Tin = 3600*6*6
states = np.zeros(shape=(T+1, 12))
outputs = np.zeros(shape=(T,6))
inputs = np.zeros(shape=(T,6))
inputs[:Tin,:] = np.loadtxt('./identification/profiling_data/testrun_inputs.txt')[:Tin]*2.25#np.loadtxt('testrun_inputs.txt')[-Tin:]/2
#inputs[:Tin,:] = (np.cos(frequencies[None,:] * 2*np.pi * np.indices((Tin,)).flatten()[:,None])/2 + 1/2) * 100
#inputs[:Tin,:] = 0.03
x0 = np.ones(12)*21
states[0] = x0

actual_temps = np.loadtxt('./identification/profiling_data/testrun_temps.txt')[:T]

for i in range(T):
    states[i+1] = A@states[i] + B@inputs[i] + G
    #outputs[i] = C@states[i] + D@inputs[i]
    outputs[i] = states[i,:6]

plt.plot(outputs[:,:], '--')
plt.gca().set_prop_cycle(None)
plt.plot(actual_temps+1.65)
plt.show()

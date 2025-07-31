'''Uses a state-space representation to find optimal feedback gains.'''

import numpy as np # For matrices
import scipy as sp
import cvxpy as cp # for optimization
import time
np.set_printoptions(precision=4,linewidth=np.inf, suppress=True)

# DEFINE CONSTANTS HERE

enable_curses = True
if(enable_curses):
    import davis_curses as dc
    dc.set_screensize((2,1))
    from davis_curses import print

statespace_dir = "./identified_system/" # the directory where all the state space data is read from
data_path = "./data/" # the directory where all the data is written

max_prediction_length=25 # the length of time to predict and control for in the future
max_recorded_length = 3 # the length of time to record past measurements to improve the system matrix

# measurement-related
loop_interval = 1.0 # seconds
moving_average_weight = 0.6 # 0->1, this is the weight of the newest measurements in the temperature exponential moving average (low pass filter)
energy_update_weighting = 0*0.01 # 0->1 this is the weight of the energy approximation correction. This acts more as a low-pass filter than anything, so set nice and low to attenuate the majority of the high-frequency fluctuations. A lower value puts more faith in the model.
A_update_weight = 0.05 # state evolution matrix update coefficient

# system-related
maximum_heater_outputs = np.array([12**2 / (4.7 + 1.33)**2 * 4.7]*5 + [12**2/(40/9 + 1.33)**2 * (40/9)]) # wattages

gradient_weight = 0.0      # weight for non-zero gradient
setpoint_weight = 1.0        # weight for setpoint offset
stable_weight =   1000.0       # weight for temperature derivative

gradient_weights = [0.0, 1.0, 0.0]                      # relative weights of individual gradients
setpoint_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]       # relative weights of individual setpoint offsets
stable_weights =   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]       # relative weights of individual temperature derivatives
# good results were found with [0.1,0.1,1,1,0.1,0.1] for stable_weights & setpoint_weights

control_method = 'mpc' # Can be one of the following: 
# 'mpc': Model Predictive Control (on-the-fly finite horizon optimization with constraints).
# 'none': No method of control - simply monitors. Useful for simulation

# sim
simulation = False or control_method=='none' # if true, won't take real data or set real heaters. Will just simulate responses.

# END OF CONSTANTS

# Programmatic definitions

tick=True # for tick/tock

if(simulation):
    simtime = 0.0
else:
    warmup_time = 0.0 # to decide if we should start averaging etc.

# End of programmatic defns

if not(simulation):
    import packages.controller.rpi_uart as rpi_uart

def restart_pyboard():
    print('Resetting PyBoard', window_coords=(0,0))
    if simulation:
        return
    rpi_uart.write_cmd('reset')
    time.sleep(1.5)
    ret = rpi_uart.read_serial_buffer()[1]
    for i in ret:
        print(ret + '\n', window_coords=(0,0))

def load_system():
    A = np.loadtxt(statespace_dir + 'ssparams_A.out')
    B = np.loadtxt(statespace_dir + 'ssparams_B.out')
    C = np.loadtxt(statespace_dir + 'ssparams_C.out')
    D = np.loadtxt(statespace_dir + 'ssparams_D.out')
    G = np.loadtxt(statespace_dir + 'ssparams_G.out')

    print('Loading system from ssparams_*.', window_coords=(0, 0))
    print('A (System):\n{}'.format(np.array2string(A)), window_coords=(1, 0))
    print('B (Input):\n{}'.format(np.array2string(B)), window_coords=(1, 0))
    print('G (Constant):\n{}'.format(np.array2string(G)), window_coords=(1,0))
    print('C (Output):\n{}'.format(np.array2string(C)), window_coords=(1, 0))
    print('D (Passthrough):\n{}'.format(np.array2string(D)), window_coords=(1,0))

    num_states,num_inputs = B.shape

    # Cost function (state)
    Q = np.array([[ setpoint_weight*setpoint_weights[i-num_inputs] if (i==j and i >= num_inputs) else 0 for j in range(0, num_states) ] for i in range(0, num_states) ])
    for i in range(num_inputs//2):
        n = i*2 + num_inputs
        c = gradient_weight * gradient_weights[i] # cross-terms (gradient related)
        Q[n,n+1] = -c
        Q[n+1,n] = -c
        Q[n,n] += c
        Q[n+1,n+1] += c
    
    # Cost function (d/dt(Temperatures))
    S = np.identity(num_inputs)*stable_weight
    for i in range(len(stable_weights)):
        S[i] *= stable_weights[i]
    
    print('Q (Cost - system):\n{}'.format(np.array2string(Q)), window_coords=(0, 0))
    print('S (Cost - temp der):\n{}'.format(np.array2string(S)), window_coords=(0,0))

    return system(A, B, G, C, D, Q, S)

class system:
    '''Useful to pass all these matrices together'''
    def __init__(self, A, B, G, C, D, Q, S):
        self.A = A
        self.B = B
        self.G = G
        self.C = C
        self.D = D
        self.Q = Q
        self.S = S
        self.num_states,self.num_inputs = B.shape
        self.aux_sensors = 2 #number of purely reading sensors
        self.recorded_states = np.zeros((max_recorded_length, self.num_states))
        self.recorded_inputs = np.zeros((max_recorded_length-1, self.num_inputs))
        self.recorded_time = 0

        # the backpropagation min problem
        self.measured_states = np.zeros((max_recorded_length, self.num_states))
        self.measured_inputs = np.zeros((max_recorded_length-1, self.num_inputs))
        self.initial_state = np.zeros((self.num_states,))

    def print(self):
        #print('State matrix:\n{}\n\n'.format(np.array2string(self.A, precision=4)), window_coords=(0,0))
        #print('State cost matrix:\n{}\n\n'.format(np.array2string(self.Q[self.num_inputs:, self.num_inputs:], precision=2)), window_coords=(0, 0))
        #print('Temperature derivative cost matrix:\n{}\n\n'.format(np.array2string(self.S, precision=2)), window_coords=(0, 0))
        print('Gradient weights:     {:<10}*{:>35}'.format(gradient_weight, str(gradient_weights)), window_coords=(0,0))
        print('Setpoint weights:     {:<10}*{:>35}'.format(setpoint_weight, str(setpoint_weights)), window_coords=(0,0))
        print('Temp d/dt weights:    {:<10}*{:>35}\n'.format(stable_weight, str(stable_weights)), window_coords=(0,0))
        print('Control method: ' + control_method + '\n', window_coords=(0,0))

class mpc_problem_recursive:
    def init_problem(self, sys, setpoint):
        self.setpoint = setpoint
        
        # last position (minimization_length+1 states)
        Q_const = cp.Constant(sys.Q[sys.num_inputs:, sys.num_inputs:])
        S_const = cp.Constant(sys.S)

        cost = cp.quad_form(self.x[sys.num_inputs:,-1] - setpoint, Q_const)
        for i in range(self.minimization_length):
            cost += cp.quad_form(self.x[sys.num_inputs:,i] - setpoint, Q_const) # only really need the temps.
            cost += cp.quad_form(self.x[sys.num_inputs:,i+1] - self.x[sys.num_inputs:,i], S_const) # cost for *changing* temps

        self.problem = cp.Problem(cp.Minimize(cost), self.constraints)

    def init_constraints(self, sys):
        # initial condition
        self.constraints = [ self.x[:,0] == self.state_0 ]
        self.constraints += [ self.u[:,0] == self.input_0 ]

        A_const = cp.Constant(sys.A)
        B_const = cp.Constant(sys.B)
        for i in range(self.minimization_length):
            self.constraints += [ self.u[:,i] >= 0.0, self.u[:,i] <= maximum_heater_outputs ]
            self.constraints += [ self.x[:,i+1] == A_const@self.x[:,i] + B_const@self.u[:,i+1] + self.sys_G ]

    def __init__(self, sys, setpoint, minimization_length=25, previous_state=None):
        self.minimization_length = minimization_length #technically +1
        self.u = cp.Variable((sys.num_inputs, self.minimization_length+1))

        if(previous_state is None):
            self.x = cp.Variable((sys.num_states, self.minimization_length+1))
        else:
            last_x = previous_state
            self.x = cp.Variable((sys.num_states, self.minimization_length+1))
            self.x.value = np.zeros((sys.num_states, self.minimization_length+1))
            self.x.value[:last_x.shape[0],:last_x.shape[1]] = last_x

        self.state_0 = cp.Parameter(sys.num_states)
        self.input_0 = cp.Parameter(sys.num_inputs)
        self.sys_G = cp.Parameter(sys.num_states)
        self.setpoint = setpoint

        self.init_constraints(sys)
        self.init_problem(sys, setpoint)

    def solve(self, state_0, input_0, sys):
        self.state_0.value = state_0
        self.input_0.value = input_0
        self.sys_G.value = sys.G
        try:
            score = self.problem.solve(warm_start=True, solver=cp.OSQP, eps_abs=0.005, canon_backend=cp.SCIPY_CANON_BACKEND)
        except:
            print('Failed to solve. Returning no gains', window_coords=(1, 0))
            self.u.value = np.zeros(sys.num_inputs)
            return ([0 for i in range(sys.num_inputs)], -1)
        if(self.problem.status != 'optimal'):
            print('Solution is ' + self.problem.status, window_coords=(1, 0))
        out = self.u.value[:,1]
        print('Next 10 steps: \n', window_coords=(1, 0))
        print(np.array2string(self.u.value[:,:10]) + '\n', window_coords=(1, 0))
        print('Target in {} steps:\n\t{}\n\t{}\n'.format(self.minimization_length, np.array2string(self.x.value[:sys.num_inputs, -1]), np.array2string(self.x.value[sys.num_inputs:,-1])), window_coords=(1, 0))
        return (out, self.problem.solver_stats.num_iters)

class mpc_problem_explicit:
    def get_later_state(self, x0, inputs): # explicitly calculates later state after inputs, starting from state x0.
        state = self.A_lut[inputs.shape[1]]@x0
        for i in range(inputs.shape[1]):
            state += cp.sum(cp.multiply((inputs[:,i])[None,:], self.AtauJ_lut[inputs.shape[1]-1-i]), axis=1)
        state += self.AtauG_cum_lut[inputs.shape[0]-1]
        return state

    def init_problem(self, sys, setpoint):
        self.setpoint = setpoint
        
        # last position (minimization_length+1 states)
        Q_const = cp.Constant(sys.Q[sys.num_inputs:, sys.num_inputs:])
        S_const = cp.Constant(sys.S)

        cost = 0
        for i in range(self.minimization_length-1):
            #cost += cp.norm((self.x[sys.num_inputs:,i+1] - setpoint).T @ Q_const @ (self.x[sys.num_inputs:,i+1] - setpoint)) # only really need the temps.
            #cost += cp.norm(((self.x[sys.num_inputs:,i+1] - self.x[sys.num_inputs:,i])/(self.minimization_number+1)).T @ S_const @ ((self.x[sys.num_inputs:,i+1] - self.x[sys.num_inputs:,i])/(self.minimization_number+1))) # cost for *changing* temps
            cost += cp.quad_form((self.x[sys.num_inputs:,i+1] - setpoint), Q_const)
            cost += cp.quad_form(((self.x[sys.num_inputs:,i+1] - self.x[sys.num_inputs:,i])/(self.minimization_number+1)), S_const)
        self.problem = cp.Problem(cp.Minimize(cost), self.constraints)

    def prepare_LUTs(self, sys):
        # lookup tables for efficient cost functions
        self.A_lut = np.ndarray(shape=(self.simulation_time, *sys.A.shape), dtype=np.longdouble)
        J_lut = np.ndarray(shape=sys.B.shape, dtype=np.longdouble)

        self.A_lut[0] = np.identity(sys.A.shape[0], dtype=np.longdouble)
        for i in range(1, self.simulation_time):
            self.A_lut[i] = self.A_lut[i-1]@sys.A
        
        for i in range(sys.B.shape[1]):
            inp = np.zeros(sys.B.shape[1])
            inp[i] = 1
            J_lut[:,i] = sys.B@inp

        self.AtauJ_lut = self.A_lut @ J_lut
        AtauG_lut = np.append(np.zeros(sys.G.shape)[None,:], self.A_lut @ sys.G, axis=0)
        self.AtauG_cum_lut = np.cumsum(AtauG_lut, axis=0)

    def init_constraints(self, sys):
        # initial condition
        self.constraints = [ self.u[:,0] == self.input_0 ]

        A_const = cp.Constant(sys.A)
        B_const = cp.Constant(sys.B)
        for i in range(1, self.simulation_time-1):
            self.constraints += [ self.u[:,i] >= 0.0, self.u[:,i] <= maximum_heater_outputs ]

        self.constraints += [ self.x[:,0] == self.state_0 ]
        for i in range(1, self.minimization_length): # for every point
            self.constraints += [ self.x[:,i] == self.get_later_state(self.x[:,i-1], self.u[:,(i-1)*self.minimization_number:i*self.minimization_number]) ]

    def __init__(self, sys, setpoint, minimization_length=10, previous_state=None):
        self.minimization_length = minimization_length # number of calculated points
        self.minimization_number = 10 # seconds between calculated points
        self.simulation_time = self.minimization_length * self.minimization_number

        self.x = cp.Variable((sys.num_states, self.minimization_length))
        self.x.value = np.zeros(shape=(sys.num_states, self.minimization_length))
        if(not(previous_state is None)):
            last_x = previous_state
            self.x.value[:last_x.shape[0],:last_x.shape[1]] = last_x
        self.u = cp.Variable((sys.num_inputs, self.simulation_time-1))

        self.state_0 = cp.Parameter(sys.num_states)
        self.input_0 = cp.Parameter(sys.num_inputs)
        self.setpoint = setpoint

        self.prepare_LUTs(sys)
        self.init_constraints(sys)
        self.init_problem(sys, setpoint)

    def solve(self, state_0, input_0, sys):
        self.state_0.value = state_0
        self.input_0.value = input_0
        failed = False
        try:
            score = self.problem.solve(warm_start=True, solver=cp.OSQP, eps_abs=0.005, canon_backend=cp.SCIPY_CANON_BACKEND)
        except Exception as e:
            print(str(e))
            failed = True
        if(not(failed) and self.problem.status != 'optimal'):
            print('Solution is ' + self.problem.status, window_coords=(1, 0))
            failed = True
        if(failed):
            print('Failed to solve. Returning no gains', window_coords=(1, 0))
            return ([0 for i in range(sys.num_inputs)], -1)
        out = self.u.value[:,1]
        print('Next 10 steps: \n', window_coords=(1, 0))
        print(np.array2string(self.u.value[:,:10]) + ' \n', window_coords=(1, 0))
        print('Target in {} steps:\n\t{}\n\t{}\n'.format(self.minimization_length*self.minimization_number, np.array2string(self.x.value[:sys.num_inputs, -1]), np.array2string(self.x.value[sys.num_inputs:,-1])), window_coords=(1, 0))
        return (out, self.problem.solver_stats.num_iters)

def connect_tmps(sys):
    print('Connecting TMPs', window_coords=(0, 0))

    if simulation:
        return

    for i in range(sys.num_inputs + sys.aux_sensors):
        rpi_uart.write_cmd('connect_tmp:' + str(i+72))
    time.sleep(0.5)
    ret = rpi_uart.read_serial_buffer()[1]
    for i in ret:
        print(i, window_coords=(0,0))

def read_temperatures(sys):
    print('Reading temperatures: ', window_coords=(0,0))
    
    if simulation:
        print('\n\n', window_coords=(0,0))
        return None
    
    for i in range(sys.num_inputs + sys.aux_sensors):
        rpi_uart.write_cmd('measure_temp:'+str(i+72))
    time.sleep(0.1)
    data = []
    st = time.time()
    while len(data) < (sys.num_inputs + sys.aux_sensors):
        ret = rpi_uart.read_serial_buffer()[1]
        for i in range(len(ret)):
            if(ret[i] == 'None'):
                ret[i] == '0'
            temp = float(ret[i])
            print(str(len(data)) + ': ' + ret[i] + 'C, ', window_coords=(0,0))
            data.append(temp)
        if time.time() - st > 5.0:
            print('Failed to read all sensors. Returning borked...', window_coords=(0,0))
            return np.array(np.append(data, np.zeros(sys.num_inputs+sys.aux_sensors-len(data))))
    print('\n', window_coords=(0,0))
    return np.array(data)

def set_heaters(input_v):
    print('Setting heaters (%PWM): ' + str(input_v) + '\n\n', window_coords=(0,0))
    if simulation:
        return
    
    for i in range(len(input_v)):
       rpi_uart.write_cmd('heater:'+str(i)+','+str(input_v[i]))

def evolve_state(sys, state_v, input_v):
    '''Performs a unit step (1 loop interval). Returns (newstate, output)'''
    newstate = sys.A@state_v + sys.B@input_v + sys.G
    output = sys.C@state_v + sys.D@input_v
    return (newstate, output)

def calculate_gains_constrained(system_mpc, state_v, input_v, sys):
    atime = time.time()
    ret, solve_iters = system_mpc.solve(state_v, input_v, sys)
    btime = time.time() - atime

    print('Calculating input ({} steps) took {}s ({} cycles)\n\n'.format(system_mpc.minimization_length, btime, solve_iters), window_coords=(1,0))

    return ret, btime

def control_step(sys, sys_problem, state_v, input_v, setpoint_v, feedback_gains):
    '''Uses past state and input to evolve to this state. Records error, finds appropriate input to reach states(temp)=setpoint_v, and fixes state vector based on sensor data. Returns (newstate, new_input, error_in_model)'''
    newstate, output = evolve_state(sys, state_v, input_v)
    
    temperatures_full = read_temperatures(sys)
    
    if(not(simulation)):
        temperatures = temperatures_full[:sys.num_inputs]
        temperatures_aux = temperatures_full[sys.num_inputs:]
    else:
        temperatures = newstate[sys.num_inputs:]
        temperatures_aux = np.array([0.0, 0.0])

    last_temperatures = state_v[sys.num_inputs:]
    last_energies = state_v[:sys.num_inputs]
    # apply a low-pass filter (exponential moving average)
    weighted_temperatures = last_temperatures*(1-moving_average_weight) + temperatures*moving_average_weight

    # quantify the errors between expected and actual temperatures
    expected_temperatures = newstate[sys.num_inputs:]
    expected_energies = newstate[:sys.num_inputs]

    model_error = [0.0] * (sys.num_inputs*2)
    model_error[sys.num_inputs:] = weighted_temperatures - expected_temperatures
    if simulation:
        # ignore real temperatures
        weighted_temperatures = expected_temperatures

    # use error to correct heater energies
    omega = np.array([[np.zeros((sys.num_inputs,sys.num_inputs))]*2]*2)
    ni = sys.num_inputs
    omega[0,0] = sys.A[:ni,:ni]
    omega[1,0] = sys.A[ni:,:ni]
    omega[0,1] = sys.A[:ni,ni:]
    omega[1,1] = sys.A[ni:,ni:]
    new_energy = omega[0,0]@np.linalg.pinv(omega[1,0])@(weighted_temperatures - omega[1,1]@last_temperatures - sys.G[ni:]) + omega[0,1]@last_temperatures + input_v + sys.G[:ni]
    weighted_energies = expected_energies * (1.0-energy_update_weighting) + new_energy * energy_update_weighting
    #weighted_energies = np.maximum(weighted_energies, 0.0)
    model_error[:sys.num_inputs] = new_energy - expected_energies
    
    # update state energies
    newstate[:sys.num_inputs] = weighted_energies

    # update state temperatures
    newstate[sys.num_inputs:] = weighted_temperatures

    #sys.correct_state_evolution(state_v, input_v)

    # find appropriate input
    state_error = np.zeros(sys.num_states)
    state_error[:sys.num_inputs] = [0.0]*sys.num_inputs # don't care about the input
    state_error[sys.num_inputs:] = weighted_temperatures - setpoint_v #offset from setpoints
    # the explanation for not taking the heater energies into account as part of the state error is that they simply don't matter. We couldn't care less about the energy stored in the heaters, as we don't have any cost associated with them, only changing them (that's the input). They're useful for diagnosing bad profiling, though.
    
    calc_time = 10
    new_input = np.zeros(sys.num_inputs)
    if control_method == 'mpc':
        new_input, calc_time = calculate_gains_constrained(sys_problem, newstate, input_v, sys)

    return (newstate, new_input, model_error, temperatures_aux, calc_time)

def record_info(state_v, input_v, setpoint_v, model_err, auxtemps_v, G_v):
    '''Saves all timestamps, states, inputs, and setpoints in separate files. The line numbers should match up.'''
    # open in bytes (for np.save), append
    with open(data_path+'state_data_times.txt', 'ba') as file:
        if(simulation):
            global simtime
            np.savetxt(file, [simtime])
            simtime += 1
        else:
            np.savetxt(file, [time.time()])
        file.close()
    with open(data_path+'state_data_states.txt', 'ba') as file:
        np.savetxt(file, [state_v])
        file.close()
    with open(data_path+'state_data_inputs.txt', 'ba') as file:
        np.savetxt(file, [input_v])
        file.close()
    with open(data_path+'state_data_setpoints.txt', 'ba') as file:
        np.savetxt(file, [setpoint_v])
        file.close()
    with open(data_path+'state_data_errors.txt', 'ba') as file:
        np.savetxt(file, [model_err])
        file.close()
    with open(data_path+'state_data_auxtemps.txt', 'ba') as file:
        np.savetxt(file, [auxtemps_v])
        file.close()
    with open(data_path+'state_data_G.txt', 'ba') as file:
        np.savetxt(file, [G_v])
        file.close()

def record_parameters():
    with open(data_path+'ControlProfile.txt', 'a') as file:
        file.write('### New control run ###\n')
        file.write('Gradient weights: {}*{}'.format(gradient_weight, str(gradient_weights)))
        file.write('Setpoint weights: {}*{}'.format(setpoint_weight, str(setpoint_weights)))
        file.write('Control method: ' + control_method + '\n')
        file.write('Start time: ' + time.ctime(time.time()) + '\n')
        file.write('### End of parameters ###\n\n')
        file.close()

def correct_system(sys, model_error):
    '''Adjustst the constant term in the system to correct constant offsets. Functions much like the I term in PID control.'''
    integration_factor = 1.0e-4
    sys.G[sys.num_inputs:] += np.ones(sys.num_inputs)*(integration_factor)*model_error[sys.num_inputs:] # only temps

def control_loop():
    print('Starting', window_coords=(0,0))

    restart_pyboard()
    system = load_system()
    connect_tmps(system)
    # initial states
    input_v = np.ones(system.num_inputs)*0.0
    input_gain_v = np.zeros(system.num_inputs)*0.0
    state_v = np.ones(system.num_states)*0.0
    setpoint_v = np.ones(system.num_inputs)*32.0
    feedback_gains = np.zeros((system.num_inputs, system.num_states))

    if(not(simulation)):
        state_v[system.num_inputs:] = read_temperatures(system)[:system.num_inputs] # get initial temperatures
    if(not(simulation)):
        try:
            energies = np.loadtxt('last_energies.txt')
            state_v[:system.num_inputs] = energies
            print('Initial energies loaded.', window_coords(0,0))
        except:
            pass # no previous energies were found. Just start from zero.
    
    # generate the mathematical optimization problem
    system_problem = mpc_problem_explicit(system, setpoint_v)

    record_parameters()
    
    while True:
        dc.wrapper.clear()
        st = time.time()

        # apply statespace and LQR
        state_v,unc_input_v,model_err,auxtemps,calc_time = control_step(system, system_problem, state_v, input_v, setpoint_v, feedback_gains)
        #if(calc_time < 0.25 and system_problem.minimization_length < max_prediction_length):
        #    print('Calculating gains took sufficiently low time. Increasing the length of prediction.', window_coords=(1, 0))
        #    system_problem = mpc_problem_explicit(system, system_problem.setpoint, min(system_problem.minimization_length+25, max_prediction_length), previous_state=system_problem.x.value)
        correct_system(system, model_err)
     
        # apply constraints
        input_v = np.clip(unc_input_v, 0, maximum_heater_outputs)

        # give useful information
        print('State vector (separated for clarity):\n\tEnergy: {}'.format(np.array2string(state_v[:system.num_inputs])) + '\n\tTemps: {}\n\n'.format(np.array2string(state_v[system.num_inputs:])), window_coords=(1,0))
        print('Proposed inputs in W: ' + str(unc_input_v) + '\n', window_coords=(1,0))
        print('Allowed inputs in W: ' + str(input_v) + '\n\n', window_coords=(1,0))
        print('Model error (measured-expected, separated for clarity):\n\tEnergy: {}'.format(np.array2string(np.array(model_err[:system.num_inputs])) + '\n\tTemps: {}\n\n'.format(np.array2string(np.array(model_err[system.num_inputs:])))), window_coords=(1,0))
        print('System G:\n\tEnergy: {}\n\tTemps: {}\n'.format(str(system.G[:system.num_inputs]), str(system.G[system.num_inputs:])), window_coords=(0,0))
        system.print()

        # write current energies & G for ez restart
        np.savetxt(data_path+'last_energies.txt', state_v[:system.num_inputs])
        np.savetxt(data_path+'ssparams_G.out', system.G)

        # use input we calculated
        pwm_input_v = np.sqrt(input_v/maximum_heater_outputs) * 100.0
        set_heaters(pwm_input_v)

        # record for analysis
        record_info(state_v, input_v, setpoint_v, model_err, auxtemps, system.G)

        print('Grad difference (L-R): {:.3f}\n\n'.format(auxtemps[1]-auxtemps[0]), window_coords=(1, 0))
        
        et = time.time()
        dt = et-st
        print('Loop took {}s\n\n'.format(dt), window_coords=(1,0))
        
        global tick
        tickstr = 'tick!' if tick else 'tock!'
        print(tickstr+'\n\n', window_coords=(1,0))
        tick = not(tick)
        
        if(loop_interval - dt > 0):
            if not(simulation):
                time.sleep(loop_interval - dt)


def control_loop_safe():
    try:
        control_loop()
    except Exception as e:
        print(str(e))
        restart_pyboard()
        print('Failed. Exiting.')

control_loop_safe()

from math import exp

# define some constants
ROOM_TEMP = 21.0#22.75
DELTA_TEMP = 38.0-ROOM_TEMP#14.86
COOLING_CONST = -0.000461
CORRECTIVE_CONST_0 = -0.00371
CORRECTIVE_CONST_1 = -0.00083

heating = 0.0 # rate of change of temperature
dH = 0.0 # rate of change of heating

def get_delta_temp_cooling(timelength, temperature):
    dT = 0.0 #dTemperature
    timestep = 0.01
    start_time = find_t(temperature)
    for i in range(0, int(timelength/timestep)):
        # integrate over ddT to find change in temperature
        ddT = cooling_derivative(start_time+i*timestep)
        dT += ddT*timestep
    return dT

def get_delta_temp_heating(timelength, heaters):
    # assuming ideal heaters
    dT = 0.0
    ddT = 12.0**2/4.7*heaters / 375.0 #(P=V^2/R)
    dT = ddT * timelength
    return dT
    

def cooling_equation(t):
    '''Based off of time, use find_t to lazy invert.'''
    return ROOM_TEMP + DELTA_TEMP*exp(COOLING_CONST * t) + CORRECTIVE_CONST_0*t*exp(CORRECTIVE_CONST_1 * t)

def cooling_derivative(t):
    return DELTA_TEMP*COOLING_CONST*exp(COOLING_CONST*t) + CORRECTIVE_CONST_0*CORRECTIVE_CONST_1*t*exp(CORRECTIVE_CONST_1*t) + CORRECTIVE_CONST_0*exp(CORRECTIVE_CONST_1*t)

def find_t(temperature):
    '''Finds "time" based off of temperature. Newton's method.'''
    epsilon = 1.0e-4
    steps = 0
    err = epsilon*2
    x = 0.0
    while abs(err) > abs(epsilon):
        steps += 1
        der = cooling_derivative(x)
        if der == 0.0:
            x += 0.01
            continue
        x = x - (cooling_equation(x)-temperature)/der
        err = abs(cooling_equation(x) - temperature)
        if steps > 1000:
            print('find_t steps over 1000 trying to find t on temperature ' + str(temperature) + '. leaving')
            return x
    return x
        
def cooling_rate(temperature):
    '''Finds the rate of cooling for a specific temperature.'''
    return cooling_derivative(find_t(temperature))

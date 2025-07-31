import sys
import time
import numpy as np
import packages.controller.rpi_uart as rpi_uart

import dependencies.misc_tools.davis_curses as dc
from dependencies.misc_tools.davis_curses import print
dc.daviscurses_args['autoflush'] = False

fn = './identification/profiles/'+input("Please enter the profile name (folder name)")+'/'
exec(open(fn+'config.py', 'r').read())
params = read_pickle()

def connect_tmps():
    print('connecting sensors', flush=True)
    for i in params['sensor_addresses']:
        rpi_uart.write_cmd(f'connect_tmp:{i[0]},{i[1]}')
    time.sleep(0.5)
    ret = rpi_uart.read_serial_buffer()[1]
    for i in ret:
        print(i, flush=True)
    dc.wrapper.refresh()

def read_temperatures():
    print('reading temps')
    for i in params['sensor_addresses']:
        rpi_uart.write_cmd(f'measure_temp:{i[0]},{i[1]}')
    data = []
    st = time.time()
    while(len(data) < 8):
        ret = rpi_uart.read_serial_buffer()[1]
        for i in range(len(ret)):
            if(ret[i] == 'None'):
                ret[i] == '0'
            temp = float(ret[i])
            print(f'{len(data)}: {temp}C, ', end='')
            data.append(temp)
        if(time.time() - st > 5.0):
            print('failed')
            return np.array(np.append(data, np.zeros(8-len(data))))
    return np.array(data)

def set_heaters(s):
    print('setting heaters: ' + str(s))
    for i in range(len(s)):
        rpi_uart.write_cmd('heater:{},{}'.format(i, s[i]))

def write_temps_inputs(ts, ins):
    with open('./identification/profiling_data/testrun_inputs.txt', 'ba') as file:
        np.savetxt(file, [ins])
        file.close()
    with open('./identification/profiling_data/testrun_temps.txt', 'ba') as file:
        np.savetxt(file, [ts])
        file.close()


def gather_data_fourier():
    rpi_uart.write_cmd('reset')
    print('resetting', flush=True)
    time.sleep(1)
    connect_tmps()

    while(True):
        st = time.time()
        inputs = np.sqrt(np.cos(np.array(params['frequencies']) * 2*np.pi * st)/2 + 1/2) * 100
        temps = read_temperatures()
        set_heaters(inputs)
        write_temps_inputs(temps, inputs)
        dc.wrapper.refresh()
        dc.wrapper.clear()
        dt = time.time() - st
        if(dt < 1.0):
            time.sleep(1.0 - dt)


def gather_data_step():
    rpi_uart.write_cmd('reset')
    print('resetting', flush=True)
    time.sleep(1)
    connect_tmps()

    heater_index = -1
    heating = False
    while(True):
        st = time.time()
        inputs = np.zeros(len(params['sensor_addresses']))
        if(heating):
            inputs[heater_index] = params['test_power']
        temps = read_temperatures()
        set_heaters(inputs)
        write_temps_inputs(temps, inputs)
        dc.wrapper.refresh()
        dc.wrapper.clear()

        heating = not(heating)
        if(heating):
            heater_index += 1
        if(heater_index == len(params['sensor_addresses'])):
            break # we're done

        dt = time.time() - st
        if(dt < params['measurement_interval']):
            time.sleep(params['measurement_interval'] - 1.0)

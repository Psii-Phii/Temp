import pico_heatercontrol as phc
import pico_pico_communication as ppc

import pico_uart

from machine import I2C, reset, soft_reset, Pin
import machine
from time import localtime, ticks_ms, ticks_diff, time
from time import sleep_ms as delay

led_pin = Pin(25, Pin.OUT)

def on_lightshow():
    for i in range(5):
        led_pin.on()
        delay(125)
        led_pin.off()
        delay(125)

# get reset cause
cause_enum = machine.reset_cause()
match cause_enum:
    case machine.PWRON_RESET:
        reason='PWRON_RESET'
    #case machine.HARD_RESET:
        #reason='HARD_RESET'
    case machine.WDT_RESET:
        reason='WDT_RESET'
    case machine.DEEPSLEEP_RESET:
        reason='DEEPSLEEP_RESET'
    case machine.SOFT_RESET:
        reason='SOFT_RESET'
pico_uart.write_raw('Starting!')
pico_uart.write_raw('Reason for reset: ' + reason)
print('Waiting 3000ms for slaves to wake up')
delay(3000)
print('Starting!')
on_lightshow()

slave_bus = I2C(scl=Pin(5), sda=Pin(4), freq=400000) # to talk to other Picos
heater_bus = I2C(scl='Y9', sda='Y10', freq=400000) # to talk to the heaters

# note all enabled picos, stored in sequence
slaves = slave_bus.scan()
print(slaves)
def measure_temperature(params):
    '''Three parameters, the address of the slave to read from, the bundle index, and the I2C address (72-75) of the sensor to read from. Sends back temperature data'''
    try:
        slave_add = int(params[0])
    except ValueError:
        pico_uart.write_raw(params[0] + ' (slave address) could not be interpreted as an integer. failed to read temperature.')
        return
    try:
        bundle_add = int(params[1])
    except ValueError:
        pico_uart.write_raw(params[1] + ' (bundle index) could not be interpreted as an integer. failed to read temperature.')
        return
    try:
        tmp_add = int(params[2])
    except ValueError:
        pico_uart.write_raw(params[2] + ' (TMP address) could not be interpreted as an integer. failed to read temperature.')
        return

    if(slave_add in slaves):
        temp = ppc.read_from_pico(slave_bus, slave_add, tmp_addresses=[ [ bundle_add, tmp_add ] ])
        tempstr = str(temp)
        pico_uart.write_data(tempstr)
    else:
        pico_uart.write_raw(f'failed to find sensor at address {params[0]}:{params[1]},{params[2]}')
        return
    if temp is None:
        return
    return

# Not strictly necessary, so not implemented currently as I'd have to rewrite C bindings.
#def tmp_get_cfg(params):
#    '''One parameter - the I2C address of the sensor to read the configuration status from. Sends back the configuration (as an integer)'''
#    try:
#        tmp_add = int(params[0])
#    except ValueError:
#        pico_uart.write_raw(params[0] + ' could not be interepreted as an integer. failed to read config.')
#        return
#    
#    if tmp_add in tmp_adds:
#        t = tmps[tmp_adds.index(tmp_add)]
#        config = t.read_config()
#        configstr = str(config)
#        pico_uart.write_data(configstr)
#    else:
#        pico_uart.write_raw('failed to find sensor at I2C address ' + params[0])
#        return
#    return
#
#def tmp_set_cfg(params):
#    '''Two parameters - the I2C address of the sensor to write to the configuration register on, and the values to write (as an integer)'''
#    try:
#        tmp_add = int(params[0])
#    except ValueError:
#        pico_uart.write_raw(params[0] + ' could not be interepreted as an integer. failed to write config.')
#        return
#
#    try:
#        config = int(params[1])
#    except ValueError:
#        pico_uart.write_raw(params[1] + ' could not be interpreted as an integer. failed to write config')
#        return
#
#    if tmp_add in tmp_adds:
#        t = tmps[tmp_adds.index(tmp_add)]
#        config = t.write_config(config)
#    else:
#        pico_uart.write_raw('failed to find sensor at I2C address ' + params[0])
#        return
#    return
#
#def tmp_set_offset(params):
#    '''Two parameters - the I2C address of the sensor, and the temperature offset to set (degC)'''
#    try:
#        tmp_add = int(params[0])
#    except:
#        pico_uart.write_raw(params[0] + ' could not be intepreted as an integer. failed to set offset')
#        return
#
#    try:
#        offset = float(params[1])
#        LSBs = int(offset/0.0078125) # num of LSBs that represent offset
#    except:
#        pico_uart.write_raw(params[1] + ' could not be interpreted as a float. failed to set offset')
#        return
#    
#    if tmp_add in tmp_adds:
#        t = tmps[tmp_adds.index(tmp_add)]
#        t.write_offset(LSBs)
#    else:
#        pico_uart.write_raw('failed to find sensor at I2C address ' + str(tmp_add))
#        return
#    return
#
#def tmp_get_offset(params):
#    '''One parameter - the I2C address of the sensor. Returns number in address (LSBs, not degC)'''
#    try:
#        tmp_add = int(params[0])
#    except:
#        pico_uart.write_raw(params[0] + ' could not be intepreted as an integer. failed to set offset')
#        return
#    
#    if tmp_add in tmp_adds:
#        t = tmps[tmp_adds.index(tmp_add)]
#        lsbs=t.read_offset()
#        pico_uart.write_data(str(lsbs))
#    else:
#        pico_uart.write_raw('failed to find sensor at I2C address ' + str(tmp_add))
#        return
#    return
#

# This one is entirely unnecessary now, as the slaves do this work. Thanks, guys!
#def connect_tmp(params):
#    '''Takes one parameter - the I2C address of the sensor to connect. Connects and stores a tmp117.'''
#    try:
#        add = int(params[0])
#        trueadd = add
#    except ValueError:
#        pico_uart.write_raw(params[0] + ' could not be interpreted as an integer. failed to connect tmp117.')
#        return
#    bus = i2c_bus
#    
#    if 75 < int(params[0]) <= 79:
#        # use bus 2, starting again at 72
#        trueadd -= 4
#        bus = i2c_bus_2
#    elif (int(params[0]) > 79) or (int(params[0]) < 72):
#        pico_uart.write_raw('failed to connect tmp117 at I2C address ' + params[0] + '. Address should be in 72-79 inc.')
#        return
#
#    # make sure we haven't done this before
#    if add in tmp_adds:
#        pico_uart.write_raw('failed to connect tmp117 at address ' + params[0] + ': tmp already connected.')
#        return
#    
#    # connect the actual sensor
#    t = pico_tmp117.connect_tmp117(bus, trueadd)
#    if not(t is None):
#        pico_uart.write_raw('tmp117 connected at port ' + params[0]) # success!
#        tmp_adds.append(add)
#        tmps.append(t)
#    else:
#        pico_uart.write_raw('tmp117 did not connect at port ' + params[0]) # failure
#    return

def powercycle(params):
    reset()

def softreset(params):
    soft_reset()

def echo(params):
    pico_uart.write_raw(params[0])

cmd_handles = { 'measure_temp':(3, measure_temperature),
                #'connect_tmp':(1, connect_tmp),
                #'tmp_set_cfg':(2, tmp_set_cfg),
                #'tmp_get_cfg':(1, tmp_get_cfg),
                #'tmp_set_offset':(2, tmp_set_offset),
                #'tmp_get_offset':(1, tmp_get_offset),
                'heater':(2, pico_heatercontrol.set_heater),
                'powercycle':(0, powercycle),
                'reset':(0, softreset),
                'echo': (1, echo) } #name, (num params, callable)

last_t = time() # seconds since epoch

while True:
    if time() != last_t:
        last_t = time()
        LED(2).toggle()
    # commands are parsed based on the form: name:param1:param2:...
    # command loop
    incoming = pico_uart.read_serial_buffer()
    # we don't handle data here, really
    cmds = incoming[0]
    for cmd in cmds:
        args = []
        if ':' in cmd:
            cmd = cmd.split(':')
            if ',' in cmd[1]:
                args = cmd[1].split(',')
            else:
                args.append(cmd[1])
            cmd = cmd[0]
        if cmd in cmd_handles: # check to make sure the cmd exists
            # it does!
            cmd_handle = cmd_handles[cmd]
            if len(args) == cmd_handle[0]: #num params plus the name
                # correct number of params. Run it!
                try:
                    cmd_handle[1](args)
                except Exception as e:
                    pico_uart.write_raw('Failed to run command (0):\n' + str(e))
                    with open('errors.log', 'a') as file:
                        file.write('failed ' + str(cmd) + ' in ' + str(incoming) + '\n')
                        file.close()
            else:
                pico_uart.write_raw('Failed to parse command (1)')
                with open('errors.log', 'a') as file:
                    file.write('failed parse ' + str(cmd) + ' in ' + str(incoming) + '\n')
                    file.close()
        else:
            pico_uart.write_raw('Failed to find command (2)')
            with open('errors.log', 'a') as file:
                file.write('failed find ' + str(cmd) + ' in ' + str(incoming) + '\n')
                file.close()
            
        









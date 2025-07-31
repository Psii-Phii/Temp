from machine import SPI, Pin
from math import ceil

import pico_uart as uart

spi = SPI(0, baudrate=15_000_000) # 15MHz is limit for the TLC5947 in cascading mode (what we have)
cs = Pin(17, Pin.OUT, value=1) # chip select on GPIO17, defaults to HIGH because that disables all chips.

def begin_transmission():
    cs(0) # pull low to show the chips that we're transmitting

def end_transmission():
    cs(1) # release chip select

max_index = 24*4 # corresponds to the number of channels. Technically the max index is max_index-1, but whatever.
high_value = (2**12)-1 # the highest value that can be written into registers. For the TLC5947, this is 4095 (or 2^12-1, as each channel has 12 bits)

heater_states = [ 0 for i in range(max_index) ]
fmt_string = ''.join(['{'+str(max_index-i)+':012b}' for i in range(1,max_index+1)]) # to generate the binary in the necessary order (MSB first, and last channel first)

def set_heater_virtual(params):
    '''Takes two parameters: heater ID (less than the max ID, specified by max_index), then PWM DC (from 0-1).'''
    # parse everything
    if(len(params) > 2):
        uart.write_raw('Exception in set_heater_virtual: too many arguments.')
        return
    elif(len(params) < 2):
        uart.write_raw('Exception in set_heater_virtual: not enough arguments.')
        return
    try:
        hid = int(params[0])
    except:
        uart.write_raw(f'Exception in set_heater_virtual: could not parse ID parameter {params[0]} as an integer')
        return
    if(hid >= max_index or hid < 0):
        uart.write_raw(f'Exception in set_heater_virtual: index {hid} is outside the allowed range [0-{max_index}].')
        return
    try:
        pwm = float(params[1])
    except:
        uart.write_raw(f'Exception in set_heater_virtual: could not parse PWM duty cycle {params[1]} as a float')
        return
    if(pwm > 1.0 or pwm < 0.0):
        uart.write_raw(f'Warning in set_heater_virtual: PWM must be in range [0-1]. Got {pwm}. Clamping to valid range')
        pwm = 1.0 if pwm > 1.0 else 0.0
    
    # we can actually set the heater now.
    heater_states[hid] = pwm * high_value

def push_heater_states(params):
    '''Takes no parameters. Simply pushes all heater states to the actual SPI chips from virtual stored values.'''
    try:
        begin_transmission()
        spi.write(int(fmt_string.format(*heater_states), 2).to_bytes(max_index * 12 // 8, 'big'))
    except Exception as e:
        uart.write_raw(f'Exception in push_heater_states: failed to write bytes.\n{str(e)}\n{heater_states}')
    finally:
        end_transmission() # always release, even if we fail
    
def set_heater(params):
    '''Takes two parameters: heater ID (less than the max ID, specified by max_index), then PWM DC (from 0-1).'''
    set_heater_virtual(params)

    # write to SPI
    push_heater_states(())

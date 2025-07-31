
from machine import PWM, Pin

import pico_uart

# init of pwm for heaters on side panel
pwm_freq = 1000
max_duty = 65535 #uint16 max


heater_list = [
        PWM(Pin(10), freq=pwm_freq, duty_u16=0),
        PWM(Pin(11), freq=pwm_freq, duty_u16=0),
        PWM(Pin(12), freq=pwm_freq, duty_u16=0),
        PWM(Pin(13), freq=pwm_freq, duty_u16=0),
        PWM(Pin(14), freq=pwm_freq, duty_u16=0),
        PWM(Pin(15), freq=pwm_freq, duty_u16=0) 
        ]

# setting the heaters
# heater_list[1].pulse_width_percent(50.3)

def set_heater(params):
    '''Takes 2 parameters: 1. heater ID (0-5), and pwm'''
    try:
        heater_num = int(params[0])
        pwm = float(params[1])
        assert(pwm <= 1.0 and pwm >= 0.0)
        heater_list[heater_num].duty_u16(int(pwm*max_duty))
    except ValueError:
        pico_uart.write_raw('failed heater command: pwm not a float between 0 (0%) and 1 (100%)')
    except IndexError:
        pico_uart.write_raw('failed heater command: heater out of range (0-' + str(len(heater_list)-1) + ')')
    except:
        # other, not sure
        pico_uart.write_raw('failed heater command: unsure - check that there are the right number of parameters')


from pyb import Timer, Pin

import pyb_uart

# init of pwm for heaters on side panel
pwm_freq = 1000
timer_list = [ Timer(5, freq=pwm_freq),
        Timer(2, freq=pwm_freq),
        Timer(9, freq=pwm_freq),
        Timer(13, freq=pwm_freq),
        Timer(14, freq=pwm_freq) ]

heater_list = [timer_list[0].channel(1, Timer.PWM, pin=Pin('X1')),
          timer_list[1].channel(2, Timer.PWM, pin=Pin('X2')),
          timer_list[2].channel(1, Timer.PWM, pin=Pin('X3')),
          timer_list[3].channel(1, Timer.PWM, pin=Pin('X7')),
          timer_list[4].channel(1, Timer.PWM, pin=Pin('X8')),
          timer_list[0].channel(4, Timer.PWM, pin=Pin('X4'))] # don't mind weird ordering, that's just a legacy wiring thing.

# setting the heaters
# heater_list[1].pulse_width_percent(50.3)

def set_heater(params):
    '''Takes 2 parameters: 1. heater ID (0-5), and pwm'''
    try:
        heater_num = int(params[0])
        pwm = float(params[1])
        heater_list[heater_num].pulse_width_percent(pwm)
    except ValueError:
        pyb_uart.write_raw('failed heater command: pwm not a float')
    except IndexError:
        pyb_uart.write_raw('failed heater command: heater out of range (0-' + str(len(heater_list)-1) + ')')
    except:
        # other, not sure
        pyb_uart.write_raw('failed heater command: unsure - check that there are the right number of parameters')

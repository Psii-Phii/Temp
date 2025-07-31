from time import sleep
from machine import Pin

led_pin = Pin(25, Pin.OUT)
def led_flash():
    for i in range(8):
        led_pin.toggle()
        sleep(0.075)

led_pin.on()
sleep(1) # for cancel-safety
led_flash()

import i2c_slave as slave
from machine import SoftI2C as soft
from pico_tmp117 import connect_tmp117

################################################### CUSTOM DEFINITIONS

SLAVE_ADDRESS = 0x17
bundle_freq = 400_000
bundle_timeout = 100

################################################### CUSTOM DEFS END

bundle_busses = []
bundle_sensors = []
tmps = []
scls = [1,3,7,9,11,13,15,17,19,21,27] # technically 5 is in there, too, but that's our slave pin (for I2C0).
sdas = [ i-1 for i in scls ]

# initialize software masters
for i in range(len(scls)):
    print(f'Opening bus {i}...')
    bundle_busses += [soft(scls[i], sdas[i], freq=bundle_freq, timeout=bundle_timeout)]

print('Busses opened.\n')

# initialize TMPs on each bundle
for i in range(len(bundle_busses)):
    print(f'Connecting sensors for bundle {i}...')
    sensors = []
    for j in range(4):
        addr = j+72
        sensor = connect_tmp117(bundle_busses[i], addr, retries=0)
        sensors += [ sensor ]
        if(sensor is None):
            print(f'Failed to connect sensor with address {addr}')
        else:
            print(f'Connected sensor with address         {addr}')
    bundle_sensors += [sensors]

print('Sensors connected.\n')

# initialize slave (address 1)
print('Initializing slave...')
slave.init(0, SLAVE_ADDRESS)
print('Slave initialized.\n')

led_flash()

print('Beginning main loop...')
count = 0
while(True):
    if(slave.is_read_required()):
        led_pin.off()
        # The master has requested data. We will provide.
        # read the tmp117
        #print('Reading')
        bitTemp = 0.0078125
        temp_sensor = bundle_sensors[slave.get_bundle()][slave.get_sensor()]
        #print(f'Found temperature sensor at bundle {slave.get_bundle()}, sensor {slave.get_sensor()}')
        if not(temp_sensor is None):
            temp = temp_sensor.read_temp()
            if(temp is None):
                temp = 0xFF # slightly different error code. Again, probably too hot to feel
            slave.set_temperature_ticks(int(temp//bitTemp)) # this could probably be done more intelligently, but it should work for now
            #print(f'Read temperature {temp} for ticks {int(temp//bitTemp)}')
        else:
            slave.set_temperature_ticks(1) # de-facto error code. Note that we can actually get this in reality, too, but it corresponds to like -270 degrees or something similar. So, if we can feel our fingers (or the surrounding air is gaseous), this is an error code

        slave.set_read_required(False) # tell ourselves we no longer need to read.
        led_pin.on()


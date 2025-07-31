# This file exists entirely so that we can read temperatures from other Picos, acting as I2C slaves.
# Note that the I2C frequency for the Picos is hard set to 400kHz (hardware limit afaik)

from time import sleep
from machine import SoftI2C, Pin

possible_addresses = [ [ i//4, 72+i%4 ] for i in range(44) ] # this corresponds to all the possible sensors for a single pico

def read_from_pico(pico_bus, pico_addr, tmp_addresses=possible_addresses):
    '''Reads a bunch of sensors, and returns a list of temperatures.'''
    # write the command to retrieve temp data
    return_val = []
    for addr in tmp_addresses[i]:
        bundle_addr = addr[0]
        sensor_addr = addr[1]
        val = (bundle_addr << 4) + (sensor_addr-72)
        pico_bus.writeto(pico_addr, val.to_bytes(1,'big'))
        sleep(0.0015)
        bitTemp = 0.0078125
        val = pico_bus.readfrom(pico_addr, 2)
        temp = int(val[0] << 4) + int(val[1])
        temp *= bitTemp
        return_val += [temp]
    return return_val

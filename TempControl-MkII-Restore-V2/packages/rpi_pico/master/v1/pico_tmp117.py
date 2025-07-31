from machine import I2C,Pin
from time import sleep_ms as delay
import ustruct

import pico_uart

class tmp117(object):
    bitTemp = 0.0078125

    def __init__(self, bus, add):
        '''Takes a bus (I2C object) and an address (72 for neutral, 73-75 for others - see tmp117 docs)'''
        self.add = add
        self.bus = bus
    
    def set_EEPROM_lock(self, state):
        '''Either unlocks (1) or locks (0) the EEPROM on the TMP117 chip.'''
        self.bus.writeto_mem(self.add, 0x04, bytes((0x80*state, 0x00)))
        delay(100) # delay for 100ms for safety
    
    def power_reset(self):
        '''Issues a general call reset'''
        self.bus.writeto_mem(self.add, 0x00, bytes((0x00, 0b00000110)))
        delay(100)

    def read_temp(self):
        try:
            self.bus.writeto(self.add, bytearray([0]))  # pointer register to temp register
            raw = self.bus.readfrom(self.add, 2)  # read 2 bytes of temp register
            temp = self.bitTemp*ustruct.unpack('>h', raw)[0]#(raw[0]*256 + raw[1])
            return temp
        except Exception as e:
            pico_uart.write_raw('Failed: '+ str(e))
            return None
        return None

    def read_config(self):
        try:
            self.bus.writeto(self.add, bytearray([1]))  # pointer register to config register
            raw = self.bus.readfrom(self.add, 2)  #read 2 bytes of config register
            return raw[0]*256 + raw[1]
        except Exception as e:
            pico_uart.write_raw('Failed: ' + str(e))
            return None
        return None
    
    def write_config(self, config):
        '''Writes a configuration to the configuration register'''
        try:
            self.set_EEPROM_lock(1) # unlock EEPROM
            self.bus.writeto_mem(self.add, 0x01, bytes(((config >> 8) & 0xFF, config & 0xFF))) # writes the actual configuration in byte chunks
            self.power_reset() # lock EEPROM
            return config
        except Exception as e:
            pico_uart.write_raw('Failed: ' + str(e))
            return None
        return None
    
    def write_offset(self, offset):
        '''Writes to the offset register (offset in LSBs (0.0078125K each))'''
        try:
            offseti = int(offset)
            self.set_EEPROM_lock(1) # unlock
            bytestowrite = ustruct.pack('>h', offseti)
            self.bus.writeto_mem(self.add, 0x07, bytestowrite) # write the data to the register
            self.power_reset() # lock up again

            return offseti
        except Exception as e:
            pico_uart.write_raw('Failed: ' + str(e))
            return None
        return None

    def read_offset(self):
        '''Reads the offset register (in LSBs)'''
        try:
            raw = self.bus.readfrom_mem(self.add, 0x07, 2)
            offset = ustruct.unpack(">h", raw)[0]

            return offset
        except Exception as e:
            pico_uart.write_raw('Failed: ' + str(e))
            return None
        return None

def connect_tmp117(bus, add, delay_millis=500, retries=3):
    '''returns a tmp117 at the address - if not found, the next available address. If none can be connected, tries again.'''
    sensor = None
    count = 0
    while sensor == None:
        adds = bus.scan()
        if add in adds:
            pico_uart.write_raw("Connecting to address " + str(add))
            return tmp117(bus, add)
        else:
            count += 1
            if count > retries:
                pico_uart.write_raw("Failed to connect tmp117 at port " + str(add) + ". Returning None")
                return None
            pico_uart.write_raw("Failed to connect tmp117 at port " + str(add) + ". Retrying (" + str(count) + f"/{retries})")
            sensor = None
        pico_uart.write_raw(f"Delaying {delay_millis}ms to try again")
        delay(delay_millis)

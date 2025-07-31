
import pico_uart

# the association of each I2C connection (Cypress-TMP bundle) with the pin
# follows form bundle#:pin# (being 8*port+bit)
sda_pins = {
    2:1,
    3:3,
    4:5,
    5:7,
    6:8*2+7,
    7:8*2+3,
    8:8*4+1,
    9:8*4+3,
    10:8*4+5,
    11:8*4+7,
    12:8*6+1,
    13:8*6+3,
    14:8*6+5,
    15:8*6+7,
    16:8*1+0,
    17:8*1+2,
    18:8*1+4,
    19:8*1+6,
    20:8*3+0,
    21:8*3+2,
    22:8*3+4,
    23:8*3+6,
    24:8*5+0,
    25:8*5+2,
    26:8*5+4,
    27:8*5+6,
    28:8*7+0,
    29:8*7+2,
    30:8*7+4,
    31:8*7+6
}

# the association of each I2C connection (Cypress-TMP bundle) with the pin
# follows form bundle#:pin# (being 8*port+bit) 
scl_pins = {
    2:0,
    3:2,
    4:4,
    5:6,
    6:8*2+6,
    7:8*2+2,
    8:8*4+0,
    9:8*4+2,
    10:8*4+4,
    11:8*4+6,
    12:8*6+0,
    13:8*6+2,
    14:8*6+4,
    15:8*6+6,
    16:8*1+1,
    17:8*1+3,
    18:8*1+5,
    19:8*1+7,
    20:8*3+1,
    21:8*3+3,
    22:8*3+5,
    23:8*3+7,
    24:8*5+1,
    25:8*5+3,
    26:8*5+5,
    27:8*5+7,
    28:8*7+1,
    29:8*7+3,
    30:8*7+5,
    31:8*7+7
}

combined_valid_gpio_port_addresses = list(sda_pins.values()) + list(scl_pins.values())
combined_valid_gpio_port_addresses.sort()

class cypress_multi_test:
    def __init__(self, bus):
        self.bus = bus
        self.address = 32 # defualt

        sda = 0xaa
        scl = 0x55
        self.state_bytes = [[ b'\x00', scl.to_bytes(1, 'big') ], [ sda.to_bytes(1, 'big'), (scl+sda).to_bytes(1, 'big') ]]
        
        self.sda_state = 1
        self.scl_state = 1
    def write_state(self):
        self.bus.writeto_mem(self.address, 0x08, self.state_bytes[self.sda_state][self.scl_state])
    def set_sda(self, state):
        self.sda_state = state
        self.write_state()
    def set_scl(self, state):
        self.scl_state = state
        self.write_state()
    def read_sda(self):
        raw = self.bus.readfrom_mem(self.address, 0x0, 1) # just the first byte
        data = []
        for i in range(len(raw)):
            for j in [1,3,5,7]:
                data += [((raw[i] >> j) & 0x1)]
        return data

class cypress_multi:
    def __init__(self, bus, address):
        self.bus = bus
        self.address = address
        self.pinstates = {}
        for i in combined_valid_gpio_port_addresses:
            self.pinstates[i] = 1

        # set all gpios to open drain low (pulls low when set low, but leaves bus "open" on set high)
        #for i in [ b'\x00', b'\x01', b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07' ]:
        #    self.bus.writeto_mem(self.address, 0x18, i) # set gport target
        #    self.bus.writeto_mem(self.address, 0x20, '\xff') # actually let that gport be open drain low
    
    def write_pins(self, pin_values):
        '''Takes a 60-value dictionary (key=bit address, value = 1 or 0). Sets the pins to HIGH for 1s, LOW for 0s.'''
    
        # the register sizes are:
        # GPIO 0,1,3,4,5,6,7: 8bit
        # GPIO 2: 4bit

        bytes_to_write = [ 0 for i in range(8) ]
        for i in pin_values:
            byte_index = i // 8
            bit_index = i % 8
            pin_val = pin_values[i]
            self.pinstates[i] = pin_val
            bytes_to_write[byte_index] += 1<<bit_index if pin_val>0 else 0
        self.bus.writeto_mem(self.address, 0x08, bytes(bytes_to_write))
        return

    def read_pins(self):
        '''Returns a 60-long buffer of pin values in the order P0b0,P0b1,...,etc. Values are 1s (if HIGH) or 0s (if LOW).'''
        #raw = self.bus.readfrom_mem(self.address, 0x08, 8)
        #raw = self.bus.readfrom_mem(self.address, 0x20, 8)
        #print(self.bus.writeto(self.address, b'\x00'))
        #print(self.bus.writeto(self.address, b'\x21'))
        raw = self.bus.readfrom_mem(self.address, 0x0, 8)
        print(raw)
        #unpack bytes
        data = {}
        for b in range(len(raw)):
            for i in range(8):
                #if (b*8 + i) in combined_valid_gpio_port_addresses: # gpio2 is 4 bits wide
                data[b*8 + i] = ((raw[b] >> i) & 0x1)
        self.pinstates = data
        return data

    def set_scl(self, state):
        '''Sets the SCL on every TMP117 to either high (state==1) or low (state==0)'''
        for i in scl_pins.values():
            self.pinstates[i] = state
        self.write_pins(self.pinstates)

    def set_sda(self, state):
        '''Sets the SDA on every TMP117 to either high (state==1) or low (state==0)'''
        for i in sda_pins.values():
            self.pinstates[i] = state
        self.write_pins(self.pinstates)

    def read_sda(self):
        self.pinstates = self.read_pins()
        returned = []
        for i in sda_pins.values():
            returned += [self.pinstates[i]]
        return returned

    def reset_state(self, state):
        '''Resets every pin to state'''
        for i in combined_valid_gpio_port_addresses:
            self.pinstates[i] = state
        self.write_pins(self.pinstates)

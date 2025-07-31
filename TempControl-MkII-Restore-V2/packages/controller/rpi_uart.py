# purposed for the unification of heaters and temp sensors over UART RX/TX

import serial
import time

UART_TTY='/dev/ttyUSB0' # the address of the UART in/out
# ttyAMA0 on Raspberry Pi
# ttyS2 on ROCK64

def get_serial_obj():
    return serial.Serial(UART_TTY, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, baudrate=9600, timeout=1, inter_byte_timeout=1.0, )
   
ser = get_serial_obj()

def read_serial_buffer():
    '''Reads in, clears the buffer, and returns a tuple: a list of cmds, a list of data, and a list of other.'''
    cmd = []
    data = []
    other = []
    while ser.in_waiting != 0:
        line = str(ser.readline(), 'utf-8')
        if line[0:3] == '%C%':
            cmd.append(line[3:-1])
        elif line[0:3] == '%D%':
            data.append(line[3:-1])
        else:
            other.append(line[:-1])
    return (cmd, data, other)

def print_serial_buffer():
    '''Reads in, clears the buffer, prints all that has been read, and returns a list of each line that was read.'''
    out = read_serial_buffer()
    cmds = out[0]
    data = out[1]
    other = out[2]
    for l in cmds:
        print('%C%'+l, flush=True)
    for l in data:
        print('%D%'+l, flush=True)
    for l in other:
        print(l, flush=True)
    return out


def denewline(x):
    '''Removes trailing newline characters. Not the best complexity'''
    while x[-1] == '\n':
        x=x[:-1]
    return x

def hold_for_empty():
    '''Holds the execution until all data has been sent out.'''
    ser.flush()

def write_cmd(cmd):
    '''Appends a newline, as well as a unique sequence for commands at the front (%C%)'''
    hold_for_empty()
    fullstr = str('%C%' + denewline(cmd) + '\n').encode('utf-8')
    ser.write(fullstr)
    hold_for_empty()
    
def write_data(data):
    '''Appends a newline, as well as a sequence unique to data at the front (%D%)'''
    hold_for_empty()
    fullstr = str('%D%' + denewline(data) + '\n').encode('utf-8')
    ser.write(fullstr)
    hold_for_empty()
    
def write_raw(x):
    '''Appends a newline, but everything else is custom.'''
    hold_for_empty()
    fullstr = str(denewline(x) + '\n').encode('utf-8')
    ser.write(fullstr)
    hold_for_empty()

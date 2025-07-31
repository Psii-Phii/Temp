from machine import UART

def get_serial_object():
    uart = UART('YA') # opens UART TX/RX on Y1/Y2
    uart.init(9600, bits=8, parity=None, stop=1, timeout_char=1000, timeout=1000, read_buf_len=256) # wait 1000ms between each incoming character
    return uart
#112500

try: # test serial initialization
    ser
except NameError:
    ser = get_serial_object()

def read_serial_buffer():
    '''Reads in, clears the buffer, and returns a tuple: a list of cmds, a list of data, then a list of other.'''
    cmds = []
    data = []
    other = []
    while ser.any() != 0:
        try:
            line = str(ser.readline(), 'utf-8') #read in until a newline character, and decode
            if line[0] == '%':
                if line[1] == 'D': #data
                    data.append(line[3:-1])
                elif line[1] == 'C':
                    cmds.append(line[3:-1])
            else:
                other.append(line[:-1])
        except Exception as e:
            write_raw('error in read_serial_buffer: ' + str(e))
            return (cmds, data, other)
    return (cmds, data, other)

def get_input():
    '''Blocks until we recieve input, terminated by a newline.'''
    while ser.any() == 0:
        continue # wait for input
    inp = str(ser.readline(), 'utf-8')
    return inp[:-1]

def denewline(x):
    '''Removes trailing newline characters. Not the best complexity'''
    while x[-1] == '\n':
        x=x[:-1]
    return x

def write_cmd(cmd):
    '''Appends a newline, as well as a unique sequence for commands at the front (%C%)'''
    fullstr = str('%C%' + denewline(cmd) + '\n').encode('utf-8')
    ser.write(fullstr)
    
def write_data(data):
    '''Appends a newline, as well as a sequence unique to data at the front (%D%)'''
    fullstr = str('%D%' + denewline(data) + '\n').encode('utf-8')
    ser.write(fullstr)
    
def write_raw(x):
    '''Appends a newline, but everything else is custom.'''
    fullstr = str(denewline(x) + '\n').encode('utf-8')
    ser.write(fullstr)

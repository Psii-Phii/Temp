import modules.controller.rpi_uart as rpi_uart

import time

while True:
    print('Listening. Press CTRL+C to input command...')
    while True:
        try:
            rpi_uart.print_serial_buffer()
        except KeyboardInterrupt: # yes, this is the best way to no-block listen for input
            break
        except:
            print('failed to read serial')
    cmd = input("Enter command\n")
    rpi_uart.write_cmd(cmd)
    

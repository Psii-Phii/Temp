# this script exists to make creating profiling, well, 'profiles' easier. Just run and answer the questions, it'll do the hard work

import os

default_profile_step = { 'rise_time': 60*60*2, # the amount of time to turn heaters on for
                          'settle_time': 60*60*5, # the amount of time to let them cool for
                          'measurement_interval': 1, # the time in seconds between measurements
                          'sensor_addresses': [ [ i//4, 72+i%4 ]  for i in range(10) ], # the sensor addresses, in the format prescribed in networking documentation
                          'test_power': 0.7, # the power to put into the heaters (this is PWM duty cycle, so not technically power)
                         }

default_profile_fourier = { 'measurement_interval': 1,
                            'test_power': 1.0, # max ampllitude, in PWM
                            'sensor_addresses': [ [ i//4, 72+i%4 ] for i in range(10) ],
                            'frequencies': [ (i+1)/3600 for i in range(10) ], # the frequencies in Hz of the oscillations
                           }

profiles_path = './identification/profiles'

def save_profile(profile):
    fn = None

    while(fn is None):
        fn = input('Profile name: ')

        directory_list = os.listdir(profiles_path)
        if(fn in directory_list):
            option = input(f'Profile {fn} already exists. Overwrite? [y/N])')
            if (option != 'y') and (option != 'Y'):
                #retry
                fn = None

    os.mkdir(f'{profiles_path}/{fn}')
    with open(f'{profiles_path}/{fn}/config.py', 'w') as file:
        file.write(f'def read_pickle():\n\treturn {str(profile)}')
        print('Profile written.')
        file.close()
        print('Success')

def verify_input(prompt, options):
    var = None
    fullprompt = prompt + f' ({options[0]}'
    for i in options[1:]:
        fullprompt += ', ' + i
    fullprompt += '): '
    while(var is None):
        var = input(fullprompt)
        if not(var in options):
            print('Sorry, that is not an option.')
            var = None
    return var

def verify_double(prompt):
    var = None
    while(var is None):
        var = input(prompt + ': ')
        try:
            var = float(var)
        except:
            var = None
            print('Sorry, that wasn\'t a valid number.')
    return var

print('\nWelcome! Please answer the following questions.\n')

profile_type = verify_input('Profile type', ['fourier', 'step'])

if(profile_type == 'step'):
    profile = default_profile_step
    profile['rise_time'] = verify_double('Rise time (s)')
    profile['settle_time'] = verify_double('Settle time (s)')
    profile['test_power'] = verify_double('PWM proportion (0-1)')
    profile['measurement_interval'] = verify_double('Measurement interval (s)')

save_profile(profile)

        

import numpy as np
import time
import matplotlib.pyplot as plt
import random
# get gekko package with:
#   pip install gekko
from gekko import GEKKO
# get tclab package with:
#   pip install tclab
from tclab import TCLab

# Colour scheme
COL = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
       '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

# save txt file
def save_txt(t, Q1, Q2, T1, T2):
    data = np.vstack((t, Q1, Q2, T1, T2)).T
    header = 'Time (sec), Heater 1, Heater 2, Temperature 1, Temperature 2'
    np.savetxt('data.txt', data, delimiter=',', header=header, comments='')

# Connect to Arduino
a = TCLab()

# Final time (minutes)
tf = 10
# number of data points (every 3 seconds)
n = tf * 20 + 1

# Configure heater levels
Q1s = np.zeros(n)
Q2s = np.zeros(n)
for i in range(n):
    if i % 20 == 0:
        Q1s[i:i+20] = random.random() * 100.0
    if (i+10) % 20 == 0:
        Q2s[i:i+20] = random.random() * 100.0

# heater configurations
Q2s[:50] = 0.0  # heater 2 initially off
Q1s[-50:] = 0.0  # heater 1 off at end

# Record initial temperatures (degC) from sensors
t_initial = a.T1
u_initial = a.T2
T1m = t_initial * np.ones(n)
T2m = u_initial * np.ones(n)

# Store MHE values for plots
Tmhe1 = T1m.copy()
Tmhe2 = T2m.copy()

# Parameter storage arrays
K1s = 0.5 * np.ones(n)
K2s = 0.3 * np.ones(n)
K3s = 0.005 * np.ones(n)

# Time-constant storage arrays
tau12s = 150.0 * np.ones(n)
tau3s = 5.0 * np.ones(n)

# Time vector
tm = np.zeros(n)
start_time = time.time()
prev_time = start_time

#########################################################
# Initialize Model as Estimator
#########################################################
m = GEKKO(name='tclab-mhe', remote=True)
# 120-second horizon, 40 steps
m.time = np.linspace(0, 120, 41)

# Parameters to Estimate
K1 = m.FV(value=0.5); K2 = m.FV(value=0.3); K3 = m.FV(value=0.2)
tau12 = m.FV(value=150); tau3 = m.FV(value=15)
for v in (K1, K2, K3, tau12, tau3):
    v.STATUS = 0; v.FSTATUS = 0
# Bounds and deltas
K1.DMAX=0.1; K1.LOWER=0.1; K1.UPPER=1.0
K2.DMAX=0.1; K2.LOWER=0.1; K2.UPPER=1.0
K3.DMAX=0.01; K3.LOWER=0.1; K3.UPPER=1.0
tau12.DMAX=5; tau12.LOWER=50; tau12.UPPER=200
tau3.DMAX=1; tau3.LOWER=10; tau3.UPPER=20

# Measured inputs
Q1 = m.MV(value=0); Q2 = m.MV(value=0)
Q1.FSTATUS = Q2.FSTATUS = 1

# State variables initialized from first sensor readings
TH1 = m.SV(value=t_initial)
TH2 = m.SV(value=u_initial)

# Measurement variables
TC1 = m.CV(value=t_initial)
TC2 = m.CV(value=u_initial)
for cv in (TC1, TC2):
    cv.STATUS = 1; cv.FSTATUS = 1; cv.MEAS_GAP = 0.1

# Ambient temperature as parameter (initial sensor reading)
Ta = m.Param(value=t_initial)

# Heat transfer between heaters
DT = m.Intermediate(TH2 - TH1)

# Empirical correlations
m.Equation(tau12 * TH1.dt() + (TH1 - Ta) == K1 * Q1 + K3 * DT)
m.Equation(tau12 * TH2.dt() + (TH2 - Ta) == K2 * Q2 - K3 * DT)
m.Equation(tau3 * TC1.dt()   + TC1 == TH1)
m.Equation(tau3 * TC2.dt()   + TC2 == TH2)

# Global options for MHE
m.options.IMODE    = 5
m.options.EV_TYPE  = 1
m.options.NODES    = 3
m.options.SOLVER   = 3
m.options.COLDSTART = 1

##################################################################
# Plot setup
plt.ion()
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for ax in axes:
    ax.grid(alpha=0.3)

# Main experiment loop
try:
    for i in range(1, n):
        # maintain approx. 3-second interval
        wait = 3.0 - (time.time() - prev_time)
        time.sleep(max(wait - 0.01, 0.01))
        now = time.time()
        tm[i] = now - start_time
        prev_time = now

        # Sensor readings
        T1m[i] = a.T1
        T2m[i] = a.T2

        # Insert measurements into MHE
        TC1.MEAS = T1m[i]
        TC2.MEAS = T2m[i]
        Q1.MEAS = Q1s[i-1]
        Q2.MEAS = Q2s[i-1]

        # enable estimation after warm-up (10 cycles)
        if i == 10:
            K1.STATUS = K2.STATUS = K3.STATUS = tau12.STATUS = tau3.STATUS = 1

        # solve MHE
        m.solve(disp=False)

        # store outputs
        if m.options.APPSTATUS == 1:
            Tmhe1[i] = TC1.MODEL
            Tmhe2[i] = TC2.MODEL
            K1s[i], K2s[i], K3s[i] = K1.NEWVAL, K2.NEWVAL, K3.NEWVAL
            tau12s[i], tau3s[i] = tau12.NEWVAL, tau3.NEWVAL
        else:
            Tmhe1[i] = Tmhe1[i-1]
            Tmhe2[i] = Tmhe2[i-1]
            K1s[i], K2s[i], K3s[i] = K1s[i-1], K2s[i-1], K3s[i-1]
            tau12s[i], tau3s[i] = tau12s[i-1], tau3s[i-1]

        # write heater outputs
        a.Q1(Q1s[i])
        a.Q2(Q2s[i])

        # update plots
        axes[0].cla()
        axes[0].grid(True, alpha=0.3)
        axes[0].plot(tm[:i], T1m[:i], marker='o', markersize=4, linestyle='None', color=COL[0], label=r'$T_1$ measured')
        axes[0].plot(tm[:i], Tmhe1[:i], linestyle='-', linewidth=2, color=COL[3], label=r'$T_1$ MHE')
        axes[0].plot(tm[:i], T2m[:i], marker='x', markersize=4, linestyle='None', color=COL[2], label=r'$T_2$ measured')
        axes[0].plot(tm[:i], Tmhe2[:i], linestyle='--', linewidth=2, color=COL[1], label=r'$T_2$ MHE')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend(loc='upper left', fontsize=8)

        axes[1].cla()
        axes[1].grid(True, alpha=0.3)
        axes[1].plot(tm[:i], K1s[:i]*100, linestyle='-', linewidth=2, color=COL[4], label='K1')
        axes[1].plot(tm[:i], K2s[:i]*100, linestyle=':', linewidth=2, color=COL[5], label='K2')
        axes[1].plot(tm[:i], K3s[:i]*100, linestyle='--', linewidth=2, color=COL[6], label='K3')
        axes[1].set_ylabel('Gains')
        axes[1].legend(loc='best', fontsize=8)

        axes[2].cla()
        axes[2].grid(True, alpha=0.3)
        axes[2].plot(tm[:i], tau12s[:i], linestyle='-', linewidth=2, color=COL[7], label=r'$\tau_{12}$')
        axes[2].plot(tm[:i], tau3s[:i]*10, linestyle='--', linewidth=2, color=COL[8], label=r'$\tau_3$')
        axes[2].set_ylabel('Time constants')
        axes[2].legend(loc='best', fontsize=8)

        axes[3].cla()
        axes[3].grid(True, alpha=0.3)
        axes[3].plot(tm[:i], Q1s[:i], linestyle='-', linewidth=2, color=COL[0], label=r'$Q_1$')
        axes[3].plot(tm[:i], Q2s[:i], linestyle=':', linewidth=2, color=COL[1], label=r'$Q_2$')
        axes[3].set_ylabel('Heaters (%)')
        axes[3].set_xlabel('Time (sec)')
        axes[3].legend(loc='best', fontsize=8)

        fig.tight_layout(pad=1.5)
        plt.pause(0.05)

    # shutdown
    a.Q1(0)
    a.Q2(0)
    save_txt(tm, Q1s, Q2s, T1m, T2m)
    plt.savefig('tclab_mhe.png')

except KeyboardInterrupt:
    a.Q1(0)
    a.Q2(0)
    a.close()
    print('Shutting down')
    plt.savefig('tclab_mhe.png')

except Exception as e:
    a.Q1(0)
    a.Q2(0)
    a.close()
    print(f'Error: {e}')
    plt.savefig('tclab_mhe.png')

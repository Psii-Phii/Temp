import numpy as np
import time
import matplotlib.pyplot as plt
import random
import json
# get gekko package with:
#   pip install gekko
from gekko import GEKKO
# get tclab package with:
#   pip install tclab
from tclab import TCLab

COL = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
       '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
# Connect to Arduino
a = TCLab()

# Make an MP4 animation?
make_mp4 = False
if make_mp4:
    import imageio  # required to make animation
    import os
    try:
        os.mkdir('./figures')
    except:
        pass

# Final time
tf = 10 # min
# number of data points (every 3 seconds)
n = tf * 20 + 1

# Percent Heater (0-100%)
Q1s = np.zeros(n)
Q2s = np.zeros(n)

# Temperatures (degC)
T1m = a.T1 * np.ones(n)
T2m = a.T2 * np.ones(n)
# Temperature setpoints
T1sp = T1m[0] * np.ones(n)
T2sp = T2m[0] * np.ones(n)

# Heater set point steps about every 150 sec
T1sp[3:] = 40.0
T2sp[40:] = 30.0
T1sp[80:] = 32.0
T2sp[120:] = 35.0
T1sp[160:] = 45.0

#########################################################
# Initialize Model
#########################################################
m = GEKKO(name='tclab-mpc',remote=True)

# with a local server
#m = GEKKO(name='tclab-mpc',server='http://127.0.0.1',remote=True)

# Control horizon, non-uniform time steps
m.time = [0,3,6,10,14,18,22,27,32,38,45,55,65, \
          75,90,110,130,150]

# Parameters from Estimation
K1 = m.FV(value=0.607)
K2 = m.FV(value=0.293)
K3 = m.FV(value=0.24)
tau12 = m.FV(value=192)
tau3 = m.FV(value=15)

# don't update parameters with optimizer
K1.STATUS = 0
K2.STATUS = 0
K3.STATUS = 0
tau12.STATUS = 0
tau3.STATUS = 0

# Manipulated variables
Q1 = m.MV(value=0,name='q1')
Q1.STATUS = 1  # manipulated
Q1.FSTATUS = 0 # not measured
Q1.DMAX = 20.0
Q1.DCOST = 0.1
Q1.UPPER = 100.0
Q1.LOWER = 0.0

Q2 = m.MV(value=0,name='q2')
Q2.STATUS = 1  # manipulated
Q2.FSTATUS = 0 # not measured
Q2.DMAX = 30.0
Q2.DCOST = 0.1
Q2.UPPER = 100.0
Q2.LOWER = 0.0

# State variables
TH1 = m.SV(value=T1m[0])
TH2 = m.SV(value=T2m[0])

# Controlled variables
TC1 = m.CV(value=T1m[0],name='tc1')
TC1.STATUS = 1     # drive to set point
TC1.FSTATUS = 1    # receive measurement
TC1.TAU = 40       # response speed (time constant)
TC1.TR_INIT = 1    # reference trajectory
TC1.TR_OPEN = 0

TC2 = m.CV(value=T2m[0],name='tc2')
TC2.STATUS = 1     # drive to set point
TC2.FSTATUS = 1    # receive measurement
TC2.TAU = 0        # response speed (time constant)
TC2.TR_INIT = 0    # dead-band
TC2.TR_OPEN = 1

Ta = m.Param(value=23.0) # degC

# Heat transfer between two heaters
DT = m.Intermediate(TH2-TH1)

# Empirical correlations
m.Equation(tau12 * TH1.dt() + (TH1-Ta) == K1*Q1 + K3*DT)
m.Equation(tau12 * TH2.dt() + (TH2-Ta) == K2*Q2 - K3*DT)
m.Equation(tau3 * TC1.dt()  + TC1 == TH1)
m.Equation(tau3 * TC2.dt()  + TC2 == TH2)

# Global Options
m.options.IMODE   = 6 # MPC
m.options.CV_TYPE = 1 # Objective type
m.options.NODES   = 3 # Collocation nodes
m.options.SOLVER  = 3 # IPOPT
m.options.COLDSTART = 1 # COLDSTART on first cycle
##################################################################
# Create plot
plt.figure(figsize=(10,7))
plt.ion()
plt.show()

# Main Loop
start_time = time.time()
prev_time = start_time
tm = np.zeros(n)

try:
    for i in range(1,n-1):
        # Sleep time
        sleep_max = 3.0
        sleep = sleep_max - (time.time() - prev_time)
        if sleep>=0.01:
            time.sleep(sleep-0.01)
        else:
            time.sleep(0.01)

        # Record time and change in time
        t = time.time()
        dt = t - prev_time
        prev_time = t
        tm[i] = t - start_time

        # Read temperatures in Celsius 
        T1m[i] = a.T1
        T2m[i] = a.T2

        # Insert measurements
        TC1.MEAS = T1m[i]
        TC2.MEAS = T2m[i]

        # Adjust setpoints
        db1 = 1.0 # dead-band
        TC1.SPHI = T1sp[i] + db1
        TC1.SPLO = T1sp[i] - db1

        db2 = 0.2
        TC2.SPHI = T2sp[i] + db2
        TC2.SPLO = T2sp[i] - db2

        # Adjust heaters with MPC
        m.solve() 

        if m.options.APPSTATUS == 1:
            # Retrieve new values
            Q1s[i+1]  = Q1.NEWVAL
            Q2s[i+1]  = Q2.NEWVAL
            # get additional solution information
            with open(m.path+'//results.json') as f:
                results = json.load(f)
        else:
            # Solution failed
            Q1s[i+1]  = 0.0
            Q2s[i+1]  = 0.0

        # Write new heater values (0-100)
        a.Q1(Q1s[i])
        a.Q2(Q2s[i])

        # Plot
        plt.clf()
        ax=plt.subplot(3,1,1)
        ax.grid()
        plt.plot(tm[0:i+1],T1sp[0:i+1]+db1,'k-',\
                 label=r'$T_1$ target',lw=3)
        plt.plot(tm[0:i+1],T1sp[0:i+1]-db1,'k-',\
                 label=None,lw=3)
        plt.plot(tm[0:i+1],T1m[0:i+1],'.',color = COL[0],label=r'$T_1$ measured')
        plt.plot(tm[i]+m.time,results['tc1.bcv'],'r',color = COL[0],\
                 label=r'$T_1$ predicted',lw=3)
        plt.plot(tm[i]+m.time,results['tc1.tr_hi'],'k--',\
                 label=r'$T_1$ trajectory')
        plt.plot(tm[i]+m.time,results['tc1.tr_lo'],'k--')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)
        ax=plt.subplot(3,1,2)
        ax.grid()        
        plt.plot(tm[0:i+1],T2sp[0:i+1]+db2,'k-',\
                 label=r'$T_2$ target',lw=3)
        plt.plot(tm[0:i+1],T2sp[0:i+1]-db2,'k-',\
                 label=None,lw=3)
        plt.plot(tm[0:i+1],T2m[0:i+1],'.',color = COL[1],label=r'$T_2$ measured')
        plt.plot(tm[i]+m.time,results['tc2.bcv'],'-',color = COL[1],\
                 label=r'$T_2$ predict',lw=3)
        plt.plot(tm[i]+m.time,results['tc2.tr_hi'],'k--',\
                 label=r'$T_2$ range')
        plt.plot(tm[i]+m.time,results['tc2.tr_lo'],'k--')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)
        ax=plt.subplot(3,1,3)
        ax.grid()
        plt.plot([tm[i],tm[i]],[0,100],'k-',\
                 label='Current Time',lw=1)
        plt.plot(tm[0:i+1],Q1s[0:i+1],'.-',color =COL[0],\
                 label=r'$Q_1$ history',lw=2)
        plt.plot(tm[i]+m.time,Q1.value,'-',color =COL[0],\
                 label=r'$Q_1$ plan',lw=3)
        plt.plot(tm[0:i+1],Q2s[0:i+1],'.-',color =COL[1],\
                 label=r'$Q_2$ history',lw=2)
        plt.plot(tm[i]+m.time,Q2.value,'-',color =COL[1],
                 label=r'$Q_2$ plan',lw=3)
        plt.plot(tm[i]+m.time[1],Q1.value[1],color=COL[0],\
                 marker='.',markersize=15)
        plt.plot(tm[i]+m.time[1],Q2.value[1],color=COL[1],\
                 marker='X',markersize=8)
        plt.ylabel('Heaters')
        plt.xlabel('Time (sec)')
        plt.legend(loc=2)
        plt.draw()
        plt.pause(0.05)
        if make_mp4:
            filename='./figures/plot_'+str(i+10000)+'.png'
            plt.savefig(filename)

    # Turn off heaters and close connection
    a.Q1(0)
    a.Q2(0)
    a.close()
    # Save figure
    plt.savefig('tclab_mpc.png')

    # generate mp4 from png figures in batches of 350
    if make_mp4:
        images = []
        iset = 0
        for i in range(1,n-1):
            filename='./figures/plot_'+str(i+10000)+'.png'
            images.append(imageio.imread(filename))
            if ((i+1)%350)==0:
                imageio.mimsave('results_'+str(iset)+'.mp4', images)
                iset += 1
                images = []
        if images!=[]:
            imageio.mimsave('results_'+str(iset)+'.mp4', images)

# Allow user to end loop with Ctrl-C           
except KeyboardInterrupt:
    # Turn off heaters and close connection
    a.Q1(0)
    a.Q2(0)
    a.close()
    print('Shutting down')
    plt.savefig('tclab_mpc.png')

# Make sure serial connection still closes when there's an error
except:           
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    a.close()
    print('Error: Shutting down')
    plt.savefig('tclab_mpc.png')
    raise

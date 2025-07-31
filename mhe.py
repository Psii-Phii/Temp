import numpy as np
import time
import matplotlib.pyplot as plt
import random
from gekko import GEKKO
from tclab import TCLab

# Connect to Arduino
a = TCLab()

# Make an MP4 animation?
make_mp4 = True
if make_mp4:
    import imageio
    import os
    try:
        os.mkdir('./figures')
    except OSError:
        pass

# Final time (minutes) and data points (one every 3 s)
tf = 10
n  = tf * 20 + 1

# Heater set‐points
Q1s = np.zeros(n)
Q2s = np.zeros(n)
Q1s[3:]   = 100.0;  Q1s[50:]  = 0.0;   Q1s[100:] = 80.0
Q2s[25:]  = 60.0;   Q2s[75:]  = 100.0; Q2s[125:] = 25.0
for i in range(130,180):
    if i % 10 == 0:
        Q1s[i:i+10] = random.random() * 100
    if (i+5) % 10 == 0:
        Q2s[i:i+10] = random.random() * 100

# Pre‐allocate
T1m    = a.T1 * np.ones(n)
T2m    = a.T2 * np.ones(n)
Tmhe1  = T1m[0] * np.ones(n)
Tmhe2  = T2m[0] * np.ones(n)
Umhe   = 10.0    * np.ones(n)
amhe1  = 0.01    * np.ones(n)
amhe2  = 0.0075  * np.ones(n)
Cpmhe  = 0.5*1000 * np.ones(n)

############################
# Initialize MHE model
############################
m      = GEKKO(name='tclab-mhe', remote=True)
m.time = np.linspace(0,60,21)

# estimate vars
U      = m.FV(10,    name='U');      U.STATUS=0; U.FSTATUS=0; U.LOWER=5;   U.UPPER=15;   U.DMAX=1
alpha1 = m.FV(0.01,  name='alpha1'); alpha1.STATUS=0; alpha1.FSTATUS=0; alpha1.LOWER=0.003;  alpha1.UPPER=0.03; alpha1.DMAX=0.001
alpha2 = m.FV(0.0075,name='alpha2'); alpha2.STATUS=0; alpha2.FSTATUS=0; alpha2.LOWER=0.002;  alpha2.UPPER=0.02; alpha2.DMAX=0.001
Cp     = m.FV(500.0, name='Cp');     Cp.STATUS=0; Cp.FSTATUS=0; Cp.LOWER=100; Cp.UPPER=1000; Cp.DMAX=50

# measured inputs
Q1 = m.MV(0, name='Q1'); Q1.STATUS=0; Q1.FSTATUS=1
Q2 = m.MV(0, name='Q2'); Q2.STATUS=0; Q2.FSTATUS=1

# measurements
TC1 = m.CV(T1m[0], name='TC1'); TC1.STATUS=1; TC1.FSTATUS=1; TC1.LOWER=0; TC1.UPPER=200; TC1.MEAS_GAP=0.1
TC2 = m.CV(T2m[0], name='TC2'); TC2.STATUS=1; TC2.FSTATUS=1; TC2.LOWER=0; TC2.UPPER=200; TC2.MEAS_GAP=0.1

# constants & params
Ta    = m.Param(23+273.15)
mass  = m.Param(4/1000)
A     = m.Param(10/100**2)
As    = m.Param(2/100**2)
eps   = m.Param(0.9)
sigma = m.Const(5.67e-8)

# intermediates
T1    = m.Intermediate(TC1+273.15)
T2    = m.Intermediate(TC2+273.15)
Q_C12 = m.Intermediate(U*As*(T2-T1))
Q_R12 = m.Intermediate(eps*sigma*As*(T2**4-T1**4))

# equations
m.Equation(TC1.dt() == (1/(mass*Cp))*(
    U*A*(Ta-T1) +
    eps*sigma*A*(Ta**4 - T1**4) +
    Q_C12 + Q_R12 +
    alpha1*Q1))

m.Equation(TC2.dt() == (1/(mass*Cp))*(
    U*A*(Ta-T2) +
    eps*sigma*A*(Ta**4 - T2**4) -
    Q_C12 - Q_R12 +
    alpha2*Q2))

# solver options
m.options.IMODE   = 5
m.options.EV_TYPE = 2
m.options.NODES   = 3
m.options.SOLVER  = 3
m.options.COLDSTART = 1

############################
# Plot setup
############################
COL = ['#4E79A7','#F28E2B','#E15759',
       '#76B7B2','#59A14F','#EDC948',
       '#B07AA1','#FF9DA7','#9C755F','#BAB0AC']

fig, axes = plt.subplots(3,1,
                         figsize=(12,9),
                         sharex=True)
ax_temp, ax_params, ax_heat = axes
ax_cp = ax_params.twinx()

for ax in (ax_temp, ax_params, ax_heat, ax_cp):
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)

plt.ion()
plt.show()

############################
# Main loop
############################
start = time.time()
prev  = start
tm    = np.zeros(n)

try:
    for i in range(1,n):
        # timing
        sleep = 3 - (time.time()-prev)
        time.sleep(max(sleep, 0.01))
        prev = time.time()
        tm[i] = prev - start

        # read & inject
        T1m[i] = a.T1
        T2m[i] = a.T2
        TC1.MEAS = T1m[i]
        TC2.MEAS = T2m[i]
        Q1.MEAS  = Q1s[i-1]
        Q2.MEAS  = Q2s[i-1]

        if i==10:
            U.STATUS=1; alpha1.STATUS=1; alpha2.STATUS=1

        m.solve(disp=False)

        if m.options.APPSTATUS==1:
            Tmhe1[i] = TC1.MODEL
            Tmhe2[i] = TC2.MODEL
            Umhe[i]  = U.NEWVAL
            amhe1[i] = alpha1.NEWVAL
            amhe2[i] = alpha2.NEWVAL
            Cpmhe[i] = Cp.NEWVAL
        else:
            Tmhe1[i]=Tmhe1[i-1]
            Tmhe2[i]=Tmhe2[i-1]
            Umhe[i] = Umhe[i-1]
            amhe1[i]=amhe1[i-1]
            amhe2[i]=amhe2[i-1]
            Cpmhe[i]=Cpmhe[i-1]

        # update heaters
        a.Q1(Q1s[i])
        a.Q2(Q2s[i])

        # clear + redraw
        for ax in (ax_temp, ax_params, ax_heat, ax_cp):
            ax.clear()
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=12)

        # 1) temperatures
        ax_temp.plot(tm[:i], T1m[:i],
                     'o', ms=5, lw=1,
                     color=COL[0], label=r'$T_1$ meas')
        ax_temp.plot(tm[:i], Tmhe1[:i],
                     lw=2, color=COL[3], label=r'$T_1$ MHE')
        ax_temp.plot(tm[:i], T2m[:i],
                     's', ms=5, lw=1,
                     color=COL[2], label=r'$T_2$ meas')
        ax_temp.plot(tm[:i], Tmhe2[:i],
                     lw=2, ls='--', color=COL[1],
                     label=r'$T_2$ MHE')
        ax_temp.set_ylabel('Temp (°C)', fontsize=14)
        ax_temp.legend(fontsize=12, loc='upper left')

        # 2) parameters on twin axes
        pU,   = ax_params.plot(tm[:i], Umhe[:i],
                               lw=2, color=COL[4], label=r'$U$')
        p1,   = ax_params.plot(tm[:i], amhe1[:i]*1e3,
                               lw=2, ls='--', color=COL[5],
                               label=r'$\alpha_1\times10^3$')
        p2,   = ax_params.plot(tm[:i], amhe2[:i]*1e3,
                               lw=2, ls='-.', color=COL[6],
                               label=r'$\alpha_2\times10^3$')
        pCp,  = ax_cp.plot(tm[:i], Cpmhe[:i],
                           lw=2, ls=':', color=COL[7],
                           label=r'$C_p$')
        ax_params.set_ylabel('U & α×10³', fontsize=14)
        ax_params.set_ylim(4, 16)
        ax_cp.set_ylabel('C_p (J/kg·K)', fontsize=14)
        ax_cp.set_ylim(100, 1000)
        ax_params.legend(
            [pU, p1, p2, pCp],
            [r'$U$', r'$\alpha_1\times10^3$',
             r'$\alpha_2\times10^3$', r'$C_p$'],
            fontsize=12, loc='upper left')

        # 3) heater signals
        ax_heat.plot(tm[:i], Q1s[:i],
                     lw=2, color=COL[8], label=r'$Q_1$')
        ax_heat.plot(tm[:i], Q2s[:i],
                     lw=2, ls='--', color=COL[9],
                     label=r'$Q_2$')
        ax_heat.set_ylabel('Heater (%)', fontsize=14)
        ax_heat.set_xlabel('Time (s)', fontsize=14)
        ax_heat.legend(fontsize=12, loc='upper left')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)

        if make_mp4:
            fn = f'./figures/plot_{i:04d}.png'
            fig.savefig(fn, dpi=150, bbox_inches='tight')

    # shut off & final save
    a.Q1(0); a.Q2(0)
    plt.tight_layout()
    fig.savefig('tclab_mhe.png', dpi=300, bbox_inches='tight')

    # build mp4
    if make_mp4:
        images=[]; iset=0
        for i in range(1,n):
            fn = f'./figures/plot_{i:04d}.png'
            images.append(imageio.imread(fn))
            if (i+1)%350==0:
                imageio.mimsave(f'results_{iset}.mp4', images)
                iset+=1; images=[]
        if images:
            imageio.mimsave(f'results_{iset}.mp4', images)

except KeyboardInterrupt:
    a.Q1(0); a.Q2(0)
    print('Shutting down')
    a.close()
    plt.tight_layout()
    fig.savefig('tclab_mhe.png', dpi=300, bbox_inches='tight')

except:
    a.Q1(0); a.Q2(0)
    print('Error: Shutting down')
    a.close()
    plt.tight_layout()
    fig.savefig('tclab_mhe.png', dpi=300, bbox_inches='tight')
    raise

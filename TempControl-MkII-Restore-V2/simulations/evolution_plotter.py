import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# load the matrices
with open('./simulations/evolved_A.txt', 'br') as file:
    matrices = np.loadtxt(file)
    file.close()

matrices = np.array(np.split(matrices, int(matrices.shape[0]/matrices.shape[1])))

fig = plt.figure()
im = plt.imshow((matrices[0]), animated=True, cmap='gray')

time = 0
def animation_func(*args):
    global time
    time += 1
    if(time >= matrices.shape[0]):
        time -= 1
    im.set_array(matrices[time])
    return im,

anim = animation.FuncAnimation(fig, animation_func, interval=50, blit=True)
plt.show()

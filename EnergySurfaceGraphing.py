from cv2 import repeat
from mpl_toolkits.mplot3d.axes3d import Axes3D 
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from dataclasses import dataclass

SURFACE_DIMENSION = (1,1)

@dataclass
class Minima:
    center_x: np.float64
    center_y: np.float64
    depth: np.float64
    sharpness: np.float64

    def get_minima(self, X, Y):
        minima = np.exp(-self.sharpness * ((X-self.center_x)**2 + (Y-self.center_y)**2))
        minima = 2*self.depth * (1/(1+minima) -1)
        return minima

    def get_height(self, x, y):
        h = np.exp(-self.sharpness * ((x-self.center_x)**2 + (y-self.center_y)**2))
        h = 2*self.depth * (1/(1+h) - 1)
        return h

    def __str__(self):
        return f"Minima(center_x={self.center_x}, center_y={self.center_y}, depth={self.depth}, sharpness={self.sharpness})"

def create_random_minima(center_x: np.float64=None, center_y: np.float64=None, depth: np.float64=None, sharpness: np.float64=None):
    if center_x is None:
        center_x = np.round(np.random.uniform(-SURFACE_DIMENSION[0], SURFACE_DIMENSION[0]), 3)
    if center_y is None:
        center_y = np.round(np.random.uniform(-SURFACE_DIMENSION[1], SURFACE_DIMENSION[1]), 3)

    if depth is None:
        depth = np.round(np.random.uniform(0.1, 1), 3)
    if sharpness is None:
        sharpness = np.round(np.random.uniform(5, 20), 3)

    return Minima(center_x, center_y, depth, sharpness)

minimas = [
    Minima(center_x=0.75, center_y=0.5, depth=1, sharpness=7.5),
    Minima(center_x=-0.8, center_y=-0.75, depth=1, sharpness=7.5),

    Minima(center_x=-0.447, center_y=0.715, depth=0.6, sharpness=10.0),
    Minima(center_x=-0.15, center_y=-0.6, depth=0.5, sharpness=75.0),
    Minima(center_x=0.8, center_y=-0.8, depth=0.3, sharpness=12.0),
]

def get_height(x,y):
    h = 0
    for minima in minimas:
        h+=minima.get_height(x,y)
    return h

for minima in minimas:
    print(minima, end=",\n")

X = np.arange(-1.25*SURFACE_DIMENSION[0], 1.25*SURFACE_DIMENSION[0], 0.02)
Y = np.arange(-1.25*SURFACE_DIMENSION[1], 1.25*SURFACE_DIMENSION[1], 0.02)
X, Y = np.meshgrid(X, Y)
Z = 0*X
for minima in minimas:
    Z += minima.get_minima(X,Y)
Z_LIMIT = np.min(Z)


fig = plt.figure(1)
ax:Axes3D=fig.add_subplot(111, projection="3d", computed_zorder=False)
ax.set_title("Energy Surface")
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
ax.set_zlim(Z_LIMIT,0)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma,
                       linewidth=0, antialiased=False, vmin=Z_LIMIT, vmax=-0.1*Z_LIMIT)
ax.view_init(20,-65)


X_START = -1
Y_START = 0
Z_START = get_height(X_START, Y_START)
point, = ax.plot3D(X_START,Y_START,Z_START,'ko')
plt.show()

# START_FRAMES = 20
# FRAMES = 100
# END_FRAMES = 20
# def init():
#     # point, = ax.plot3D(0,0,0,'ko')
#     ax.xaxis.set_ticklabels([])
#     ax.yaxis.set_ticklabels([])
#     ax.zaxis.set_ticklabels([])
#     surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma,
#                     linewidth=0, antialiased=False, vmin=Z_LIMIT, vmax=-0.1*Z_LIMIT)

# def animate(t, plot):
#     print(f"FRAME: {t}")
#     time_scale = min(((t-START_FRAMES)/FRAMES), 1)
#     if t<START_FRAMES:
#         time_scale=0
#     target_minima = minimas[1]
#     curr_x = X_START + time_scale * (target_minima.center_x-X_START)
#     curr_y = Y_START + time_scale * (target_minima.center_y-Y_START)
#     curr_z = get_height(curr_x, curr_y)

#     plot[0].remove()
#     point, = ax.plot3D(curr_x,curr_y,curr_z,'ko')
#     plot[0] = point
#     return plot

# plot = [point]
# anim = FuncAnimation(fig, animate, init_func=init, frames=START_FRAMES+FRAMES+END_FRAMES, fargs=(plot,), repeat_delay=2000)
# anim.save('graphs/ReportGraphs/EWCGraphs/report.gif', fps=20)

# Custom cloud generation script
#
# Controls:
# - Click and drag to add clouds with a position and velocity
# - Scroll while dragging to alter the cloud radius
# - Scroll while not dragging to move the clouds along their velocity
# - Exit the figure to print all current cloud data to stdout
# - Editing cloud intensity must be done manually afterwards, default is base_int
#
# Part of ACCE at the VU, Period 5 2025-2026.


from matplotlib import pyplot as plt
from math import sin, cos, sqrt, atan2
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def get_height(scenario, row: int, col: int, rows: int, columns: int ):
    # Generates a heightmap from the assignment
    x_min , x_max, y_min, y_max = [None] * 4
    if ( scenario == 'M' ): # Mountains scenario
        x_min = -3.3;
        x_max = 5.1;
        y_min = -0.5;
        y_max = 8.8;
    else: # Valley scenarios
        x_min = -5.5;
        x_max = -3;
        y_min = -0.1;
        y_max = 4.2;

    # Compute scenario coordinates of the cell position
    x = x_min + ( (x_max - x_min) / columns ) * col
    y = y_min + ( (y_max - y_min) / rows ) * row

    # Compute function height
    height = -1 / ( x*x+1 ) + 2 / ( y*y+1 ) + 0.5 * sin( 5 * sqrt( x*x+y*y ) ) / sqrt( x*x+y*y) + (x+y) / 3 + sin( x )* cos( y ) + 0.4 * sin( 3*x+y )+ 0.25 * cos( 4*y+x );

    # Substitute by the dam height in the proper scenarios
    LOW_DAM_HEIGHT = -1.0
    HIGH_DAM_HEIGHT = -0.4
    if ( scenario == 'D' and -4.96 >= x >= -5.0 ):
        if ( height < HIGH_DAM_HEIGHT ):
            height = HIGH_DAM_HEIGHT

    elif ( scenario == 'd' and -5.3 >= x >= -5.34 ):
        if ( height < LOW_DAM_HEIGHT ):
            height = LOW_DAM_HEIGHT

    # Transform to meters
    if ( scenario == 'M' ) :
        return height * 30 + 400
    else:
        return height * 20 + 100

def generate_grid(size = (90, 90), kind = 'D'):
    """Generate a full numpy array containing a heightmap
    matching the assignment code."""
    (rows, cols) = size
    array = np.empty(size,dtype=float)
    for y in range(rows):
        for x in range(cols):
            array[y,x] = get_height(kind, y, x, rows, cols)
    return array

clouds = []
working = False
# cloud start x (km)        Initial x-coordinate of the cloud centre.
# cloud start y (km)        Initial y-coordinate of the cloud centre.
# cloud radius (km)         Radius of the cloud.
# cloud intensity (mm/h)    Rainfall intensity.
# cloud speed (km/h)        Cloud speed.
# cloud angle (degrees)     Direction of cloud movement, in degrees.

map_x, map_y = 30, 30
speed_factor = 1/20
base_rad = 4
base_int = 60


def onclick(event):
    global working, base_rad, base_int, clouds
    assert not working
    if event.xdata and event.ydata:
        clouds.append([event.xdata, event.ydata, base_rad, base_int, 0, 0])
        working = True

def onrelease(event):
    global working, clouds
    if working:
        working = False
        if event.xdata and event.ydata:
            [sx, sy] = clouds[-1][0:2]
            xv, yv = event.xdata - sx, event.ydata - sy
            clouds[-1][4] = sqrt(xv**2 + yv**2)
            clouds[-1][5] = 180 * atan2(yv, xv) / np.pi
            draw()
        else:
            clouds.pop()

def onscroll(event):
    global working, clouds
    if working:
        clouds[-1][2] = max(clouds[-1][2] + event.step, 1)
    elif event.xdata and event.ydata:
        for c in clouds:
            dir = c[5] * np.pi / 180
            c[0] = c[0] + event.step * speed_factor * c[4] * cos(dir)
            c[1] = c[1] + event.step * speed_factor * c[4] * sin(dir)
        draw()



def draw():
    """Add lines and patches to the graph according to the saved clouds and redraw."""
    global clouds
    if len(clouds) > len(ax.patches):
        [x, y] = clouds[-1][0:2]
        c = Circle((x,y), clouds[-1][2], color='b', clip_on=False)
        [mag, dir] = clouds[-1][4:6]
        dir *= np.pi / 180
        l = Line2D([x, x + mag * cos(dir)], [y, y + mag * sin(dir)], lw=2, color='black', axes=ax, clip_on=False)
        ax.add_patch(c)
        ax.add_line(l)
    else:
        for i, line in enumerate(ax.lines):
            [x, y] = clouds[i][0:2]
            ax.patches[i].set_center((x,y))            
            [mag, dir] = clouds[i][4:6]
            dir *= np.pi / 180
            line.set_xdata([x, x + mag * cos(dir)])
            line.set_ydata([y, y + mag * sin(dir)])
        

    plt.show()
    
def print_clouds(clouds):
    """Print the clouds according to the required input format."""
    (ax, ay) = array.shape
    for c in clouds:
        [cx, cy, cr, ci, cs, ca] = c
        print(map_x * cx / ax, map_y * cy / ay, cr, ci, cs * speed_factor, ca, end=' ')
    print()


# Start execution
array = generate_grid()

fig, ax = plt.subplots()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('button_release_event', onrelease)
cid = fig.canvas.mpl_connect('scroll_event', onscroll)

# with plt.ion():
aximg = ax.imshow(array, cmap="gray", origin='lower')

plt.show()
print_clouds(clouds)

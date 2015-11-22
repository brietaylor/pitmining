#!/usr/bin/env python3

import pulp
import sys
import math
import numpy as np
import pandas
from tqdm import tqdm, trange
from matplotlib import pyplot

#fixed cost per block is $1.66/t, here we will assume its 1.66 perblock
fixed = 1.66
#process cost is prices at 5.6/t, we will assume its 5.6 perblock
process = 5.6
#copper price as of 31/05/2015 is 6294.78/t, again we will assume its per block
price = 6294.78

def block_profit(cu, btopo=1):
    # grade is by percent
    return (price * cu / 100 - process - fixed) * btopo

def block_profit_simple(cu):
    return cu - .1

def load_data(filename, scale=1):
    print("Loading data...", end='')
    sys.stdout.flush()
    df = pandas.read_csv(filename).drop_duplicates()
    print("done")

    print("Computing block values...", end='')
    sys.stdout.flush()
    x_spacing = 20 * scale
    y_spacing = 20 * scale
    z_spacing = 15 * scale

    # Translate the mine so it starts at (0,0,0)
    df.xcen -= df.xcen.min()
    df.ycen -= df.ycen.min()
    df.zcen -= df.zcen.min()

    # Adjust the size of the mine so that elements are separated by 1.0.
    df.xcen /= x_spacing
    df.ycen /= y_spacing
    df.zcen /= z_spacing

    df['profit'] = block_profit(df.cu)
    print("done")

    image = df_to_image(df)
    return image

def dropwhile(pred, it, last=None):
    """Sorry itertools, but you are way too slow. :(

    Differences from itertools.dropwhile():
     * perform in-place, for performance
     * immediately return first value failing predicate
     * store previous value as function parameter."""

    try:
        if last is None:
            last = next(it)

        while pred(last):
            last = next(it)
    except StopIteration:
        pass
    return last

def df_to_image(df):
    xlim = int(np.floor(df.xcen.max()) + 1)
    ylim = int(np.floor(df.ycen.max()) + 1)
    zlim = int(np.floor(df.zcen.max()) + 1)

    i = 0
    niter = zlim * ylim * xlim
    arr = np.zeros((zlim, ylim, xlim))
    arr[:,:,:] = block_profit(0)

    df_iter = df.sort(["zcen", "ycen", "xcen"]).itertuples()

    # df.columns doesn't include the index column, added by itertuples().
    xcen = df.columns.get_loc("xcen") + 1
    ycen = df.columns.get_loc("ycen") + 1
    zcen = df.columns.get_loc("zcen") + 1
    profit = df.columns.get_loc("profit") + 1

    row = None
    for z in tqdm(range(zlim), "Convert df to image", leave=True):
        for y in range(ylim):
            for x in range(xlim):
                def same_pixel(row):
                    """Use tuple ordering to determine whether we're ahead
                    or behind the dataframe."""
                    image_index = (z,y,x)
                    df_index = (row[zcen], row[ycen], row[xcen])
                    return df_index < image_index

                row = dropwhile(same_pixel, df_iter, row)

                if row[xcen] == x and row[ycen] == y and row[zcen] == z:
                    arr[z,y,x] = row[profit]
                else:
                    # TODO: Should check this doesn't come up too often.
                    arr[z,y,x] = block_profit(0)
    return arr

def do_pitmine(price):
    zlim,ylim,xlim = price.shape
    print("Optimizing in {} variables".format(zlim*ylim*xlim))

    prob = pulp.LpProblem("pitmine", pulp.LpMaximize)

    # Create a dict of variables "d_zyx"
    ds = {}
    for z in tqdm(range(zlim), "Formulating LP variables", leave=True):
        for y in range(ylim):
            for x in range(xlim):
                if (z <= y < ylim - z) and (z <= x < xlim - z):
                    # Give our variables names like d003002001 for 3-deep,
                    # 2-across.  These names aren't actually used anywhere,
                    # but PuLP makes them mandatory and they have to be
                    # unique.
                    name = "d{:03}{:03}{:03}".format(z, y, x)
                    ds[z,y,x] = pulp.LpVariable(name, cat=pulp.LpBinary)
                else:
                    # These cells are guaranteed to be zero, no need to
                    # include them in the model.
                    ds[z,y,x] = 0

                if z == 0:
                    continue

                def addconstraint(dy, dx):
                    nonlocal prob
                    if (0 <= y + dy < ylim and 0 <= x + dx < xlim):
                        prob += ds[z,y,x] <= ds[z-1,y+dy,x+dx]

                # Add nine constraints, one per block above this one
                addconstraint(-1, -1)
                addconstraint(-1, 0)
                addconstraint(-1, 1)
                addconstraint(0, -1)
                addconstraint(0, 0)
                addconstraint(0, 1)
                addconstraint(1, -1)
                addconstraint(1, 0)
                addconstraint(1, 1)

    # Maximizing profit
    obj = 0
    for z,y,x in tqdm(ds, "Formulating Objective Function", leave=True):
        obj += ds[z,y,x] * price[z,y,x]

    prob += obj

    print("Solving LP...", end='')
    sys.stdout.flush()
    outcome = prob.solve()
    print(pulp.LpStatus[outcome])

    # Plot an image
    image = np.zeros((zlim,ylim,xlim))
    for z, y, x in ds:
        image[z,y,x] = pulp.value(ds[z,y,x])

    # Print out how deep we went (a birds-eye view of the mine depth)
    print(np.sum(image, 0))

    # Do a 3D plot of the mine
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, xlim)
    y = np.arange(0, ylim)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, -np.sum(image, 0), cmap='terrain', rstride=1,
            cstride=1, linewidth=0)
    ax.set_zlim(-xlim/1.4, 0) # hack to make it look square
    pyplot.show()
    pyplot.imsave('output.png', np.sum(image, 0), cmap='terrain')

lamb = 1.03

def main():
    try:
        filename = sys.argv[1]
        if len(sys.argv) == 3:
            scale = int(sys.argv[2])
        else:
            scale = 20
    except IndexError:
        print("usage: {} <filename.csv> [scale]".format(sys.argv[0]))
        exit(1)

    #import scipy.stats
    #price = scipy.stats.expon.rvs(loc=-1, scale=lamb, size=(zlim, ylim, xlim))
    price = load_data(filename, scale=scale)
    do_pitmine(price)

if __name__ == "__main__":
    main()


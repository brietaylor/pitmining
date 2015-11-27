#!/usr/bin/env python3

import sys
import numpy as np
from tqdm import tqdm, trange
from matplotlib import pyplot
np.set_printoptions(threshold=np.nan)
pyplot.rc('figure', figsize=(11,8))

#fixed cost per block is $1.66/t, here we will assume its 1.66 perblock
fixed = 15900 * 1.66
#process cost is prices at 5.6/t, we will assume its 5.6 perblock
process = 15900 * 7.31
#copper price as of 31/05/2015 is 6294.78/t, again we will assume its per block
cu_price = 15900 * 4698 * .88

def block_profit(cu, btopo=1, price=cu_price):
    """Compute block profit for given copper percentage"""
    return btopo * ((price * cu / 100 - process) - fixed)

def block_profit_choice(cu, btopo, *args, **kwargs):
    """Same as block profit, but now we can throw away blocks that aren't
    worth processing."""
    process = block_profit(cu, *args, **kwargs)
    noprocess = -fixed * btopo
    return np.maximum(process, noprocess)

def block_profit_aironly(cu, btopo=None):
    """Nothing is profitable."""
    return -1

def block_profit_everything(cu, btopo=None):
    """Everything is profitable"""
    return +1

def normalize_data(df):
    global zmin, zmax
    zmin = df.zcen.min()
    zmax = df.zcen.max()

    # Adjust the size of the mine so that elements are separated by 1.0.
    df.xcen /= 20
    df.ycen /= 20
    df.zcen /= -15

    # Translate the mine so it starts at (0,0,0)
    df.xcen -= df.xcen.min()
    df.ycen -= df.ycen.min()
    df.zcen -= df.zcen.min()

def load_data(filename):
    print("Loading data...", end='')
    sys.stdout.flush()
    import pandas
    df = pandas.read_csv(filename).drop_duplicates()
    print("done")

    print("Computing block values...", end='')
    sys.stdout.flush()

    normalize_data(df)
    print("done")

    return df

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

def topography(df):
    xlim = int(np.floor(df.xcen.max()) + 1)
    ylim = int(np.floor(df.ycen.max()) + 1)

    # Take z in reverse order, so that when we iterate through them, the
    # lowest z value (the top of the topography) will come up first.
    print("Computing topography", end='\r')
    sys.stdout.flush()
    df_iter = df.sort(['ycen', 'xcen', 'zcen']).itertuples()

    xcen = df.columns.get_loc("xcen") + 1
    ycen = df.columns.get_loc("ycen") + 1
    zcen = df.columns.get_loc("zcen") + 1

    row = None
    topo = np.zeros((ylim, xlim), dtype=np.int32)
    for y in trange(ylim, desc="Computing topography", leave=True):
        for x in range(xlim):
            def same_pixel(row):
                """Use tuple ordering to determine whether we're ahead
                or behind the dataframe."""
                image_index = (y,x)
                df_index = (row[ycen], row[xcen])
                return df_index < image_index

            # Drop until x,y coordinates match.  Since this is sorted by
            # x,y,z, the next pixel will be the lowest z.
            row = dropwhile(same_pixel, df_iter, row)
            if row[xcen] == x and row[ycen] == y:
                topo[y,x] = row[zcen]
            else:
                raise RuntimeError("No data at coordinate x={},y={}"
                        .format(x,y))

    return topo

def mine_profit(df, profit_model, **model_args):
    xlim = int(np.floor(df.xcen.max()) + 1)
    ylim = int(np.floor(df.ycen.max()) + 1)
    zlim = int(np.floor(df.zcen.max()) + 1)

    i = 0
    niter = zlim * ylim * xlim
    value = np.zeros((zlim, ylim, xlim))

    print("Converting df to profit", end='\r')
    sys.stdout.flush()
    df_iter = df.sort(["zcen", "ycen", "xcen"]).itertuples()

    # df.columns doesn't include the index column, added by itertuples().
    xcen = df.columns.get_loc("xcen") + 1
    ycen = df.columns.get_loc("ycen") + 1
    zcen = df.columns.get_loc("zcen") + 1
    cu = df.columns.get_loc("cu") + 1
    btopo = df.columns.get_loc("btopo") + 1

    row = None
    for z in trange(zlim, desc="Converting df to profit", leave=True):
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
                    value[z,y,x] = profit_model(row[cu], row[btopo], **model_args)
                else:
                    # No data.  Treat it like empty dirt.
                    value[z,y,x] = profit_model(0, 1, **model_args)
    if False:
        pyplot.hist(value.flatten())
        pyplot.title("dirt value")
        pyplot.show()

    return value

def optimal_pitmine(price, topo):
    import pulp
    zlim,ylim,xlim = price.shape

    assert price.shape[1:] == topo.shape

    prob = pulp.LpProblem("pitmine", pulp.LpMaximize)

    def at_edge(y,x):
        if x == 0 or x == xlim - 1:
            return True
        if y == 0 or y == ylim - 1:
            return True
        return False

    # Create a dict of variables "d_zyx"
    ds = {}
    num_variables = 0
    for z in tqdm(range(zlim), "Formulating LP variables", leave=True):
        for y in range(ylim):
            for x in range(xlim):
                if z < topo[y,x]:
                    # We're above the topography.  This block is
                    # unconditionally mined.
                    price[z,y,x] = 0
                    ds[z,y,x] = 1

                elif z == topo[y,x] and at_edge(y,x):
                    # We're at the topography line and at the edge of the map.
                    # We can't dig here.
                    price[z,y,x] = 0
                    ds[z,y,x] = 0

                else:
                    # This is potential pitmining space.
                    #
                    # Give our variables names like d003002001 for 3-deep,
                    # 2-across.  These names aren't actually used anywhere,
                    # but PuLP makes them mandatory and they have to be
                    # unique.
                    name = "d{:03}{:03}{:03}".format(z, y, x)
                    ds[z,y,x] = pulp.LpVariable(name, cat=pulp.LpBinary)
                    num_variables += 1

                def addconstraint(dy, dx):
                    nonlocal prob
                    if (0 <= y + dy < ylim and 0 <= x + dx < xlim):
                        constraint = ds[z,y,x] <= ds[z-1,y+dy,x+dx]
                        if constraint is False:
                            raise RuntimeError(
                                    "Problem Infeasible by definition: "
                                    "ds[{},{},{}]: {} <= ds[{},{},{}]: {}"
                                    .format(z,y,x,ds[z,y,x],
                                            z-1,y+dy,x+dx,ds[z-1,y+dy,x+dx]))
                        prob += constraint

                if z != 0:
                    # No constraints on the top layer of blocks, otherwise,
                    # add nine constraints, one per block above this one
                    addconstraint(0, 0)
                    addconstraint(-1, -1)
                    addconstraint(-1, 0)
                    addconstraint(-1, 1)
                    addconstraint(0, -1)
                    addconstraint(0, 1)
                    addconstraint(1, -1)
                    addconstraint(1, 0)
                    addconstraint(1, 1)

    print("Optimizing in {} variables".format(num_variables))

    # Maximizing profit
    obj = 0
    for z,y,x in tqdm(ds, "Formulating Objective Function", leave=True):
        obj += ds[z,y,x] * price[z,y,x]

    prob += obj

    print("Solving LP...", end='')
    sys.stdout.flush()
    outcome = prob.solve()
    print(pulp.LpStatus[outcome])

    # Convert solution to a 3d image
    image = np.zeros((zlim,ylim,xlim), dtype=np.uint)
    for z, y, x in ds:
        image[z,y,x] = pulp.value(ds[z,y,x])

    # Get a 2D projection of the image
    projection = np.sum(image, 0, dtype=np.uint)

    obj_value = np.sum(image * price)
    print("Objective: {}".format(obj_value))

    return projection, obj_value

def downsample(image, scale):
    """Downsample an image down to a smaller image by a factor of `scale`, by
    averaging the bins."""
    result_shape = np.uint(np.array(image.shape) // scale)
    result = np.zeros(result_shape, dtype=image.dtype)
    num_avg = scale**2
    if len(result_shape) == 2:
        # 2d downsample
        # XXX: I know this is topography, so scale down the z-axis, too.
        ylim, xlim = result_shape
        for y in range(ylim):
            for x in range(xlim):
                xmin, xmax = x * scale, (x+1) * scale
                ymin, ymax = y * scale, (y+1) * scale
                result[y,x] = np.mean(image[ymin:ymax, xmin:xmax] / scale)
    elif len(result_shape) == 3:
        # 3d downsample
        # XXX: I know this is price data, so sum it instead of averaging.
        zlim, ylim, xlim = result_shape
        for z in range(zlim):
            for y in range(ylim):
                for x in range(xlim):
                    zmin, zmax = z * scale, (z+1) * scale
                    xmin, xmax = x * scale, (x+1) * scale
                    ymin, ymax = y * scale, (y+1) * scale
                    result[z,y,x] = np.sum(image[zmin:zmax, ymin:ymax, xmin:xmax])
    return result

def plot_pitmine_2d(pitmine, scale, name='output'):
    """Do a 2D plot of the mine, and save the data for later use"""
    # Print out how deep we went (a birds-eye view of the mine depth)
    x_scale = 20 * scale
    y_scale = 20 * scale
    z_scale = 15 * scale
    ylim, xlim = pitmine.shape

    print(pitmine)
    np.save("{}.npy".format(name), pitmine)
    pyplot.imsave("{}.png".format(name), zmax - z_scale * pitmine,
            cmap='terrain', vmin=zmin, vmax=zmax)

def plot_pitmine_3d(pitmine, scale):
    """Plot a pitmine or topography map in 3D"""
    from mpl_toolkits.mplot3d import Axes3D
    x_scale = 20 * scale
    y_scale = 20 * scale
    z_scale = 15 * scale
    ylim, xlim = pitmine.shape

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, xlim) * x_scale
    y = np.arange(0, ylim) * y_scale
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, zmax - z_scale * pitmine, cmap='terrain',
            rstride=1, cstride=1, linewidth=0)
    ax.set_zlim(zmin, zmax)
    pyplot.show()

def do_resolution_sensitivity(filename):
    """Run at several resolutions to check for convergence."""
    scales = (8, 7, 6, 5, 4, 3, 2)
    obj = []
    df = load_data(filename)
    topo = topography(df)
    value = mine_profit(df, profit_model=block_profit_choice, price=7*cu_price)
    for s in scales:
        topo_scaled = downsample(topo, s)
        value_scaled = downsample(value, s)
        _, objective = optimal_pitmine(value_scaled, topo_scaled)
        obj.append(objective)
    pyplot.plot(scales, obj)
    pyplot.savefig("convergence.png")
    pyplot.clf()

def do_single(filename, scale):
    """Compute a single pitmine and plot it in 3D."""
    df = load_data(filename)
    topo = downsample(topography(df), scale)
    plot_pitmine_2d(topo, scale=scale, name='topo')
    value = downsample(mine_profit(df, profit_model=block_profit_choice,
                price=7*cu_price), scale)
    pitmine, objective = optimal_pitmine(value, topo)
    plot_pitmine_2d(pitmine, scale=scale, name='pitmine')
    plot_pitmine_3d(pitmine, scale=scale)

def do_price_sensitivity(filename, scale):
    """Compute a number of pitmines with various copper prices, and plot the
    resulting values"""
    df = load_data(filename)
    topo = downsample(topography(df), scale)
    cu_price_factors = np.logspace(0,1,20)
    obj = []
    for cu_price_factor in cu_price_factors:
        value = downsample(mine_profit(df, profit_model=block_profit_choice,
                price=cu_price_factor*cu_price), scale)
        pitmine, objective = optimal_pitmine(value, topo)
        obj.append(objective)
        plot_pitmine_2d(pitmine, scale=scale,
                name=str(int(cu_price_factor)))

    pyplot.plot(cu_price_factors, obj)
    pyplot.savefig("price_sense.png")
    pyplot.clf()

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

    do_single(filename, scale)
    #do_resolution_sensitivity(filename)
    #do_price_sensitivity(filename, scale)

if __name__ == "__main__":
    main()


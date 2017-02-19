import h5py
import glob

light = 'Abrupt BNC565 CantClk.tp light [s]'
tip = 'Abrupt BNC565 CantClk.tp tip [s]'
files = glob.glob("../data/pk-efm/*.h5")
for fname in files:
    with h5py.File(fname, 'r') as fh:
        attrs = fh['data/0000'].attrs
        print("\n")
        print(fname[22:])
        print("===========================")
        print("tp light = {} s".format(attrs[light]))
        print("tp tip = {} s".format(attrs[tip]))


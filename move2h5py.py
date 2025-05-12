
import numpy as np
import h5py
import time
import glob
import os

nrand = 400
nz = 128
n = 256
box_length = 200

frequencies = np.linspace(137.46, 144.60, 128)  # MHz
redshifts = 1420. / frequencies - 1

files = ['Lightcone_FID_400_Samples.npz']  # np.sort(glob.glob('Lightcone*npz'))
print(f'{len(files)} files to convert.')

for file in files:
    print(f'\n{file}')
    newfile = '../'+file[:-3]+'h5'
    if not os.path.exists(newfile):
        print('Loading npz data...')
        t0 = time.time()
        data = np.load(file, allow_pickle=True)['arr_0']
        t1 = time.time()
        print(f'Took {(t1-t0)/60.:.1f} minutes.')
        print(data.shape)

        t0 = time.time()
        print('Saving to h5py...')
        with h5py.File(newfile, "w") as f:
            f.create_dataset('brightness_lightcone', data=data, compression="gzip", chunks=(1, nz, n, n))
            f.create_dataset('frequencies', data=frequencies)
            f.create_dataset('redshifts', data=redshifts)
            f.create_dataset('box_length', shape=(1,), data=box_length, dtype=float)
            f.create_dataset('ngrid', shape=(1,), data=n, dtype=int)
            f.create_dataset('nrealisations', shape=(1,), data=nrand, dtype=int)
        t1 = time.time()
        print(f'Took {(t1-t0)/60.:.1f} minutes.')

    print('Reading one slice from h5py...')
    t0 = time.time()
    with h5py.File(newfile, 'r') as f:
        subarray = f['brightness_lightcone'][10, 12, :, :]  # Reads only (nz, n, n), not entire data
    t1 = time.time()
    print(f'Took {(t1-t0):.1f} seconds.')
    print('Done.')

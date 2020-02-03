import numpy as np
import os
import glob

folders = glob.glob('./simulations/*')
fileexec = '000_simulation_matrix_map.py'

for ff in folders:

    try:
        with open(ff + '/' +fileexec, 'r') as fid:
            lines = fid.readlines()

        i_init_all = np.where([('sim_content.init_all()' in ll) for ll in lines])[0][0]

        with open(ff+ '/temp.py', 'w') as fid:
            fid.writelines(lines[:i_init_all])


        os.system(f"""
            cd {ff}
            (PYTHONPATH=$PYTHONPATH:../../../; python temp.py)
            rm temp.py
            cd ..
            """)
    except Exception:
        pass

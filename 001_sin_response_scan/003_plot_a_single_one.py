import PyECLOUD.myfilemanager as mfm
import PyECLOUD.mystyle as ms

import numpy as np

folder_sims = 'simulations'


sin_ampl = 0
cos_ampl = 1e-4
n_osc_list = range(200)

cos_ampl = 0
sin_ampl = 1e-4
n_osc_list = range(1,200)

for n_osc in n_osc_list:

    current_sim_ident= f'n_{n_osc:.1f}_c{cos_ampl:.2e}_s{sin_ampl:.2e}'
    ob = mfm.myloadmat_to_obj(folder_sims + '/' + current_sim_ident + '/response.mat')

    import matplotlib.pyplot as plt
    plt.close('all')
    ms.mystyle(fontsz=14, traditional_look=False)
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(3,1,1)
    ax2 = fig1.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig1.add_subplot(3,1,3, sharex=ax1)

    #ax1.plot(ob.z_slices, ob.int_slices)
    ax2.plot(ob.z_slices, ob.x_slices)
    ax3.plot(ob.z_slices, ob.dpx_slices_all_clouds)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True)

    fig2 = plt.figure(2)
    ax21 = fig2.add_subplot(2,1,1)
    ax22 = fig2.add_subplot(2,1,2, sharex=ax21)

    ax21.pcolormesh(ob.z_slices, 1e3*ob.xg, ob.rho_cut.T)
    ax21.plot(ob.z_slices, 1e3*ob.x_slices, 'k', lw=2)
    ax22.plot(ob.z_slices, 1e6*ob.dpx_slices_all_clouds)
    ax22.set_ylim(1e6*np.nanmax(np.abs(ob.dpx_slices_all_clouds))*np.array([-1.1, 1.1]))
    ax21.set_ylim(-2.5, 2.5)

    ax21.set_ylabel('x [mm]')
    ax21.set_ylim(-2.5, 2.5)
    ax22.set_ylabel('Dpx [urad]')
    ax22.set_xlabel('z [m]')
    ax22.grid(linestyle=':', alpha=.9)

    fig2.suptitle(f'n={n_osc} ' + {True: 'sine', False: 'cosine'}[sin_ampl>cos_ampl], fontsize=14)
    fig2.subplots_adjust(
        top=0.91,
        bottom=0.115,
        left=0.16,
        right=0.915,
        hspace=0.2,
        wspace=0.2)
    
    fig2.savefig(current_sim_ident + '.png', dpi=200)

plt.show()


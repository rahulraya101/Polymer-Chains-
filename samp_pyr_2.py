import numpy as np
import matplotlib.pyplot as plt
import espressomd
from espressomd import interactions
from espressomd import polymer
from espressomd.io.writer import vtf
import espressomd.observables
import math
required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)
import sys
from espressomd import thermostat

def write_xyz(system,filename,mode, max_part, pid):
        part=system.part
        cnt_H=0
        f=open(filename, mode)
        f.write("%d\n#box"%(max_part)+str(system.box_l)+"\n");
        type_id=[]
        pos_id=[]
        for p in part:
                pos=p.pos
                type_id.append(p.type)
                pos_id.append(pos)
        for i in range(0, max_part):
                if i < len(type_id):
                        f.write("H{0:.0f} {1:.6g} {2:.6g} {3:.6g}\n".format(type_id[i], pos_id[i][0],          pos_id[i][1], pos_id[i][2]))
                else:
                        f.write("H{} 0.0000 0.0000 0.0000\n".format(pid))
# system parameters

nchain = 1
chain_len = 25    # [mass] = [time]^2/[length]^2

volume = chain_len
box_l = volume **(1./3.)   
temperature = 300 #kelvin
#energy = kT
#time = 1ps

system = espressomd.System(box_l = [box_l]*3)
np.random.seed(seed=42)

system.time_step = 0.01
system.cell_system.skin = 0.4
system.cell_system.set_n_square(use_verlet_lists=False)

lj_sig = 0.308 #distance b/n sigmabonds in nm
lj_eps = 1/temperature
lj_cut = 2.5 * lj_sig
lj_cap = 20

unit_length = 1e-9 #now it is from nm to m
kT = 1.38e-23 * temperature

#creating polymer chain
system.non_bonded_inter[0, 0].wca.set_params(epsilon=lj_eps, sigma= lj_sig)

fene = interactions.FeneBond(k=10, d_r_max=2, r_0 = lj_sig)
system.bonded_inter.add(fene)

polymers = polymer.linear_polymer_positions(n_polymers= nchain,
                                            beads_per_chain=chain_len,
                                            bond_length=0.9, 
                                            seed=23)
for i in range(nchain):
    for j in range(chain_len):
        id = len(system.part)
        pos = polymers[i][j]
        system.part.add(id = id, pos = pos)
        if j > 0:
            system.part[id].add_bond((fene, id - 1))
#        print(system.part[id].bonds)
write_xyz(system,filename = "partpos.xyz", mode ="w", max_part = len(system.part), pid = 0)    #mode =  A (append) , W (write).    
obs_file = open('pmr.obs', 'w')
obs_file.write("# Time\tE_tot\tE_kin\tE_pot\tE_fene\n")

# wormup step

# warmup integration (steepest descent)
warm_steps = 20
warm_n_times = 10

# convergence criterion (particles are separated by at least 90% sigma)
min_dist = 0.9 * lj_sig


act_min_dist = system.analysis.min_dist()
print("Start with minimal distance {}".format(act_min_dist))


# integration
int_steps = 1000
int_n_times = 5

system.integrator.set_steepest_descent(f_max=0, gamma=1e-3,
                                       max_displacement=lj_sig / 100)




# Warmup Integration Loop
i = 0
while (i < warm_n_times and act_min_dist < min_dist):
    system.integrator.run(steps=warm_steps)
    # Warmup criterion
    act_min_dist = system.analysis.min_dist()
    i += 1
#open set file
set_file = open("pmr.set", "w")
set_file.write("box_l %s\ntime_step %s\nskin %s\n" %
               (box_l, system.time_step, system.cell_system.skin))




#Integration loop

times, e_tots, e_kins, e_pots, e_fene, rg, rh = [], [], [], [], [], [], []

for i in range(int_n_times):
    system.integrator.run(steps=int_steps)
    time = system.time
    energy = system.analysis.energy()
#    rg = espressomd.analyze.Analysis.calc_rg()
#    rh = espressomd.analyze.Analysis.calc_rh()
    e_tot = energy['total']
    e_kin = energy['kinetic']
    e_pot = energy['non_bonded']
    e_fene = energy['bonded']
    rg = system.analysis.calc_rg()
    rh = system.analysis.calc_rh()
    obs_file.write(" %f %f %f %f %f %f %f\n" % (time, e_tot , e_kin , e_pot, e_fene, rg, rh))
    print(e_fene)





#print(polymers[0][0])
# terminate program
print("\nFinished.")

from pythtb import *
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from common.baseclasses import AWA
from NearFieldOptics.Materials.material_types import * #For Logger, overkill
from scipy.sparse import *
import pickle
import pdb
from tbg_module import ServiceClass as sv
from tbg_module import FeatureClass as feat


# dictionary of numpy matrix representation of hopping Hamiltonian
def get_hopping_Hamiltonian_dict(model,num_band=5, spin_degeneracy=2,valley_degeneracy=1,bdg_degeneracy=2):

    Ham_dict = {}
    
    for t,m,n,Rs in model._hoppings:
        Rx=Rs[0]
        Ry=Rs[1]
        try:
            Ham_dict[(Rx,Ry)][m,n] = t
        except:
            total = num_band * spin_degeneracy * valley_degeneracy * bdg_degeneracy
            Ham_dict[(Rx,Ry)] = np.zeros((total,total),dtype='complex64')
            Ham_dict[(Rx,Ry)][m,n] = t

    return Ham_dict


def pickle_dump(model,file_name):

    if type(file_name) != str:
        Logger.raiseException('Wrong type for file_name input. Need to be of type string', exception=TypeError)
    file = open(file_name+'.pickle', 'wb')
    pickle.dump(model, file)
    file.close()

def pickle_load(file_name):

    if type(file_name) != str:
        Logger.raiseException('Wrong type for file_name input. Need to be of type string', exception=TypeError)

    file = open(file_name+'.pickle', 'rb')
    model = pickle.load(file)
    file.close()
    return model


class tbg_model():
    
    def __init__(self,file_name='source/hmat_5band_bot_1p10_rewan_RS.dat',lattice=[[1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]],\
                real_orbit_position=False,spin_is_degen=False,valley_is_degen=False,bdg_is_degen=False,orbital_symmetry=False,\
                kspace_numsteps=100):
        
        # if file_name[:25] == 'source/monolayer_graphene':
        #     num_band = 2
        # else:
        num_band = int(file_name[12])    #based on the file naming convention in Carr's Github repo
        
        # filename is not checked, num_band is checked instead
        # will raiseException if input type/value is not as expected
        checker = sv.InputChecker(num_band=num_band,lattice=lattice,real_orbit_position=real_orbit_position,\
                                spin_is_degen=spin_is_degen,valley_is_degen=valley_is_degen,bdg_is_degen=bdg_is_degen,\
                                orbital_symmetry=orbital_symmetry,kspace_numsteps=kspace_numsteps)
        checker.check_argument_clean()

        self.data = np.loadtxt(file_name)
        self.num_band = num_band
        self.spin_is_degen, self.valley_is_degen, self.bdg_is_degen = spin_is_degen, valley_is_degen, bdg_is_degen
        self.spin_degeneracy, self.valley_degeneracy, self.bdg_degeneracy = self.set_degeneracy(spin_is_degen,valley_is_degen,bdg_is_degen)
        self.lattice = lattice
        orbiter = feat.Orbiter(num_band=num_band,real_orbit_position=real_orbit_position,spin_degeneracy=self.spin_degeneracy,\
                            valley_degeneracy=self.valley_degeneracy,bdg_degeneracy=self.bdg_degeneracy)
        self.orbit = orbiter.construct_orbit()
        self.orbital_symmetry = orbital_symmetry
        self.model = tb_model(2,2,self.lattice,self.orbit) 
        self.temp_model = None
        self.models = None
        
        self.feature_mode_list = ["swave_singlet","swave_triplet_antisymmetric_orbit","swave_triplet_antisymmetric_hopping","chemical_potential"]
        self.evaluator = sv.Evaluator(num_band=self.num_band,spin_degeneracy=self.spin_degeneracy,valley_degeneracy=self.valley_degeneracy,\
                                    bdg_degeneracy=self.bdg_degeneracy,spin_is_degen=self.spin_is_degen,kspace_numsteps=100)
        self.plotter = sv.Plotter(num_band=self.num_band,spin_degeneracy=self.spin_degeneracy,bdg_degeneracy=self.bdg_degeneracy)

        self.add_hopping()
        self.feature_mode = None
        self.parameter = None

    def set_degeneracy(self,spin_is_degen,valley_is_degen,bdg_is_degen):
        
        if self.spin_is_degen: 
            spin_degeneracy = 2
        else:
            spin_degeneracy = 1
        if self.valley_is_degen: 
            valley_degeneracy = 2
        else:
            valley_degeneracy = 1
        if self.bdg_is_degen: 
            bdg_degeneracy = 2
        else:
            bdg_degeneracy = 1
        return spin_degeneracy,valley_degeneracy,bdg_degeneracy

    def get_full_Hamiltonian_index(self,m,spin_index,valley_index,bdg_index):

        #create a hierarchy of indices, from smallest to largest: m, spin, valley, bdg
        degen_index_list = []
        if self.spin_is_degen:
            degen_index_list.append(spin_index)
        if self.valley_is_degen:
            degen_index_list.append(valley_index)
        if self.bdg_is_degen:
            degen_index_list.append(bdg_index)

        full_index = m
        i = 0
        # add contribution from each of the degeneracy index, according to the hierarchy
        for index in degen_index_list:
            add = index * self.num_band * (2**i)
            full_index += add
            i += 1
        return full_index

    # only for the case both spin and BdG degeneracy
    def get_indices(self,full_index):
        bdg_index = int(full_index/(self.num_band*2))
        partial_index = full_index
        if bdg_index==1:
            partial_index = full_index - self.num_band*2
        spin_index = int(partial_index/(self.num_band))
        m = partial_index % self.num_band
        return bdg_index,spin_index,m

    # Specify hopping term for bare Hamiltonian
    def add_hopping(self):
        for bdg_index in range(0,self.bdg_degeneracy,1):
            for valley_index in range(0,self.valley_degeneracy,1):
                for spin_index in range(0,self.spin_degeneracy,1):
                    self.iterate_hopping_terms(spin_index,valley_index,bdg_index)
    
    # only for the case spin and BdG degeneracy 
    def iterate_hopping_terms(self,spin_index,valley_index,bdg_index):
        for i in range(0,len(self.data),1):
            t_real = self.data[i,4]
            t_imag = self.data[i,5]
            # For lower right block of BdG degeneracy, hamiltonian is -H*=-H_real+H_imag
            if bdg_index==1:
                t = (-t_real+1j*t_imag)/2   #divide by 2 as the conjugate pair will be summed to give extra factor of 2
            else:
                t = (t_real+1j*t_imag)/2   #divide by 2 as the conjugate pair will be summed to give extra factor of 2

            local_m = self.data[i,2]-1
            local_n = self.data[i,3]-1

            m = self.get_full_Hamiltonian_index(local_m,spin_index,valley_index,bdg_index)
            n = self.get_full_Hamiltonian_index(local_n,spin_index,valley_index,bdg_index)
            Rx = self.data[i,0]
            Ry = self.data[i,1]

            if m!=n or Rx!=0 or Ry!=0:
                self.model.set_hop(t, m, n, [ Rx, Ry],allow_conjugate_pair=True,mode='reset')
            
            else:    #on site energy
                if bdg_index==1:
                    self.model.set_onsite(-t_real, int(m), mode="reset")    #t_imag=0 for on-site terms
                else:
                    self.model.set_onsite(t_real, int(m), mode="reset")    #t_imag=0 for on-site terms


    ################################################################################################################################
    # Features
    ################################################################################################################################

    # only for the case spin and BdG degeneracy 
    def add_chemical_potential(self,bdg_index,valley_index,spin_index,μ):

        for local_m in range(0,self.num_band,1):
            m = self.get_full_Hamiltonian_index(local_m,spin_index,valley_index,bdg_index)
            if bdg_index == 1:
                self.temp_model.set_onsite(μ, int(m), mode="add")    #t_imag=0 for on-site terms
            else:
                self.temp_model.set_onsite(-μ, int(m), mode="add")    #t_imag=0 for on-site terms

    # only for the case spin and BdG degeneracy
    # only onsite pairing
    def add_swave_singlet(self,parameter):      
        
        if not self.spin_is_degen:
            Logger.raiseException('Need to enable spin degeneracy to allow for s-wave pairing.',exception=ValueError)
        
        if self.orbital_symmetry:
            
            if self.num_band==5:
                # for hopping between orbitals with p+- symmetry
                for index in range(0,2,1):
                    self._set_antisymmetric_spin_symmetric_orbit_(index,int(not index),parameter)
                # for hopping between orbitals with s symmetry
                diagonal_index = 2
                self._set_antisymmetric_spin_symmetric_orbit_(diagonal_index,diagonal_index,parameter)
                # for hopping between orbitals with pz symmetry
                for diagonal_index in range(3,5,1):
                    self._set_antisymmetric_spin_symmetric_orbit_(diagonal_index,diagonal_index,parameter)
                
            else:    #self.num_band=8
                # for hopping between orbitals with p+- symmetry
                for local_m in range(0,2,1):
                    for local_n in range(0,2,1):
                        self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)
                # for hopping between orbitals with pz symmetry
                for local_m in range(3,5,1):
                    for local_n in range(3,5,1):
                        self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)
                # for hopping between orbitals with s symmetry
                ## (τ,s) with itself
                local_n = 2
                local_m = 2
                self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)
                ## (τ,s) with three (κ,s)
                local_n = 2
                for local_m in range(5,8,1):
                    self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)
                    self._set_antisymmetric_spin_symmetric_orbit_(local_n,local_m,parameter)
                ## three (κ,s) within themselves
                for local_m in range(5,8,1):
                    for local_n in range(5,8,1):
                        self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)

        else:    # implement an identity matrix for the orbital block matrix
            for local_m in range(0,self.num_band,1):
                self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_m,parameter)

    
    def _set_antisymmetric_spin_symmetric_orbit_(self,local_m,local_n,parameter):

        # positive sign for the upper right spin block matrix
        n = self.get_full_Hamiltonian_index(local_m,spin_index=1,valley_index=0,bdg_index=1)
        m = self.get_full_Hamiltonian_index(local_n,spin_index=0,valley_index=0,bdg_index=0)
        self.temp_model.set_hop(parameter, m, n, [ 0, 0],mode='reset')
        #negative sign for the lower left spin block matrix
        n = self.get_full_Hamiltonian_index(local_m,spin_index=0,valley_index=0,bdg_index=1)
        m = self.get_full_Hamiltonian_index(local_n,spin_index=1,valley_index=0,bdg_index=0)
        self.temp_model.set_hop(-parameter, m, n, [ 0, 0],mode='reset')


    # only for the case spin and BdG degeneracy
    # only onsite
    def add_swave_triplet_antisymmetric_orbit(self,parameter):
        
        if not self.spin_is_degen:
            Logger.raiseException('Need to enable spin degeneracy to allow for s-wave pairing.',exception=ValueError)
        
        if self.orbital_symmetry:

            if self.num_band==5:
                # for hopping between orbitals with p+- symmetry
                for local_m in range(0,2,1):
                    for local_n in range(0,2,1):
                        self._set_symmetric_spin_antisymmetric_orbit_(local_m,local_n,parameter)
                # zero for onsite of orbitals with pz symmetry
                # zero for onsite of orbital with s symmetry
                

            else:    #self.num_band==8
                
                # for hopping between orbitals with p+- symmetry
                for local_m in range(0,2,1):
                    for local_n in range(0,2,1):
                        self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)
                
                # for hopping between orbitals with pz symmetry
                for local_m in range(3,5,1):
                    for local_n in range(3,5,1):
                        self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)
                
                # for hopping between orbitals with s symmetry
                ## (τ,s) with three (κ,s)
                local_m = 2
                for local_n in range(5,8,1):
                    self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)
                    self._set_antisymmetric_spin_symmetric_orbit_(local_n,local_m,parameter)
                ## three (κ,s) within themselves
                for local_m in range(5,8,1):
                    for local_n in range(5,8,1):
                        self._set_antisymmetric_spin_symmetric_orbit_(local_m,local_n,parameter)

        else:    # implement an identity matrix for the orbital block matrix
            for local_m in range(0,self.num_band,1):
                for local_n in range(0,self.num_band,1):
                    # assume local_m is the vertical index, and local_n is the horizontal index
                    self._set_symmetric_spin_antisymmetric_orbit_(local_m,local_n,parameter)

    def _set_symmetric_spin_antisymmetric_orbit_(self,local_m,local_n,parameter):

        # assume local_m is the vertical index, and local_n is the horizontal index
        
        if local_m > local_n:    # for lower triangular part of the orbital matrix, set hopping = -Δ
            sign = -1
        elif local_m < local_n:    # for upper triangular part of the orbital matrix, set hopping = Δ
            sign = 1
        else:    # for diagonal, set hopping = 0
            sign = 0
        
        n = self.get_full_Hamiltonian_index(local_n,spin_index=1,valley_index=0,bdg_index=1)
        m = self.get_full_Hamiltonian_index(local_m,spin_index=0,valley_index=0,bdg_index=0)
        self.temp_model.set_hop(sign*parameter, m, n, [ 0, 0],mode='reset')

        n = self.get_full_Hamiltonian_index(local_n,spin_index=0,valley_index=0,bdg_index=1)
        m = self.get_full_Hamiltonian_index(local_m,spin_index=1,valley_index=0,bdg_index=0)
        self.temp_model.set_hop(sign*parameter, m, n, [ 0, 0],mode='reset')
        

    # only for the case spin and BdG degeneracy
    # simplet hopping: to nearest neighbor, same orbit
    def add_swave_triplet_antisymmetric_hopping(self,parameter):
        
        if not self.spin_is_degen:
            Logger.raiseException('Need to enable spin degeneracy to allow for s-wave pairing.',exception=ValueError)
        
        for bdg_index in range(0,self.bdg_degeneracy,1):
            for valley_index in range(0,self.valley_degeneracy,1):
                for spin_index in range(0,self.spin_degeneracy,1):
                    for local_m in range(0,self.num_band,1):
                        n = self.get_full_Hamiltonian_index(local_m,spin_index,valley_index=0,bdg_index=1)
                        m = self.get_full_Hamiltonian_index(local_m,int(not spin_index),valley_index=0,bdg_index=0)
                        self.temp_model.set_hop( parameter, m, n, [ 1, 0],allow_conjugate_pair=True,mode='reset')
                        self.temp_model.set_hop( parameter, m, n, [ 0, 1],allow_conjugate_pair=True,mode='reset')
                        self.temp_model.set_hop(-parameter, m, n, [-1, 0],allow_conjugate_pair=True,mode='reset')
                        self.temp_model.set_hop(-parameter, m, n, [ 0,-1],allow_conjugate_pair=True,mode='reset')

    # Remember to update self.feature_mode_list if added new feature_mode!
    def eval(self,parameters=[],feature_mode='base',path_points=['K','Γ','M','K']):

        sv.InputChecker()._check_path_points_(path_points)
        self.evaluator.set_kspace_path(path_points)
        if feature_mode == "base":
            self.evaluator.eval_base(self.model)

        elif feature_mode == "swave_singlet":
            self.models=[]
            if len(parameters) == 1:
                self.temp_model = cp.deepcopy(self.model)
                parameter = parameters[0]
                self.add_swave_singlet(parameter)
                self.models.append(self.temp_model)
                self.evaluator.eval_base(self.temp_model)
            else:
                for parameter in parameters:
                    self.temp_model = cp.deepcopy(self.model)
                    self.add_swave_singlet(parameter)
                    self.models.append(self.temp_model)
                self.evaluator.eval_parameters(self.models,parameters)

        elif feature_mode == "swave_triplet_antisymmetric_orbit":
            self.models=[]
            for parameter in parameters:
                self.temp_model = cp.deepcopy(self.model)
                self.add_swave_triplet_antisymmetric_orbit(parameter)
                self.models.append(self.temp_model)
            self.evaluator.eval_parameters(self.models,parameters)
        
        elif feature_mode == "swave_triplet_antisymmetric_hopping":
            self.models=[]
            for parameter in parameters:
                self.temp_model = cp.deepcopy(self.model)
                self.add_swave_triplet_antisymmetric_hopping(parameter)
                self.models.append(self.temp_model)
            self.evaluator.eval_parameters(self.models,parameters)

        elif feature_mode == "chemical_potential":
            self.models=[]
            for parameter in parameters:
                self.temp_model = cp.deepcopy(self.model)

                for bdg_index in range(0,self.bdg_degeneracy,1):
                    for valley_index in range(0,self.valley_degeneracy,1):
                        for spin_index in range(0,self.spin_degeneracy,1):
                            self.add_chemical_potential(spin_index,valley_index,bdg_index,parameter)
                
                self.models.append(self.temp_model)
            self.evaluator.eval_parameters(self.models,parameters)
        
        else:
            Logger.raiseException('Wrong value for \"feature_mode\". Only accept '+str(self.feature_mode_list),exception=ValueError)
        
        #set class variable
        self.feature_mode = feature_mode
        self.parameters = parameters

    # def eval(self):
        
    #     self.evaluator.eval_base(self.model)
    
    # note: the plot function is separate from evaluate, because sometimes need to plot multiple times for each eval; eval takes more time than plot
    def plot(self):

        self.plotter.update(self.evaluator)

        if self.feature_mode == "base" or len(self.parameters)==1:
            self.plotter.plot_base()
        else:
            self.plotter.plot_parameters(self.parameters)









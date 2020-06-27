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




class tbg_model_featured():

    def __init__(self,file_name='source/hmat_5band_bot_1p10_rewan_RS.dat',lattice=[[1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]],\
                real_orbit_position=False,spin_is_degen=False,valley_is_degen=False,bdg_is_degen=False,orbital_symmetry=False,\
                kspace_numsteps=100,Δs=[0],μs=[0]):
        
        num_band = int(file_name[12])    #based on the file naming convention in Carr's Github repo

        # filename is not checked, num_band is checked instead
        # will raiseException if input type/value is not as expected
        checker = sv.InputChecker(num_band=num_band,lattice=lattice,real_orbit_position=real_orbit_position,\
                                spin_is_degen=spin_is_degen,valley_is_degen=valley_is_degen,bdg_is_degen=bdg_is_degen,\
                                orbital_symmetry=orbital_symmetry,kspace_numsteps=kspace_numsteps)
        checker.check_argument_clean()

        self.data = np.loadtxt(file_name)
        self.num_band = num_band
        self._set_degeneracy(spin_is_degen,valley_is_degen,bdg_is_degen)
        self.lattice = lattice
        self._set_orbitals(real_orbit_position,orbital_symmetry)
        self.feature_mode_list = ["hopping","chemical_potential","intra_unit_cell_singlet","intra_unit_cell_triplet_antisymmetric_orbit",\
                                  "intra_unit_cell_triplet_antisymmetric_hopping"]
        self._set_helpers()
        self.Δs = Δs
        self.μs = μs
        self.feature_mode = None

        self._set_model()

    def _set_degeneracy(self,spin_is_degen,valley_is_degen,bdg_is_degen):
        
        self.spin_is_degen = spin_is_degen
        self.valley_is_degen = valley_is_degen
        self.bdg_is_degen = bdg_is_degen

        if self.spin_is_degen: 
            self.spin_degeneracy = 2
        else:
            self.spin_degeneracy = 1
        if self.valley_is_degen: 
            self.valley_degeneracy = 2
        else:
            self.valley_degeneracy = 1
        if self.bdg_is_degen: 
            self.bdg_degeneracy = 2
        else:
            self.bdg_degeneracy = 1

    def _set_orbitals(self,real_orbit_position,orbital_symmetry):

        orbiter = feat.Orbiter(num_band=self.num_band,real_orbit_position=real_orbit_position,spin_degeneracy=self.spin_degeneracy,\
                            valley_degeneracy=self.valley_degeneracy,bdg_degeneracy=self.bdg_degeneracy)
        self.orbit = orbiter.construct_orbit()
        self.orbital_symmetry = orbital_symmetry

    def _set_helpers(self):

        self.evaluator = sv.Evaluator(num_band=self.num_band,spin_degeneracy=self.spin_degeneracy,valley_degeneracy=self.valley_degeneracy,\
                                    bdg_degeneracy=self.bdg_degeneracy,spin_is_degen=self.spin_is_degen,kspace_numsteps=100)
        self.plotter = sv.Plotter(num_band=self.num_band,spin_degeneracy=self.spin_degeneracy,bdg_degeneracy=self.bdg_degeneracy)
        self.tensorist = feat.Tensorist(data=self.data,num_band=self.num_band,feature_mode_list=self.feature_mode_list,\
                                        spin_is_degen=self.spin_is_degen,valley_is_degen=self.valley_is_degen,bdg_is_degen=self.bdg_is_degen)

    def _set_model(self):

        self.model = tb_model(2,2,self.lattice,self.orbit) 
        self.temp_model = None
        self.models = None
        self.add_hopping()

    # Specify hopping term for bare Hamiltonian; called in superclass when tbg_model_featured class is initiated
    def add_hopping(self):

        self.tensorist.set_subspace_dicts()

        # input hopping to pythTB Hamiltonian
        subspace_dict = self.tensorist.get_largest_subspace_dict()
        for (Rx,Ry), matrix in subspace_dict.items():
            for m, n in zip(*matrix.nonzero()):
                if m!=n or Rx!=0 or Ry!=0:
                    t = matrix[m,n]/2    #divide by 2 as the conjugate pair will be summed to give extra factor of 2
                    self.model.set_hop(t, m, n, [ Rx, Ry],allow_conjugate_pair=True,mode='reset')
                else:    #on site energy
                    t_real = np.real(matrix[m,n])
                    self.model.set_onsite(t_real, int(m), mode="reset")
    
    def add_chemical_potential(self,μ):

        orbit_matrix,spin_matrix,valley_matrix = self.tensorist.get_subspace_matrices('chemical_potential')

        # enlarge subspace in order of: orbit, spin, valley, bdg
        subspace = orbit_matrix
        if self.spin_is_degen:
            subspace = self.tensorist.tensor_to_spin(spin_matrix,subspace)
        if self.valley_is_degen:
            subspace = self.tensorist.tensor_to_valley(subspace)
        if self.bdg_is_degen:
            subspace = self.tensorist.tensor_to_bdg_hopping_part(subspace)

        # input pairing to pythTB Hamiltonian
        for m, n in zip(*subspace.nonzero()):
            value = subspace[m,n]*(-μ)

            if m!=n:
                Logger.raiseException('Chemical potential is not onsite. Check physics.',exception=ValueError)
            else:    #on site energy must be real
                self.temp_model.set_onsite(value.real, m, mode='add')

    # just Rx=0,Ry=0 pairing
    def add_pairing(self,Δ,feature_mode):

        if not self.bdg_is_degen:
            Logger.raiseException('Need to set bdg_is_degen to True in order to include pairing.',exception=ValueError)

        orbit_matrix,spin_matrix,valley_matrix = self.tensorist.get_subspace_matrices(feature_mode)
        
        # enlarge subspace in order of: orbit, spin, valley, bdg
        subspace = orbit_matrix
        if self.spin_is_degen:
            subspace = self.tensorist.tensor_to_spin(spin_matrix,subspace)
        if self.valley_is_degen:
            subspace = self.tensorist.tensor_to_valley(subspace)
        bdg_subspace_csr = self.tensorist.tensor_to_bdg_pairing_part(subspace)

        # input pairing to pythTB Hamiltonian
        for m, n in zip(*bdg_subspace_csr.nonzero()):
            value = bdg_subspace_csr[m,n]*Δ
            self.temp_model.set_hop(value, m, n, [ 0, 0],mode='reset')

    # for just Δs
    def eval(self,feature_mode='hopping',path_points=['K','Γ','M','K']):

        sv.InputChecker()._check_path_points_(path_points)
        self.evaluator.set_kspace_path(path_points)
        self.models=[]
        
        if len(self.Δs)>1 and len(self.μs)>1:
            self.model_dict={}
            for Δ in self.Δs:
                for μ in self.μs:
                    self.temp_model = cp.deepcopy(self.model)
                    self.add_chemical_potential(μ)
                    self.add_pairing(Δ,feature_mode)
                    self.model_dict[(Δ,μ)] = self.temp_model
                    
            self.evaluator.eval_parameter_dict(self.model_dict,self.Δs,self.μs)
            self.plotter._axis = 'both'

        elif len(self.Δs) > 1:
            for Δ in self.Δs:
                self.temp_model = cp.deepcopy(self.model)
                self.add_pairing(Δ,feature_mode)

                if len(self.μs) == 1 and self.μs[0]!=0:
                    μ = self.μs[0]
                    self.add_chemical_potential(μ)

                self.models.append(self.temp_model)
            self.evaluator.eval_parameters(self.models,self.Δs)
            self.plotter._axis = 'Δ'

        elif len(self.μs) > 1:
            for μ in self.μs:
                self.temp_model = cp.deepcopy(self.model)
                self.add_chemical_potential(μ)

                if len(self.Δs) == 1 and self.Δs[0]!=0:
                    Δ = self.Δs[0]
                    self.add_pairing(Δ)

                self.models.append(self.temp_model)
            self.evaluator.eval_parameters(self.models,self.μs)
            self.plotter._axis = 'μ'

        else: 
            self.temp_model = cp.deepcopy(self.model)

            # nonzero single Δ
            if len(self.Δs) == 1 and self.Δs[0]!=0:
                Δ = self.Δs[0]
                self.add_pairing(Δ,feature_mode)
            
            # nonzero single μ
            if len(self.μs) == 1 and self.μs[0]!=0:
                μ = self.μs[0]
                self.add_chemical_potential(μ)
                        
            self.evaluator.eval_base(self.temp_model)
            self.plotter._axis = 'none'

        self.feature_mode = feature_mode

    def plot(self):

        self.plotter.update(self.evaluator)

        if self.plotter._axis == 'none':
            self.plotter.plot_base()

        elif self.plotter._axis == 'Δ':
            self.plotter.plot_parameters(self.Δs,param_name='Δ')

        elif self.plotter._axis == 'μ':
            self.plotter.plot_parameters(self.μs,param_name='μ')
        elif self.plotter._axis == 'both':
            self.plotter.plot_parameter_dict(self.Δs,self.μs)
        else:
            Logger.raiseException('Wrong value for private value \'_axis\'. This should not be set manually.',exception=ValueError)


    
































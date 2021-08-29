from pythtb import *
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from common.baseclasses import AWA
from NearFieldOptics.Materials.material_types import * #For Logger, overkill
from scipy.sparse import *
import pickle
import pdb
import datetime
from bdg_module import ServiceClass as serv
from bdg_module import FeatureClass as feat
import importlib
importlib.reload(serv)
importlib.reload(feat)

# dictionary of numpy matrix representation of hopping Hamiltonian
def get_hopping_Hamiltonian_dict(model,num_band=5, spin_degeneracy=2,valley_degeneracy=1,bdg_degeneracy=2):
    """(verb)
        
    Args:
        

    Return:
        void

    """

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
    """(verb)
        
    Args:
        

    Return:
        void

    """

    if type(file_name) != str:
        Logger.raiseException('Wrong type for file_name input. Need to be of type string', exception=TypeError)
    file = open(file_name+'.pickle', 'wb')
    pickle.dump(model, file)
    file.close()

def pickle_load(file_name):
    """(verb)
        
    Args:
        

    Return:
        void

    """

    if type(file_name) != str:
        Logger.raiseException('Wrong type for file_name input. Need to be of type string', exception=TypeError)

    file = open(file_name+'.pickle', 'rb')
    model = pickle.load(file)
    file.close()
    return model




class tight_binding_model():
    """tight_binding_model class 
        
    Attributes:
        
    """

    def __init__(self,file_name='source/hmat_5band_bot_1p10_rewan_RS.dat',layer_material='tbg',\
                lattice=[[1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]],real_orbit_position=False,orbital_symmetry=False,\
                spin_is_degen=False,valley_is_degen=False,bdg_is_degen=False,complex_conjugate_all_hopping=False,\
                kspace_numsteps=100,include_hopping=True,conjugate_for_hopping=True):
        """Construct a tight_binding_model object. 
        
        Args:
            
    
        Return:
            void
    
        """
        
        if layer_material=='tbg':
            num_band = int(file_name[12])    #based on the file naming convention in Carr's Github repo
        elif layer_material=='monolayer graphene':
            num_band = 2
        elif layer_material=='kagome':
            num_band = 3
        else:
            Logger.raiseException('Invalid input for \"layer_material\". Only accept \"tbg\", \"monolayer graphene\" or \"kagome\"',\
                exception=ValueError)

        # filename is not checked, num_band is checked instead
        # will raiseException if input type/value is not as expected
        checker = serv.InputChecker(num_band=num_band,lattice=lattice,real_orbit_position=real_orbit_position,\
                                spin_is_degen=spin_is_degen,valley_is_degen=valley_is_degen,bdg_is_degen=bdg_is_degen,\
                                orbital_symmetry=orbital_symmetry,kspace_numsteps=kspace_numsteps)
        checker.check_argument_clean()

        self.data = np.loadtxt(file_name)
        self.layer_material = layer_material
        self.num_band = num_band
        if self.layer_material=='tbg' and self.num_band==5:
            self._set_top_or_bottom_for_5band(file_name)
        self._set_degeneracy(spin_is_degen,valley_is_degen,bdg_is_degen)
        self.lattice = lattice
        self._set_orbitals(real_orbit_position,orbital_symmetry)
        self.kspace_numsteps = kspace_numsteps
        self._set_helpers()

        self.include_hopping = include_hopping
        self.conjugate_for_hopping = conjugate_for_hopping

        self._set_model(complex_conjugate_all_hopping=complex_conjugate_all_hopping)

    def _set_top_or_bottom_for_5band(self,file_name):
        """(Private method) set the model type to be either top or bottom based on the input data file name.
            Naming convention is based on that of the TBG tight-binding parameters in Stephen Carr's Github repo.
        
        Args:
            file_name (string): the name of the data file; has to be in the convention of Carr et al. 
    
        Return:
            void
    
        """
        self._top_or_bottom = file_name[18:21]

    def _set_degeneracy(self,spin_is_degen,valley_is_degen,bdg_is_degen):
        """(verb)
        
        Args:
            
    
        Return:
            void
    
        """
        
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
        """(verb)
        
        Args:
            
    
        Return:
            void
    
        """

        orbiter = feat.Orbiter(num_band=self.num_band,real_orbit_position=real_orbit_position,spin_degeneracy=self.spin_degeneracy,\
                            valley_degeneracy=self.valley_degeneracy,bdg_degeneracy=self.bdg_degeneracy)
        self.orbit = orbiter.construct_orbit()
        self.orbital_symmetry = orbital_symmetry

    def _set_helpers(self):
        """(verb)
        
        Args:
            
    
        Return:
            void
    
        """

        self.evaluator = serv.Evaluator(num_band=self.num_band,spin_degeneracy=self.spin_degeneracy,valley_degeneracy=self.valley_degeneracy,\
                                    bdg_degeneracy=self.bdg_degeneracy,spin_is_degen=self.spin_is_degen,kspace_numsteps=self.kspace_numsteps)
        self.plotter = serv.Plotter(num_band=self.num_band,spin_degeneracy=self.spin_degeneracy,valley_degeneracy=self.valley_degeneracy,\
                                    bdg_degeneracy=self.bdg_degeneracy)
        self.tensorist = feat.Tensorist(data=self.data,layer_material=self.layer_material,num_band=self.num_band,\
                                        spin_is_degen=self.spin_is_degen,valley_is_degen=self.valley_is_degen,bdg_is_degen=self.bdg_is_degen)

    def _set_model(self,complex_conjugate_all_hopping=False):
        """(verb)
        
        Args:
            
    
        Return:
            void
    
        """

        # the dimension of real space and k-space are both 2
        self.model = tb_model(2,2,self.lattice,self.orbit) 
        self.temp_model = None
        self.models = None
        if self.include_hopping:
            self.add_hopping(complex_conjugate_all_hopping=complex_conjugate_all_hopping)
        if self.layer_material=='tbg':
            if self.num_band==5:
                if self._top_or_bottom=='bot':
                    self.add_chemical_potential(μ=-0.00800504,add_to_base_model=True)    # to adjust for the offset in Carr's model
                elif self._top_or_bottom=='top':
                    self.add_chemical_potential(μ=-0.00824911,add_to_base_model=True)    # to adjust for the offset in Carr's model
                else:
                    Logger.raiseException('\'_top_or_bottom\' parameter for tbg can only be either \'top\' or \'bot\'.',exception=ValueError)
            elif self.num_band==8:
                self.add_chemical_potential(μ=-0.008394419,add_to_base_model=True)    # to adjust for the offset in Carr's model
            else: 
                Logger.raiseException('\'num_band\' parameter for tbg can only be either 5 or 8.',exception=ValueError)

    
    def add_hopping(self,complex_conjugate_all_hopping=False):
        """(verb)

        Specify hopping term for bare Hamiltonian; called in superclass when tbg_model_featured class is initiated
        
        Args:
            
    
        Return:
            void
    
        """

        self.tensorist.set_subspace_dicts()

        state = True

        # input hopping to pythTB Hamiltonian
        subspace_dict = self.tensorist.get_largest_subspace_dict()
        for (Rx,Ry), matrix in subspace_dict.items():
            for m, n in zip(*matrix.nonzero()):
                if m!=n or Rx!=0 or Ry!=0:
                    
                    if self.conjugate_for_hopping:
                        t = matrix[m,n]/2    #divide by 2 as the conjugate pair will be summed to give extra factor of 2
                        if complex_conjugate_all_hopping:
                            t = np.conj(t)
                            if state:
                                print('complex conjugating hopping parameter t')
                                state = False
                        self.model.set_hop(t, m, n, [ Rx, Ry],allow_conjugate_pair=True,mode='reset')
                    else:
                        t = matrix[m,n]
                        if complex_conjugate_all_hopping:
                            t = np.conj(t)
                            if state:
                                print('complex conjugating hopping parameter t')
                                state = False
                        self.model.set_hop(t, m, n, [ Rx, Ry],mode='reset')
                else:    #on site energy
                    t_real = np.real(matrix[m,n])
                    self.model.set_onsite(t_real, int(m), mode="reset")
    
    def add_chemical_potential(self,μ,add_to_base_model=False):
        """(verb)
        
        Args:
            
    
        Return:
            void
    
        """

        orbit_matrix_dict,spin_matrix,valley_matrix = self.tensorist.get_subspace_matrix_dicts(feature_mode='chemical potential')
        orbit_matrix = orbit_matrix_dict[(0,0)]   # chemical potential is onsite, so entry is only nonzero for Rx=Ry=0
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
                if add_to_base_model:
                    self.model.set_onsite(value.real, m, mode='add')
                else:
                    self.temp_model.set_onsite(value.real, m, mode='add')

    def add_pairing(self,Δ,feature_mode):
        """(verb)
        
        Args:
            
    
        Return:
            void
    
        """

        if not self.bdg_is_degen:
            Logger.raiseException('Need to set bdg_is_degen to True in order to include pairing.',exception=ValueError)
        orbit_matrix_dict,spin_matrix,valley_matrix = self.tensorist.get_subspace_matrix_dicts(feature_mode)
        
        message = True
        # enlarge subspace in order of: orbit, spin, valley, bdg
        for (Rx,Ry),orbit_matrix in orbit_matrix_dict.items():
            subspace = orbit_matrix
            if self.spin_is_degen:
                subspace = self.tensorist.tensor_to_spin(spin_matrix,subspace)
            if self.valley_is_degen:
                if feature_mode == "(C3) spin singlet, inter-valley singlet iτ2, inter-sublattice singlet iσ2" \
                or feature_mode == "(M_y 3 fold) spin singlet, inter-valley singlet iτ2, inter-sublattice singlet iσ2":
                    if message:
                        print('tensor to τ2')
                        message = False
                    subspace = self.tensorist.tensor_to_valley_pairing_part_iτ2_C3_intersublattice(subspace)
                elif feature_mode == "(C3) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1" \
                or feature_mode == "(C3) spin singlet, inter-valley triplet τ1, intra-sublattice triplet σ0" \
                or feature_mode == "(chiral) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1"\
                or feature_mode == "(chiral) spin singlet, inter-valley triplet τ1, intra-sublattice triplet σ0"\
                or feature_mode == "(total chiral) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1"\
                or feature_mode == "(M_y 3 fold) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1":
                    if message:
                        print('tensor to τ1')
                        message = False
                    subspace = self.tensorist.tensor_to_valley_pairing_part_τ1_C3_intersublattice(subspace)
                elif feature_mode == "(C3) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0" \
                or feature_mode == "(C3) spin singlet, intra-valley triplet τ0, inter-sublattice triplet σ1" \
                or feature_mode == "(chiral) spin singlet, intra-valley triplet τ0, inter-sublattice triplet σ1"\
                or feature_mode == "(chiral) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0"\
                or feature_mode == "(total chiral) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0"\
                or feature_mode == "(M_y 3 fold) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0":
                    if message:
                        print('tensor to τ0')
                        message = False
                    subspace = self.tensorist.tensor_to_valley_pairing_part_τ0_C3_intrasublattice(subspace)
                else:
                    subspace = self.tensorist.tensor_to_valley_pairing_part(valley_matrix,subspace)
            bdg_subspace_csr = self.tensorist.tensor_to_bdg_pairing_part(subspace)

            # input pairing to pythTB Hamiltonian
            for m, n in zip(*bdg_subspace_csr.nonzero()):
                value = bdg_subspace_csr[m,n]*Δ
                self.temp_model.set_hop(value, m, n, [ Rx, Ry],mode='reset')

    def add_feature(self,feature_mode='hopping'):
        """Add the pairing to the Hamiltonian matrix.

        Based on the input The pairing amplitude (Δ) and chemical potential (μ) will be added for each 
        
        Args:
            feature_mode (string): the type of pairing matrix. 
    
        Return:
            void
    
        """

        self.models=[]

        if len(self.Δs)>1 and len(self.μs)>1:
            self.model_dict={}
            for Δ in self.Δs:
                for μ in self.μs:
                    self.temp_model = cp.deepcopy(self.model)
                    self.add_chemical_potential(μ)
                    self.add_pairing(Δ,feature_mode)
                    self.model_dict[(Δ,μ)] = self.temp_model
                
        elif len(self.Δs) > 1:
            for Δ in self.Δs:
                self.temp_model = cp.deepcopy(self.model)
                self.add_pairing(Δ,feature_mode)

                if len(self.μs) == 1 and self.μs[0]!=0:
                    μ = self.μs[0]
                    self.add_chemical_potential(μ)

                self.models.append(self.temp_model)
            
        elif len(self.μs) > 1:
            for μ in self.μs:
                self.temp_model = cp.deepcopy(self.model)
                self.add_chemical_potential(μ)

                if len(self.Δs) == 1 and self.Δs[0]!=0:
                    Δ = self.Δs[0]
                    self.add_pairing(Δ,feature_mode)

                self.models.append(self.temp_model)
            
        else: 
            self.temp_model = cp.deepcopy(self.model)

            # nonzero single Δ
            if len(self.Δs) == 1 and self.Δs[0]!=0:
                Δ = self.Δs[0]
                print('Create model for single Δ='+str(Δ))
                self.add_pairing(Δ,feature_mode)
            
            # nonzero single μ
            if len(self.μs) == 1 and self.μs[0]!=0:
                μ = self.μs[0]
                print('Create model for single μ='+str(μ))
                self.add_chemical_potential(μ)
                        
        self.evaluator.feature_mode = feature_mode


    def eval(self,feature_mode='hopping',Δs=[0],μs=[0],\
        path_points=['K','Γ','M','K'],eval_all_k=False,cal_eig_vectors=False,\
        k_mesh_dim=(100,100),restrict_k_mesh=False,k_mesh_center=None,k_mesh_length=None,time_stamp=False):
        """Evaluate the eigenvalues of the pythTB model(s) at k points. 
        
        Args:
            feature_mode (string): Determine the pairing (if no pairing, it is the default: 'hopping'). For a full list of supported pairing feature, check out the Tensorist Class within the FeatureClass.
            path_points (array): A list of strings of special points in the Brillouin zone. k points will be evaluated at straight line connecting these special points.
            eval_all_k (boolean): Evaluate at a uniform square mesh of k points covering the Brillouin zone.
            cal_eig_vectors (boolean): If True, the eigenvectors are also calculated.
            k_mesh_dim (2-tuple): the number of k points in a uniform square mesh x and y direction.
            restrict_k_mesh (boolean): If True, the uniform square mesh will not cover the entire Brillouin zone.
            k_mesh_center (2-tuple): the x,y coordiante of the center of the square mesh of k points (in the reduced coordinates in k-space). Only used if restrict_k_mesh=True.
            k_mesh_length (float): the side length of the square mesh of k points (in the reduced coordinates in k-space). Only used if restrict_k_mesh=True.
            time_stamp (boolean): provide the total time of calculation, and time for each set of (Δ,μ) parameters.
    
        Return:
            void
    
        """

        if time_stamp:
            start_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
            print("Start datetime:",start_time)

        self.Δs = Δs
        self.μs = μs
        serv.InputChecker()._check_path_points_(path_points)
        self.evaluator.set_kspace_path(path_points)
        self.models=[]
        self.evaluator.eval_all_k = eval_all_k
        self.evaluator.restrict_k_mesh = restrict_k_mesh
        self.evaluator.cal_eig_vectors = cal_eig_vectors
        if eval_all_k or restrict_k_mesh:
            self.evaluator.k_mesh_dim = k_mesh_dim
        if restrict_k_mesh:
            self.evaluator.k_mesh_center = k_mesh_center
            self.evaluator.k_mesh_length = k_mesh_length

        self.add_feature(feature_mode=feature_mode)

        if len(self.Δs)>1 and len(self.μs)>1:
            self.evaluator.eval_parameter_dict(self.model_dict,self.Δs,self.μs,time_stamp=time_stamp)
            self.plotter._axis = 'both'
        elif len(self.Δs) > 1:
            self.evaluator.eval_parameters(self.models,self.Δs,time_stamp=time_stamp)
            self.plotter._axis = 'Δ'
        elif len(self.μs) > 1:
            self.evaluator.eval_parameters(self.models,self.μs,time_stamp=time_stamp)
            self.plotter._axis = 'μ'
        else: 
            self.evaluator.eval_base(self.temp_model,time_stamp=time_stamp)
            self.plotter._axis = 'none'

        if time_stamp:
            end_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
            print("\nEnd datetime:", end_time)




    #TO-DOs: 
    #boolean selection for either inline or interactive plot. 
    #boolean for save figure.
    def plot(self,xlim=None,ylim=None,zlim=None,plot_3D=False,plot_contour=False,zero_crossing_threshold=0.05):
        """Plot the evaluated eigenvalues, i.e. the band structure. 

        It uses the iteractive plot widgets to control parameter value of (Δ,μ); need to enable with the command '%matplotlib notebook'.
        
        Args:
            xlim (array of length 2): The min and max value of x coordinate in the plot. 
            ylim (array of length 2): The min and max value of y coordinate in the plot. 
            zlim (array of length 2): The min and max value of z coordinate in the plot. Only if plot_3D=True.
            plot_3D (boolean): If true, plot as a 3D graph.
            plot_contour (boolean): If true, plot as a 2D contour plot. Note: the k points are in the reduced k coordinates.
            zero_crossing_threshold (float): If the eigenvalue is within the threshold distance from 0, it would light up as red in the contour plot.
    
        Return:
            void
    
        """

        self.plotter.evaluator = self.evaluator
        if plot_contour:
            self.plotter.zero_crossing_threshold = zero_crossing_threshold
        
        if self.plotter._axis == 'none':
            self.plotter.plot_base(xlim=xlim,ylim=ylim,zlim=zlim,plot_3D=plot_3D,plot_contour=plot_contour)

        elif self.plotter._axis == 'Δ':
            self.plotter.plot_parameters(self.Δs,param_name='Δ',xlim=xlim,ylim=ylim,zlim=zlim,plot_3D=plot_3D,plot_contour=plot_contour)

        elif self.plotter._axis == 'μ':
            self.plotter.plot_parameters(self.μs,param_name='μ',xlim=xlim,ylim=ylim,zlim=zlim,plot_3D=plot_3D,plot_contour=plot_contour)
        elif self.plotter._axis == 'both':
            self.plotter.plot_parameter_dict(self.Δs,self.μs,xlim=xlim,ylim=ylim,zlim=zlim,plot_3D=plot_3D,plot_contour=plot_contour)
        else:
            Logger.raiseException('Wrong value for private value \'_axis\'. This should not be set manually.',exception=ValueError)


    
































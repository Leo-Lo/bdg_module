from pythtb import *
from ipywidgets import *    #for slider when plotting
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from common.baseclasses import AWA
from scipy.sparse import *
from NearFieldOptics.Materials.material_types import * #For Logger, overkill


#Contains the physics of real orbit position; hard-coded
class Orbiter():
    
    def __init__(self,**kwargs):

        self.__dict__.update(kwargs)

    def construct_orbit(self):
        
        total_degen = self.bdg_degeneracy * self.valley_degeneracy * self.spin_degeneracy
        orb=[]
        if self.real_orbit_position:
            if self.num_band==5:
                self.add_5band_orbits(orb,total_degen)
            else:    #num_band==8
                self.add_8band_orbits(orb,total_degen)     
        else:
            for i in range(0,self.num_band*total_degen,1):
                orb.append([0.0,0.0])
        return orb
    
    def add_5band_orbits(self,orb,total_degen):
        # in units of lattice vectors
        for i in range(0,total_degen,1):
            orb.append([0.0,0.0])
            orb.append([0.0,0.0])
            orb.append([0.0,0.0])
            orb.append([2.0/3,2.0/3])
            orb.append([1.0/3,1.0/3])
    
    def add_8band_orbits(self,orb,total_degen):
        # in units of lattice vectors
        for i in range(0,total_degen,1):
            #triangular
            orb.append([0.0,0.0])
            orb.append([0.0,0.0])
            orb.append([0.0,0.0])
            #hexagonal
            orb.append([2.0/3,2.0/3])
            orb.append([1.0/3,1.0/3])
            #kagome
            orb.append([0.5,0.0])
            orb.append([0.5,0.5])
            orb.append([0.0,0.5])



# Contains the physics 
class Tensorist():

    def __init__(self,**kwargs):

        self.__dict__.update(kwargs)

        self.set_degeneracy()
        self.orbit_matrix_dict = {}
        self.spin_subspace_dict = {}
        self.valley_subspace_dict = {}
        self.bdg_subspace_dict = {}


    # Contains all pairing mechanism physics
    def get_subspace_matrices(self,feature_mode):

        if self.valley_is_degen:
            valley_matrix = np.matrix([[0,0],[0,1]])
        else:
            valley_matrix = 1

        if feature_mode == "hopping":
            orbit_matrix = None
            spin_matrix = np.identity(self.spin_degeneracy)
        
        elif feature_mode == "chemical potential":

        	orbit_matrix = np.identity(self.num_band)
        	spin_matrix = np.identity(self.spin_degeneracy)


        elif feature_mode == "monolayer onsite spin singlet":

            if self.num_band != 2:
                Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

            if self.spin_is_degen:
                spin_matrix = np.matrix([[0,1],
                                         [-1,0]])
            else:
                spin_matrix = 1

            orbit_matrix = np.matrix([
                                     [ 1, 0],
                                     [ 0, 1]
                                    ])

        elif feature_mode == "intra cell spin singlet":
            if self.spin_is_degen:
                spin_matrix = np.matrix([[0,1],
                                         [-1,0]])
            else:
                Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

            if self.num_band == 5:
                orbit_matrix = np.matrix([
                                        [ 0, 1, 0, 0, 0],
                                        [ 1, 0, 0, 0, 0],
                                        [ 0, 0, 1, 0, 0],
                                        [ 0, 0, 0, 1, 0],
                                        [ 0, 0, 0, 0, 1]
                                        ])
            elif self.num_band == 8:
                orbit_matrix = np.matrix([
                                        [ 0, 1, 0, 0, 0, 0, 0, 0],
                                        [ 1, 0, 0, 0, 0, 0, 0, 0],
                                        [ 0, 0, 1, 0, 0, 1, 1, 1],
                                        [ 0, 0, 0, 1, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 1, 0, 0, 0],
                                        [ 0, 0, 1, 0, 0, 1, 0, 0],
                                        [ 0, 0, 1, 0, 0, 0, 1, 0],
                                        [ 0, 0, 1, 0, 0, 0, 0, 1],
                                        ])
            
            else:
                Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)
            
        elif feature_mode == "intra cell triplet antisymmetric orbit":
            if self.spin_is_degen:
                spin_matrix = np.matrix([[0,1],
                                         [1,0]])
            else: 
                spin_matrix = 1
            if self.num_band == 5:
                orbit_matrix = np.matrix([
                                        [ 0, 1, 0, 0, 0],
                                        [-1, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0]
                                        ])
            elif self.num_band == 8:
                orbit_matrix = np.matrix([
                                        [ 0, 1, 0, 0, 0, 0, 0, 0],
                                        [-1, 0, 0, 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0, 1, 1, 1],
                                        [ 0, 0, 0, 0, 1, 0, 0, 0],
                                        [ 0, 0, 0,-1, 0, 0, 0, 0],
                                        [ 0, 0,-1, 0, 0, 0, 1, 1],
                                        [ 0, 0,-1, 0, 0,-1, 0, 1],
                                        [ 0, 0,-1, 0, 0,-1,-1, 0],
                                        ])
            else:
                Logger.raiseException('Invalid value of num_band, can only be 2,5,8',exception=ValueError)
            
        elif feature_mode == "intra cell triplet antisymmetric hopping":
            spin_matrix = np.identity(self.spin_degeneracy)
            orbit_matrix = np.identity(self.num_band)
            
        else:
            Logger.raiseException('Wrong value for \"feature_mode\". Only accept '+ str(self.feature_mode_list),exception=ValueError)

        orbit_matrix_csr = csr_matrix(orbit_matrix,dtype='complex64')
        spin_matrix_csr = csr_matrix(spin_matrix,dtype='complex64')
        valley_matrix_csr = csr_matrix(valley_matrix,dtype='complex64')

        return orbit_matrix_csr,spin_matrix_csr,valley_matrix_csr

    def get_subspace_matrix_dicts(self,feature_mode):

        if self.valley_is_degen:
            valley_matrix = np.matrix([[0,0],[0,1]])
        else:
            valley_matrix = 1
        
        
        if feature_mode == "monolayer onsite intra unit cell spin singlet":

            if self.num_band != 2:
                Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

            if self.spin_is_degen:
                spin_matrix = np.matrix([[0,1],
                                         [-1,0]])
            else:
                spin_matrix = 1

            orbit_matrix_intra = np.matrix([
                                       [ 0, 1],
                                       [ 1, 0]
                                       ])

            orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

            orbit_dict = {
                            (0,0): orbit_matrix_intra_csr,
                            }

        elif feature_mode == "monolayer onsite spin singlet":

            if self.num_band != 2:
                Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

            if self.spin_is_degen:
                spin_matrix = np.matrix([[0,1],
                                         [-1,0]])
            else:
                spin_matrix = 1

            orbit_matrix_intra = np.matrix([
                                       [ 0, 1],
                                       [ 1, 0]
                                       ])

            orbit_matrix_inter = np.matrix([
                                       [ 0, 1],
                                       [ 1, 0]
                                       ])

            orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')
            orbit_matrix_inter_csr = csr_matrix(orbit_matrix_inter,dtype='complex64')

            orbit_dict = {
                            (0,0): orbit_matrix_intra_csr,
                            (1,0): orbit_matrix_inter_csr,
                            (0,1): orbit_matrix_inter_csr
                            }

        else:
            Logger.raiseException('Wrong value for \"feature_mode\". Only accept '+ str(self.feature_mode_list),exception=ValueError)

        spin_matrix_csr = csr_matrix(spin_matrix,dtype='complex64')
        valley_matrix_csr = csr_matrix(valley_matrix,dtype='complex64')

        return orbit_dict,spin_matrix_csr,valley_matrix_csr


    def set_subspace_dicts(self):

        self.set_orbit_matrix_dict()
        if self.spin_is_degen:
            self.set_spin_subspace_dict()
        if self.valley_is_degen:
            self.set_valley_subspace_dict()
        if self.bdg_is_degen:
            self.set_bdg_subspace_dict()

    def set_orbit_matrix_dict(self):

        orbit_matrix_dict_lil = {}

        for i in range(0,len(self.data),1):
            t_real = self.data[i,4]
            t_imag = self.data[i,5]
            t = (t_real+1j*t_imag)
            m = self.data[i,2]-1
            n = self.data[i,3]-1
            Rx = self.data[i,0]
            Ry = self.data[i,1]
            try:
                orbit_matrix_dict_lil[(Rx,Ry)][m,n] = t
            except:    # catch the exception the there is no key with value (Rx,Ry); instantiate a new matrix
                orbit_matrix_dict_lil[(Rx,Ry)] = lil_matrix((self.num_band, self.num_band), dtype='complex64')
                orbit_matrix_dict_lil[(Rx,Ry)][m,n] = t

        # store matrices as csr format in dictionary
        for (Rx,Ry), orbit_matrix_lil in orbit_matrix_dict_lil.items():
            self.orbit_matrix_dict[(Rx,Ry)] = orbit_matrix_lil.tocsr()

    def set_spin_subspace_dict(self):

        _,spin_matrix,_ = self.get_subspace_matrices(feature_mode="hopping")
        for (Rx,Ry), orbit_matrix in self.orbit_matrix_dict.items():
            spin_subspace_csr = self.tensor_to_spin(spin_matrix,orbit_matrix)
            self.spin_subspace_dict[(Rx,Ry)] = spin_subspace_csr

    def set_valley_subspace_dict(self):

        if self.spin_is_degen:
            subspace_dict = self.spin_subspace_dict
        else:
            subspace_dict = self.orbit_matrix_dict

        for (Rx,Ry), subspace in subspace_dict.items():
            valley_subspace_csr = self.tensor_to_valley(subspace)
            self.valley_subspace_dict[(Rx,Ry)] = valley_subspace_csr

    def set_bdg_subspace_dict(self):

        # set subspace_dict to largest possible allowed by degeneracy
        if self.valley_is_degen:
            subspace_dict = self.valley_subspace_dict
        elif self.spin_is_degen:
            subspace_dict = self.spin_subspace_dict
        else: 
            subspace_dict = self.orbit_matrix_dict

        for (Rx,Ry), subspace in subspace_dict.items():
            bdg_subspace_csr = self.tensor_to_bdg_hopping_part(subspace)
            self.bdg_subspace_dict[(Rx,Ry)] = bdg_subspace_csr
    
    def tensor_to_spin(self,spin_matrix,subspace):

        spin_subspace = kron(spin_matrix,subspace)
        spin_subspace_csr = spin_subspace.tocsr()
        return spin_subspace_csr
        
    def tensor_to_valley(self,subspace):

        # set lower right block matrix: complex conjugate of subspace matrix
        lower_right_subspace = subspace.conj()
        lower_right_matrix = np.matrix([[0,0],[0,1]])
        lower_right = kron(lower_right_matrix,lower_right_subspace)
        lower_right_csr = lower_right.tocsr()

        # set upper left block matrix: original subspace matrix
        upper_left_matrix = np.matrix([[1,0],[0,0]])
        upper_left = kron(upper_left_matrix,subspace)
        upper_left_csr = upper_left.tocsr()
        
        valley_subspace_csr = lower_right_csr + upper_left_csr
        return valley_subspace_csr

    def tensor_to_bdg_hopping_part(self,subspace):

        # set lower right bdg block matrix: negative complex conjugate of Hamiltonian
        lower_right_subspace = subspace.conj()*(-1)
        lower_right_matrix = np.matrix([[0,0],[0,1]])
        lower_right = kron(lower_right_matrix,lower_right_subspace)
        lower_right_csr = lower_right.tocsr()

        # set upper left bdg block matrix: original Hamiltonian
        upper_left_matrix = np.matrix([[1,0],[0,0]])
        upper_left = kron(upper_left_matrix,subspace)
        upper_left_csr = upper_left.tocsr()
        
        bdg_subspace_csr = lower_right_csr + upper_left_csr
        
        return bdg_subspace_csr

    def tensor_to_bdg_pairing_part(self,subspace):

        # set upper right bdg block matrix unmodified; lower left is automatically taken care of by PythTB
        upper_right_matrix = np.matrix([[0,1],[0,0]])
        bdg_subspace = kron(upper_right_matrix,subspace)
        bdg_subspace_csr = bdg_subspace.tocsr()
        return bdg_subspace_csr

    def set_degeneracy(self):
        
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

    def get_largest_subspace_dict(self):

        if self.bdg_is_degen:
            return self.bdg_subspace_dict
        elif self.valley_is_degen:
            return self.valley_subspace_dict
        elif self.spin_is_degen:
            return self.spin_subspace_dict
        else:
            return self.orbit_matrix_dict
        



















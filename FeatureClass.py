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
            elif self.num_band==8:
                self.add_8band_orbits(orb,total_degen)     
            elif self.num_band==2:
                self.add_2band_orbits(orb,total_degen)
            else: 
                Logger.raiseException('Incorrect num_band, only accept 5 or 8.',exception=ValueError)
        else:
            for i in range(0,self.num_band*total_degen,1):
                orb.append([0.0,0.0])
        return orb
    
    def add_2band_orbits(self,orb,total_degen):
        # in units of lattice vectors
        for i in range(0,total_degen,1):
            orb.append([1.0/3,1.0/3])
            orb.append([2.0/3,2.0/3])

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



# Contains the physics of pairing mechanism
class Tensorist():

    def __init__(self,**kwargs):

        self.__dict__.update(kwargs)

        self.set_degeneracy()
        self.orbit_matrix_dict = {}
        self.spin_subspace_dict = {}
        self.valley_subspace_dict = {}
        self.bdg_subspace_dict = {}

    def get_subspace_matrix_dicts(self,feature_mode):

        use_default_valley = True

        if feature_mode == "hopping":
            orbit_matrix = None
            spin_matrix = np.identity(self.spin_degeneracy)
            orbit_dict = {(0,0): orbit_matrix}
        
        elif feature_mode == "chemical potential":

            orbit_matrix = np.identity(self.num_band)
            spin_matrix = np.identity(self.spin_degeneracy)
            orbit_dict = {(0,0): orbit_matrix}

        else:

            if self.layer_material=='monolayer graphene':

                if feature_mode == "monolayer intra-unit-cell inter-sublattice spin singlet":

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

                elif feature_mode == "monolayer inter-sublattice spin singlet (3 fold symmetric)":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix_both = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])

                    orbit_matrix_up = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])

                    orbit_matrix_down = np.matrix([
                                               [ 0, 0],
                                               [ 1, 0]
                                               ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 0, 0): orbit_matrix_both_csr,
                                    ( 1, 0): orbit_matrix_down_csr,
                                    (-1, 0): orbit_matrix_up_csr,
                                    ( 0, 1): orbit_matrix_down_csr,
                                    ( 0,-1): orbit_matrix_up_csr
                                    }
                elif feature_mode == "monolayer inter-sublattice spin singlet (3 fold chiral clockwise)":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix_both = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])

                    orbit_matrix_up = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])

                    orbit_matrix_down = np.matrix([
                                               [ 0, 0],
                                               [ 1, 0]
                                               ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 0, 0): orbit_matrix_both_csr,
                                    ( 1, 0): orbit_matrix_down_csr*np.exp(1j*2*np.pi/3),
                                    (-1, 0): orbit_matrix_up_csr*np.exp(1j*2*np.pi/3),
                                    ( 0, 1): orbit_matrix_down_csr*np.exp(1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)
                                    }

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
                    orbit_dict = {
                                    (0,0): orbit_matrix,
                                    }

                elif feature_mode == "monolayer intra-unit-cell inter-sublattice spin triplet":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix = np.matrix([
                                             [ 0, 1],
                                             [-1, 0]
                                            ])
                    orbit_dict = {
                                    (0,0): orbit_matrix,
                                    }
                
                # deprecated pairing mode: valley degree of freedom just duplicate the spectrum, as the hopping parameters are all real
                elif feature_mode == "spin singlet, valley triplet τ1, intra-unit-cell inter-sublattice triplet σ1":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_intra = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                elif feature_mode == "spin singlet, valley triplet τ0, intra-unit-cell inter-sublattice triplet σ1":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 1]
                                               ])
                    use_default_valley = False

                    orbit_matrix_intra = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                elif feature_mode == "spin singlet, valley triplet τ1, intra-unit-cell onsite triplet σ0":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_intra = np.matrix([
                                               [ 1, 0],
                                               [ 0, 1]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                elif feature_mode == "spin singlet, valley triplet τ0, intra-unit-cell onsite triplet σ0":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 1]
                                               ])
                    use_default_valley = False

                    orbit_matrix_intra = np.matrix([
                                               [ 1, 0],
                                               [ 0, 1]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                elif feature_mode == "spin singlet, valley singlet, intra-unit-cell inter-sublattice singlet":

                    if self.num_band != 2:
                        Logger.raiseException('Can only use this feature mode for the monolayer case, i.e. self.num_band = 2.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [-1, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_intra = np.matrix([
                                               [ 0, 1],
                                               [-1, 0]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                
                else: 
                    Logger.raiseException('Invalid feature mode for self.layer_material=monolayer graphene.', exception=ValueError)

            elif self.layer_material=='tbg':

                if feature_mode == "intra cell spin singlet":
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

                    orbit_dict = {(0,0): orbit_matrix}

                ##### pairing with no explicit valley degree of freedom ######
                elif feature_mode == "spin singlet inter-sublattice intra-unit-cell 12":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                elif feature_mode == "spin singlet inter-sublattice intra-unit-cell 11 22":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                elif feature_mode == "(C3) spin singlet, intra-sublattice triplet σ0":# this pairing is the sub-block from intra-valley pairing
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    orbit_matrix_both = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)
                                    }
                elif feature_mode == "(intra-unit-cell) spin singlet, intra-sublattice triplet σ0":# this pairing is the sub-block from intra-valley pairing
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                    

                ##### pairing with explicit valley degree of freedom ######
                # valley triplet inter-sublattice
                elif feature_mode == "(intra-unit-cell) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])
                    use_default_valley = False

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                elif feature_mode == "(intra-unit-cell) spin singlet, intra-valley triplet τ0, inter-sublattice triplet σ1":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 1]
                                               ])
                    use_default_valley = False

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                elif feature_mode == "(intra-unit-cell) spin singlet, intra-valley triplet τ3, inter-sublattice triplet σ1":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0,-1]
                                               ])
                    use_default_valley = False

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                # valley triplet intra-sublattice
                elif feature_mode == "(intra-unit-cell) spin singlet, inter-valley triplet τ1, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 1, 0]
                                               ])
                    use_default_valley = False

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                elif feature_mode == "(intra-unit-cell) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 1]
                                               ])
                    use_default_valley = False

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                elif feature_mode == "(intra-unit-cell) spin singlet, intra-valley triplet τ3, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0,-1]
                                               ])
                    use_default_valley = False

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}
                # valley singlet
                elif feature_mode == "(intra-unit-cell) spin singlet, inter-valley singlet iτ2, inter-sublattice singlet iσ2":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [-1, 0]
                                               ])
                    use_default_valley = False

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
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}

                ##### C3 pairing ######
                elif feature_mode == "(C3) spin singlet, inter-valley singlet iτ2, inter-sublattice singlet iσ2":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [-1, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [-1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [-1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)
                                    }
                elif feature_mode == "(C3) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)
                                    }
                elif feature_mode == "(C3) spin singlet, inter-valley triplet τ1, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_both_csr,
                                    ( 1,-1): orbit_matrix_both_csr,
                                    ( 0,-1): orbit_matrix_both_csr,
                                    ( 0, 1): orbit_matrix_both_csr
                                    }
                elif feature_mode == "(C3) spin singlet, intra-valley triplet τ0, inter-sublattice triplet σ1":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_both_csr,
                                    ( 1,-1): orbit_matrix_both_csr,
                                    ( 0,-1): orbit_matrix_both_csr,
                                    ( 0, 1): orbit_matrix_both_csr
                                    }
                elif feature_mode == "(C3) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)
                                    }

                ##### 3-fold chiral pairing ######
                elif feature_mode == "(chiral) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)*np.exp(1j*2*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3)*np.exp(-1j*2*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)*np.exp(1j*2*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3)*np.exp(-1j*2*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)*np.exp(-1j*4*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)*np.exp(-1j*4*np.pi/3)
                                    }
                elif feature_mode == "(chiral) spin singlet, inter-valley triplet τ1, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_both_csr*np.exp(1j*2*np.pi/3),
                                    ( 1,-1): orbit_matrix_both_csr*np.exp(1j*2*np.pi/3),
                                    ( 0,-1): orbit_matrix_both_csr*np.exp(1j*4*np.pi/3),
                                    ( 0, 1): orbit_matrix_both_csr*np.exp(1j*4*np.pi/3)
                                    }
                elif feature_mode == "(chiral) spin singlet, intra-valley triplet τ0, inter-sublattice triplet σ1":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_both_csr*np.exp(1j*2*np.pi/3),
                                    ( 1,-1): orbit_matrix_both_csr*np.exp(1j*2*np.pi/3),
                                    ( 0,-1): orbit_matrix_both_csr*np.exp(1j*4*np.pi/3),
                                    ( 0, 1): orbit_matrix_both_csr*np.exp(1j*4*np.pi/3)
                                    }
                elif feature_mode == "(chiral) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)*np.exp(1j*2*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3)*np.exp(-1j*2*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)*np.exp(1j*2*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3)*np.exp(-1j*2*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)*np.exp(-1j*4*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)*np.exp(-1j*4*np.pi/3)
                                    }

                ##### 3-fold total chiral pairing ######
                elif feature_mode == "(total chiral) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): (orbit_matrix_up_csr*np.exp(1j*4*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3))*np.exp(1j*2*np.pi/3),
                                    ( 1,-1): (orbit_matrix_up_csr*np.exp(1j*4*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3))*np.exp(1j*2*np.pi/3),
                                    ( 0,-1): (orbit_matrix_up_csr*np.exp(1j*8*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3))*np.exp(1j*4*np.pi/3),
                                    ( 0, 1): (orbit_matrix_up_csr*np.exp(1j*8*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3))*np.exp(1j*4*np.pi/3)
                                    }
                elif feature_mode == "(total chiral) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): (orbit_matrix_up_csr*np.exp(1j*4*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3))*np.exp(1j*2*np.pi/3),
                                    ( 1,-1): (orbit_matrix_up_csr*np.exp(1j*4*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3))*np.exp(1j*2*np.pi/3),
                                    ( 0,-1): (orbit_matrix_up_csr*np.exp(1j*8*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3))*np.exp(1j*4*np.pi/3),
                                    ( 0, 1): (orbit_matrix_up_csr*np.exp(1j*8*np.pi/3) + orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3))*np.exp(1j*4*np.pi/3)
                                    }



                ##### M_y 3 fold inter-sublattice pairing ######
                elif feature_mode == "(M_y 3 fold) spin singlet, inter-valley singlet iτ2, inter-sublattice singlet iσ2":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [-1, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [-1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [-1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)
                                    }
                elif feature_mode == "(M_y 3 fold) spin singlet, inter-valley triplet τ1, inter-sublattice triplet σ1":

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 0, 1],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)
                                    }
                elif feature_mode == "(M_y 3 fold) spin singlet, intra-valley triplet τ0, intra-sublattice triplet σ0":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if not self.valley_is_degen:
                        Logger.raiseException('Can only use this feature mode with valley degeneracy.', exception=ValueError)
                    # the valley_matrix will not be used; a special tensor to valley function is used instead
                    valley_matrix = np.matrix([
                                               [ 1, 0],
                                               [ 0, 0]
                                               ])
                    use_default_valley = False

                    orbit_matrix_both = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_up = np.matrix([
                                                [ 1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_down = np.matrix([
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 1, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])

                    orbit_matrix_both_csr = csr_matrix(orbit_matrix_both,dtype='complex64')
                    orbit_matrix_up_csr = csr_matrix(orbit_matrix_up,dtype='complex64')
                    orbit_matrix_down_csr = csr_matrix(orbit_matrix_down,dtype='complex64')

                    orbit_dict = {
                                    ( 1, 0): orbit_matrix_both_csr,
                                    (-1, 0): orbit_matrix_both_csr,
                                    (-1, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0, 1): orbit_matrix_up_csr*np.exp(1j*4*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*4*np.pi/3),
                                    ( 0,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3),
                                    ( 1,-1): orbit_matrix_up_csr*np.exp(1j*8*np.pi/3)+orbit_matrix_down_csr*np.exp(-1j*8*np.pi/3)
                                    }


                ##### Other pairings ######
                elif feature_mode == "spin triplet inter-sublattice intra-unit-cell 12":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if self.num_band == 5:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0],
                                                [ -1, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0]
                                                ])
                    elif self.num_band == 8:
                        orbit_matrix = np.matrix([
                                                [ 0, 1, 0, 0, 0, 0, 0, 0],
                                                [ -1, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                [ 0, 0, 0, 0, 0, 0, 0, 0],
                                                ])
                    
                    else:
                        Logger.raiseException('Invalid value of num_band, can only be 5 or 8',exception=ValueError)

                    orbit_dict = {(0,0): orbit_matrix}

                elif feature_mode == "inter-sublattice spin singlet (3 fold symmetric)(deprecated)":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [-1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if self.num_band == 5:
                        
                        orbit_dict = {}
                        orbit_dict[(0,0)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 1],
                                                            [ 0, 0, 0, 1, 0]
                                                            ])
                        orbit_dict[(-1,1)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 1, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                        orbit_dict[(1,0)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 1, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 1, 0]
                                                            ])
                        orbit_dict[(0,1)] = np.matrix([
                                                            [ 0, 1, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 1, 0]
                                                            ])
                        orbit_dict[(1,-1)] = np.matrix([
                                                            [ 0, 1, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                        orbit_dict[(-1,0)] = np.matrix([
                                                            [ 0, 1, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 1],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                        orbit_dict[(0,-1)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 1, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 1],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                    
                    # not updated yet
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
                
                elif feature_mode == "inter-sublattice spin triplet (3 fold symmetric)(deprecated)":
                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[0,1],
                                                 [1,0]])
                    else:
                        Logger.raiseException('Require spin degeneracy in order for spin singlet pairing.', exception=ValueError)

                    if self.num_band == 5:
                        
                        orbit_dict = {}
                        orbit_dict[(0,0)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 1],
                                                            [ 0, 0, 0,-1, 0]
                                                            ])
                        orbit_dict[(-1,1)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 1, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                        orbit_dict[(1,0)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 1, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 1, 0]
                                                            ])
                        orbit_dict[(0,1)] = np.matrix([
                                                            [ 0, 1, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 1, 0]
                                                            ])
                        orbit_dict[(1,-1)] = np.matrix([
                                                            [ 0,-1, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                        orbit_dict[(-1,0)] = np.matrix([
                                                            [ 0,-1, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0,-1],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                        orbit_dict[(0,-1)] = np.matrix([
                                                            [ 0, 0, 0, 0, 0],
                                                            [-1, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0, 0],
                                                            [ 0, 0, 0, 0,-1],
                                                            [ 0, 0, 0, 0, 0]
                                                            ])
                    
                    # not updated yet
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
                        
                    orbit_dict = {(0,0): orbit_matrix}
                    
                elif feature_mode == "intra cell triplet antisymmetric hopping":
                    spin_matrix = np.identity(self.spin_degeneracy)
                    orbit_matrix = np.identity(self.num_band)
                
                    orbit_dict = {(0,0): orbit_matrix}

                else: 
                    Logger.raiseException('Invalid feature mode for self.layer_material=tbg.', exception=ValueError)
            
            elif self.layer_material=='kagome':

                if feature_mode == "spin singlet, intra-unit-cell inter-sublattice 12":

                    if self.num_band != 3:
                        Logger.raiseException('Can only use this feature mode for the kagome case, i.e. self.num_band = 3.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix_intra = np.matrix([
                                               [ 0, 1, 0],
                                               [ 1, 0, 0],
                                               [ 0, 0, 0]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                elif feature_mode == "spin singlet, intra-unit-cell inter-sublattice 13":

                    if self.num_band != 3:
                        Logger.raiseException('Can only use this feature mode for the kagome case, i.e. self.num_band = 3.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix_intra = np.matrix([
                                               [ 0, 0, 1],
                                               [ 0, 0, 0],
                                               [ 1, 0, 0]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                elif feature_mode == "spin singlet, intra-unit-cell inter-sublattice 23":

                    if self.num_band != 3:
                        Logger.raiseException('Can only use this feature mode for the kagome case, i.e. self.num_band = 3.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix_intra = np.matrix([
                                               [ 0, 0, 0],
                                               [ 0, 0, 1],
                                               [ 0, 1, 0]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }
                elif feature_mode == "spin singlet, intra-unit-cell inter-sublattice 123":

                    if self.num_band != 3:
                        Logger.raiseException('Can only use this feature mode for the kagome case, i.e. self.num_band = 3.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix_intra = np.matrix([
                                               [ 0, 1, 1],
                                               [ 1, 0, 1],
                                               [ 1, 1, 0]
                                               ])

                    orbit_matrix_intra_csr = csr_matrix(orbit_matrix_intra,dtype='complex64')

                    orbit_dict = {
                                    (0,0): orbit_matrix_intra_csr,
                                    }                  
                elif feature_mode == "spin singlet, onsite 123":

                    if self.num_band != 3:
                        Logger.raiseException('Can only use this feature mode for the kagome case, i.e. self.num_band = 3.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix = np.matrix([
                                             [ 1, 0, 0],
                                             [ 0, 1, 0],
                                             [ 0, 0, 1]
                                            ])
                    orbit_dict = {
                                    (0,0): orbit_matrix,
                                    }
                elif feature_mode == "spin singlet, onsite 12":

                    if self.num_band != 3:
                        Logger.raiseException('Can only use this feature mode for the kagome case, i.e. self.num_band = 3.', exception=ValueError)

                    if self.spin_is_degen:
                        spin_matrix = np.matrix([[ 0,1],
                                                 [-1,0]])
                    else:
                        spin_matrix = 1

                    orbit_matrix = np.matrix([
                                             [ 1, 0, 0],
                                             [ 0, 1, 0],
                                             [ 0, 0, 0]
                                            ])
                    orbit_dict = {
                                    (0,0): orbit_matrix,
                                    }
                else: 
                    Logger.raiseException('Invalid feature mode for self.layer_material=kagome.', exception=ValueError)

            else:
                Logger.raiseException('Wrong value for \"feature_mode\". Check the documentation of get_subspace_matrix_dicts\
                     method, in Tensorist Class in FeatureClass module for allowed input.',exception=ValueError)

        # valley matrix
        if use_default_valley:
            if self.valley_is_degen:
                valley_matrix = np.matrix([[0,0],[0,1]])
            else:
                valley_matrix = 1

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
            m = self.data[i,2]-1    # note, this is to account for indexing starting at 1 for Carr's data
            n = self.data[i,3]-1    # note, this is to account for indexing starting at 1 for Carr's data
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

        _,spin_matrix,_ = self.get_subspace_matrix_dicts(feature_mode="hopping")
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

    def tensor_to_valley_pairing_part(self,valley_matrix,subspace):

        valley_subspace = kron(valley_matrix,subspace)
        valley_subspace_csr = valley_subspace.tocsr()
        return valley_subspace_csr

    def tensor_to_valley_pairing_part_iτ2_C3_intersublattice(self,subspace):

        # set lower left block matrix: negative complex conjugate of subspace matrix
        lower_block_subspace = -subspace.conj()
        lower_block_skeleton = np.matrix([[0,0],[1,0]])
        lower_block = kron(lower_block_skeleton,lower_block_subspace)
        lower_block_csr = lower_block.tocsr()

        # set upper right block matrix: original subspace matrix
        upper_block_skeleton = np.matrix([[0,1],[0,0]])
        upper_block = kron(upper_block_skeleton,subspace)
        upper_block_csr = upper_block.tocsr()

        valley_subspace_csr = lower_block_csr + upper_block_csr
        return valley_subspace_csr

    def tensor_to_valley_pairing_part_τ1_C3_intersublattice(self,subspace):

        # set lower left block matrix: complex conjugate of subspace matrix
        lower_block_subspace = subspace.conj()
        lower_block_skeleton = np.matrix([[0,0],[1,0]])
        lower_block = kron(lower_block_skeleton,lower_block_subspace)
        lower_block_csr = lower_block.tocsr()

        # set upper right block matrix: original subspace matrix
        upper_block_skeleton = np.matrix([[0,1],[0,0]])
        upper_block = kron(upper_block_skeleton,subspace)
        upper_block_csr = upper_block.tocsr()

        valley_subspace_csr = lower_block_csr + upper_block_csr
        return valley_subspace_csr

    def tensor_to_valley_pairing_part_τ0_C3_intrasublattice(self,subspace):

        # set lower right block matrix: complex conjugate of subspace matrix
        lower_block_subspace = subspace.conj()
        lower_block_skeleton = np.matrix([[0,0],[0,1]])
        lower_block = kron(lower_block_skeleton,lower_block_subspace)
        lower_block_csr = lower_block.tocsr()

        # set upper left block matrix: original subspace matrix
        upper_block_skeleton = np.matrix([[1,0],[0,0]])
        upper_block = kron(upper_block_skeleton,subspace)
        upper_block_csr = upper_block.tocsr()

        valley_subspace_csr = lower_block_csr + upper_block_csr
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
        



















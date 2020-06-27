from pythtb import *
from ipywidgets import *    #for slider when plotting
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from common.baseclasses import AWA
from NearFieldOptics.Materials.material_types import * #For Logger, overkill

#Contains the specification of the Brillouin zone in reciprocal space; hard-coded
# def get_kspace_path():
#     # Define plot path in k-space: K -> Γ -> M -> K
#     ## π rotation of standard path from simple graphene model
#     # angle = np.pi
#     angle = 0
#     R = np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    
#     K = R*[[1./3.], [2./3.]]
#     M = R*[[0.5], [0.5]]
#     M_minus = R*[[-0.5], [-0.5]]
    
#     K_in, M_in, Kp_in,M_minus_in = _kpt_array_to_list_(K,M,Kp,M_minus)
#     return [K_in,[0.0, 0.0],M_in,K_in]

def get_kspace_dict():
    
    # Define plot path in k-space: K -> Γ -> M -> K
    ## π rotation of standard path from simple graphene model
    angle = np.pi
    # angle = 0
    R = np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    
    K = R*[[1./3.], [2./3.]]
    K_minus = R*[[-1./3.], [-2./3.]]
    M = R*[[0.5], [0.5]]
    M_minus = R*[[-0.5], [-0.5]]
    Kp = R*[[2./3.], [1./3.]]
    Kp_minus = R*[[-2./3.], [-1./3.]]
    
    K_in, M_in, Kp_in,K_minus_in,M_minus_in,Kp_minus_in = _kpt_array_to_list_(K,M,Kp,K_minus,M_minus,Kp_minus)
    Γ = [0.0,0.0]

    kspace_dict = {'K': K_in,'M':M_in,'Kp':Kp_in,'-K': K_in,'-M':M_in,'-Kp':Kp_in,'Γ':Γ}
    return kspace_dict

def _kpt_array_to_list_(*args):
    
    new_args = []
    for item in args:
        new_item = np.reshape(item,2).tolist()[0]
        new_args.append(new_item)
    return new_args

class InputChecker():
    def __init__(self,**kwargs) :
        
        self.__dict__.update(kwargs)

    def check_argument_clean(self):
            
            self._check_lattice_clean_()
            self._check_numband_clean_()
            self._check_bool_clean_()
            
    def _check_lattice_clean_(self):
        
        if type(self.lattice)!=list:
            Logger.raiseException('Data type of lattice has to be list',exception=TypeError)
        
        if len(self.lattice)!=2:
            Logger.raiseException('Need to specify two lattice vectors',exception=ValueError)
    
    def _check_numband_clean_(self):
        
        if self.num_band != 5 and self.num_band != 8 and self.num_band != 2:
            Logger.raiseException('Number of bands has to be either 5 or 8',exception=ValueError)
            
    def _check_bool_clean_(self):
        
        args = [self.real_orbit_position,self.spin_is_degen,self.valley_is_degen,self.bdg_is_degen,self.orbital_symmetry]
        for item in args:
            if type(item)!= bool:
                item_str = _debug_(item)
                Logger.raiseException(item_str+' has to be Boolean typed',exception=TypeError)

    def _check_path_points_(self,path_points):

        point_list = ['K','M','Kp','-K','-M','-Kp','Γ']
        all_present = all(item in point_list for item in path_points)
        if not all_present:
            Logger.raiseException('Some points in path_points is not accepted. Only accept '+ str(point_list)+' as input.',exception = ValueError)

class Evaluator():

    def __init__(self,**kwargs):

        #self.kspace_numsteps passed from tbg_model
        self.kspace_dict = get_kspace_dict()
        self.kspace_path = None
        self.path_points = None
        self.evals = None
        self.eval_list = []
        self.eval_dict = {}
        self.k_vec = None
        self.k_dist = None
        self.k_node = None

        self.__dict__.update(kwargs)
    
    def set_kspace_path(self,path_points=['K','Γ','M','K']):

        path = []
        for point_str in path_points:
            point = self.kspace_dict[point_str]
            path.append(point)
        #store into class variable
        self.kspace_path = path
        self.path_points = path_points

    # only for the case spin and BdG degeneracy 
    def eval_base(self,model):
        
        #Use PythTB to evaluate
        k_vec,k_dist,k_node = model.k_path(self.kspace_path, self.kspace_numsteps,report=False)
        evals = model.solve_all(k_vec)
        #store into class variable
        self.evals = evals
        self.k_vec = k_vec
        self.k_dist = k_dist
        self.k_node = k_node

    # 2D model array, for the case of both pairing parameters and chemical potentials
    def eval_parameter_dict(self,model_dict,Δs,μs):

        for (Δ,μ),model in model_dict.items():
            
            #control for floating point error in dictionary key values
            Δ = round(Δ*1e7)/1e7
            μ = round(μ*1e7)/1e7

            k_vec,k_dist,k_node = model.k_path(self.kspace_path, self.kspace_numsteps,report=False)
            evals = model.solve_all(k_vec)
            self.eval_dict[(Δ,μ)] = evals

        # store into class variable
        self.k_vec = k_vec
        self.k_dist = k_dist
        self.k_node = k_node

    # only for the case spin and BdG degeneracy
    def eval_parameters(self,models,parameters):
        
        eval_list = []
        for model in models:
            k_vec,k_dist,k_node = model.k_path(self.kspace_path, self.kspace_numsteps,report=False)
            evals = model.solve_all(k_vec)
            eval_list.append(evals)
        eval_list = AWA(eval_list,axes=[parameters],axis_names=['parameters']) 
        # store into class variable
        self.eval_list = eval_list
        self.k_vec = k_vec
        self.k_dist = k_dist
        self.k_node = k_node

class Plotter():

    def __init__(self,evaluator=None,**kwargs):
        self.__dict__.update(kwargs)
        self.kspace_path = None
        self.path_points = None
        self.floatslider = None
        self.evals,self.eval_list,self.eval_dict,self.kspace_path,self.path_points,self.k_vec,self.k_dist,self.k_node = self.set_eval_values(evaluator)

        self.fig = None
        self.ax = None

    def set_eval_values(self,evaluator):
        if evaluator==None:
            eval_values = (None,None,None,None,None,None,None,None)
        else: 
            eval_values = (evaluator.evals,evaluator.eval_list,evaluator.eval_dict,evaluator.kspace_path,evaluator.path_points,\
                            evaluator.k_vec,evaluator.k_dist,evaluator.k_node)
        return eval_values

    def update(self,evaluator):

        self.evals = evaluator.evals
        self.eval_list = evaluator.eval_list
        self.eval_dict = evaluator.eval_dict 
        self.kspace_path = evaluator.kspace_path
        self.path_points = evaluator.path_points
        self.k_vec = evaluator.k_vec
        self.k_dist = evaluator.k_dist
        self.k_node = evaluator.k_node

    def plot_base(self):

        self.fig, self.ax = plt.subplots(figsize=[4,6])
        total_num_band = self.num_band * self.spin_degeneracy * self.bdg_degeneracy
        for i in range(0,total_num_band,1):
            self.ax.plot(self.k_dist,self.evals[i,:],color='blue')
        self.ax.set_xticks(self.k_node)
        self.ax.set_xticklabels(self.path_points)
        self.ax.set_xlim(self.k_node[0],self.k_node[-1])

    def plot_parameters(self,parameters,param_name='Parameter'):
        
        self.fig, self.ax = plt.subplots(figsize=[4,6])

        floatslider = self.construct_floatslider(parameters,param_name)

        def update(i = floatslider):
            self.ax.clear()
            evals = self.eval_list.cslice[i]
            total_num_band = self.num_band * self.spin_degeneracy * self.bdg_degeneracy
            for band_index in range(0,total_num_band,1):
                self.ax.plot(self.k_dist,evals[band_index,:],color='blue')
            self.ax.set_xticks(self.k_node)
            self.ax.set_xticklabels(self.path_points)
            self.ax.set_xlim(self.k_node[0],self.k_node[-1])
        
        interact(update,i = floatslider)

    def plot_parameter_dict(self,Δs,μs):

        self.fig, self.ax = plt.subplots(figsize=[4,6])

        Δ_floatslider = self.construct_floatslider(Δs,param_name='Δ')
        μ_floatslider = self.construct_floatslider(μs,param_name='μ')

        def update(Δ = Δ_floatslider, μ=μ_floatslider):
            self.ax.clear()
            evals=self.eval_dict[(Δ,μ)]
            total_num_band = self.num_band * self.spin_degeneracy * self.bdg_degeneracy
            for band_index in range(0,total_num_band,1):
                self.ax.plot(self.k_dist,evals[band_index,:],color='blue')
            self.ax.set_xticks(self.k_node)
            self.ax.set_xticklabels(self.path_points)
            self.ax.set_xlim(self.k_node[0],self.k_node[-1])
        
        interact(update,Δ = Δ_floatslider, μ = μ_floatslider)

    def get_fig_ax(self):

        fig = cp.deepcopy(self.fig)
        ax = cp.deepcopy(self.ax)
        return fig,ax

    def construct_floatslider(self,parameters,param_name='Parameter'):
        floatslider=widgets.FloatSlider(
            value=0,
            min=min(parameters),
            max=max(parameters),
            step=(max(parameters)-min(parameters))/(len(parameters)-1),
            description=param_name,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.4f'
        )
        return floatslider


    
from pythtb import *
from ipywidgets import *    #for slider when plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.
from matplotlib import cm
import numpy as np
import copy as cp
from common.baseclasses import AWA
from NearFieldOptics.Materials.material_types import * #For Logger, overkill
import time
import datetime

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
    
    Kpp = R*[[1./3.],[-1./3.]]
    K = R*[[1./3.], [2./3.]]
    K_minus = R*[[-1./3.], [-2./3.]]
    M = R*[[0.5], [0.5]]
    M_minus = R*[[-0.5], [-0.5]]
    Kp = R*[[2./3.], [1./3.]]
    Kp_minus = R*[[-2./3.], [-1./3.]]
    M_off = R*[[0.6], [0.5]]
    
    K_in, M_in, Kp_in, Kpp_in,K_minus_in,M_minus_in,Kp_minus_in,M_off_in = _kpt_array_to_list_(K,M,Kp,Kpp,K_minus,M_minus,Kp_minus,M_off)
    Γ = [0.0,0.0]

    kspace_dict = {'K': K_in,'M':M_in,'Kp':Kp_in,'Kpp': Kpp_in,'-K': K_in,'-M':M_in,'-Kp':Kp_in,'Γ':Γ,'M_off':M_off_in}
    return kspace_dict

def _kpt_array_to_list_(*args):
    
    new_args = []
    for item in args:
        new_item = np.reshape(item,2).tolist()[0]
        new_args.append(new_item)
    return new_args

def k_uniform_mesh_restricted(mesh_size,k_mesh_center,k_mesh_length):
    """Create a uniform mesh of k points in a restricted region in Brillouin zone. 

        Most of the code is adapted from the k_uniform_mesh method in the PythTB module.
        Currently only for 2D, but it can be extended to other dimensions.
        
        Args:
            mesh_size (2-tuple): the number of k points in the x and y directions
            k_mesh_center (2-tuple): the x,y coordiante of the center of the square mesh of k points (in the reduced coordinates in k-space). Only used if restrict_k_mesh=True.
            k_mesh_length (float): the side length of the square mesh of k points (in the reduced coordinates in k-space). Only used if restrict_k_mesh=True.
    
        Return:
            k_vec: a N by 2 array of k points, in dictionary order (for each x value, going through all y values in the square mesh). N is the total number of grid points.
    
        """
    dim1,dim2 = mesh_size
    x_center,y_center = k_mesh_center

    use_mesh=np.array(list(map(round,(dim1,dim2))),dtype=int)
    k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1]]
    # normalize the mesh
    norm=np.tile(np.array(use_mesh/k_mesh_length,dtype=float),use_mesh)
    norm=norm.reshape(use_mesh.tolist()+[2])
    norm=norm.transpose([2,0,1])
    k_vec=k_vec/norm
    # final reshape
    k_vec=k_vec.transpose([1,2,0]).reshape([use_mesh[0]*use_mesh[1],2])
    k_vec=k_vec+[x_center,y_center]-[k_mesh_length/2,k_mesh_length/2]
    return k_vec


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
        
        if self.num_band != 5 and self.num_band != 8 and self.num_band != 2 and self.num_band !=3:
            Logger.raiseException('Number of bands has to be either 2,3,5 or 8',exception=ValueError)
            
    def _check_bool_clean_(self):
        
        args = [self.real_orbit_position,self.spin_is_degen,self.valley_is_degen,self.bdg_is_degen,self.orbital_symmetry]
        for item in args:
            if type(item)!= bool:
                item_str = _debug_(item)
                Logger.raiseException(item_str+' has to be Boolean typed',exception=TypeError)

    def _check_path_points_(self,path_points):

        point_list = ['K','M','Kp','Kpp','-K','-M','-Kp','Γ','M_off']
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
        self.evecs = None
        self.cal_eig_vectors = None
        self.eval_list = []
        self.evec_list = []
        self.eval_dict = {}
        self.evec_dict = {}
        self.k_vec = None
        self.k_dist = None
        self.k_node = None
        self.eval_all_k = None
        self.k_mesh_dim = None
        self.restrict_k_mesh = None
        self.k_mesh_center = None
        self.k_mesh_length = None
        self.feature_mode = None

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
    def eval_base(self,model,time_stamp=False):
        
        start = time.time()

        if self.eval_all_k:
            dim1,dim2 = self.k_mesh_dim
            k_vec = model.k_uniform_mesh([dim1,dim2])
        elif self.restrict_k_mesh:
            k_vec = k_uniform_mesh_restricted(self.k_mesh_dim,self.k_mesh_center,self.k_mesh_length)
        else:
            k_vec,k_dist,k_node = model.k_path(self.kspace_path, self.kspace_numsteps,report=False)
            self.k_dist = k_dist
            self.k_node = k_node
        
        if self.cal_eig_vectors:
            (evals,evecs) = model.solve_all(k_vec, eig_vectors=True)
            self.evals = evals
            self.evecs = evecs
        else:
            evals = model.solve_all(k_vec)
            self.evals = evals
        self.k_vec = k_vec
        
        end = time.time()
        elapsed = end - start
        
        if time_stamp:
            print("Time elapsed:" + str(datetime.timedelta(seconds=elapsed)))
            ct = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
            print(" current time:", ct) 

        
    # 2D model array, for the case of both pairing parameters and chemical potentials
    def eval_parameter_dict(self,model_dict,Δs,μs,time_stamp=False):

        for (Δ,μ),model in model_dict.items():
            
            start = time.time()

            #control for floating point error in dictionary key values
            Δ = round(Δ*1e7)/1e7
            μ = round(μ*1e7)/1e7

            if self.eval_all_k:
                dim1,dim2 = self.k_mesh_dim
                k_vec = model.k_uniform_mesh([dim1,dim2])
            elif self.restrict_k_mesh:
                k_vec = k_uniform_mesh_restricted(self.k_mesh_dim,self.k_mesh_center,self.k_mesh_length)
            else:
                k_vec,k_dist,k_node = model.k_path(self.kspace_path, self.kspace_numsteps,report=False)
                self.k_dist = k_dist
                self.k_node = k_node
            
            if self.cal_eig_vectors:
                (evals,evecs) = model.solve_all(k_vec, eig_vectors=True)
                self.eval_dict[(Δ,μ)] = evals
                self.evec_dict[(Δ,μ)] = evecs
            else:
                evals = model.solve_all(k_vec)
                self.eval_dict[(Δ,μ)] = evals

            end = time.time()
            elapsed = end - start
            
            if time_stamp:
                print("\nElapsed time for (Δ,μ)=("+ str(Δ)+","+str(μ)+")\n\t" + str(datetime.timedelta(seconds=elapsed)))
                ct = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
                print(" current time:", ct) 


        self.k_vec = k_vec
        

    # only for the case spin and BdG degeneracy
    def eval_parameters(self,models,parameters,time_stamp=False):

        eval_list = []
        evec_list = []
        for parameter,model in zip(parameters,models):

            start = time.time()

            if self.eval_all_k:
                dim1,dim2 = self.k_mesh_dim
                k_vec = model.k_uniform_mesh([dim1,dim2])
            elif self.restrict_k_mesh:
                k_vec = k_uniform_mesh_restricted(self.k_mesh_dim,self.k_mesh_center,self.k_mesh_length)
            else:
                k_vec,k_dist,k_node = model.k_path(self.kspace_path, self.kspace_numsteps,report=False)
                self.k_dist = k_dist
                self.k_node = k_node
            
            if self.cal_eig_vectors:
                (evals,evecs) = model.solve_all(k_vec, eig_vectors=True)
                eval_list.append(evals)
                evec_list.append(evecs)
            else:
                evals = model.solve_all(k_vec)
                eval_list.append(evals)

            end = time.time()
            elapsed = end - start
            
            if time_stamp:
                print("\nElapsed time for parameter="+ str(parameter)+"\n\t" + str(datetime.timedelta(seconds=elapsed)))
                ct = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
                print(" current time:", ct) 

        # store into class variable
        self.k_vec = k_vec
        eval_list = AWA(eval_list,axes=[parameters],axis_names=['parameters']) 
        self.eval_list = eval_list
        if self.cal_eig_vectors:
            evec_list = AWA(evec_list,axes=[parameters],axis_names=['parameters']) 
            self.evec_list = evec_list
        

class Plotter():

    def __init__(self,evaluator=None,**kwargs):
        self.__dict__.update(kwargs)
        
        self.evaluator = None
        self.kspace_path = None
        self.path_points = None
        self.floatslider = None
        self.fig = None
        self.ax = None
        self.plot_3D = None
        self.plot_contour = None
        self.zero_crossing_threshold = None
        self.feature_mode = None

    def update(self,evaluator):

        self.evaluator = evaluator

        self.evals = evaluator.evals
        self.eval_list = evaluator.eval_list
        self.eval_dict = evaluator.eval_dict 
        self.kspace_path = evaluator.kspace_path
        self.path_points = evaluator.path_points
        self.k_vec = evaluator.k_vec
        self.k_dist = evaluator.k_dist
        self.k_node = evaluator.k_node
        self.k_mesh_dim = evaluator.k_mesh_dim
        self.feature_mode = evaluator.feature_mode

    def plot_base(self,plot_3D=False,plot_contour=False,xlim=None,ylim=None,zlim=None):

        total_num_band = self.num_band * self.spin_degeneracy * self.valley_degeneracy * self.bdg_degeneracy

        if plot_3D:
            self.fig = plt.figure()
            self.ax = self.fig.gca(projection='3d')
            dim1,dim2 = self.evaluator.k_mesh_dim
            X = np.arange(0,dim1,1)
            Y = np.arange(0,dim2,1)
            X, Y = np.meshgrid(X, Y)
            K_pt_xs = np.array([33,67])
            K_pt_ys = np.array([67,33])
            K_pt_zs = np.array([0,0])
            for Z in self.evaluator.evals:
                Z = np.reshape(Z,(dim1,dim2))
                surf = self.ax.plot_surface(X, Y, Z, cmap=cm.viridis,linewidth=0, antialiased=False)
                self.ax.text(K_pt_xs[0], K_pt_ys[0]+10., K_pt_zs[0], 'K point', zdir=None,color='red')
                self.ax.text(K_pt_xs[1], K_pt_ys[1]+10., K_pt_zs[1], 'K\' point', zdir=None,color='red')
                self.ax.scatter(K_pt_xs,K_pt_ys,K_pt_zs,color='red',marker='o')
                self.ax.view_init(elev=5., azim=28.)
            if xlim != None:
                self.ax.set_xlim(xlim)
            if ylim != None:
                self.ax.set_ylim(ylim)
            if zlim != None:
                self.ax.set_zlim(zlim)

        # assume the number of bands (including degeneracy) is even
        # assume the the band closest to zero point is ordered 
        elif plot_contour:
            
            total_num_band = self.num_band * self.spin_degeneracy * self.valley_degeneracy * self.bdg_degeneracy
            top_band_index = int(total_num_band/2)
            bot_band_index = top_band_index-1
            eval_top_pre = self.evaluator.evals[top_band_index]
            eval_bot_pre = self.evaluator.evals[bot_band_index]
            dim1,dim2 = self.evaluator.k_mesh_dim
            eval_top = np.reshape(eval_top_pre,(dim1,dim2))
            eval_bot = np.reshape(eval_bot_pre,(dim1,dim2))

            over_cmap = cm.get_cmap('viridis')
            over_cmap.set_over('r')
            under_cmap = cm.get_cmap('viridis')
            under_cmap.set_under('r')

            self.fig,(ax1, ax2) = plt.subplots(ncols=2,figsize=(10,4))
            im1 = ax1.imshow(eval_bot,origin='lower',aspect='auto',vmax=-self.zero_crossing_threshold,cmap=over_cmap)
            im2 = ax2.imshow(eval_top,origin='lower',aspect='auto',vmin=self.zero_crossing_threshold,cmap=over_cmap)
            ax1.set_title('bottom')
            ax2.set_title('top')
            self.fig.colorbar(im1, ax=ax1)
            self.fig.colorbar(im2, ax=ax2)
            self.ax = [ax1,ax2]
            if xlim != None:
                self.ax.set_xlim(xlim)
            if ylim != None:
                self.ax.set_ylim(ylim)
            if zlim != None:
                self.ax.set_zlim(zlim)

        else:
            self.fig, self.ax = plt.subplots(figsize=[4,6])
            for eval in self.evaluator.evals:
                self.ax.plot(self.evaluator.k_dist,eval,color='blue')
            self.ax.set_xticks(self.evaluator.k_node)
            self.ax.set_xticklabels(self.evaluator.path_points)
            if xlim != None:
                self.ax.set_xlim(xlim)
            else:
                self.ax.set_xlim(self.evaluator.k_node[0],self.evaluator.k_node[-1])
            if ylim != None:
                self.ax.set_ylim(ylim)
            if zlim != None:
                self.ax.set_zlim(zlim)

    def plot_parameters(self,parameters,param_name='Parameter',plot_3D=False,plot_contour=False,xlim=None,ylim=None,zlim=None):
        
        floatslider = self.construct_floatslider(parameters,param_name)

        if plot_3D:
            self.fig = plt.figure()
            self.ax = self.fig.gca(projection='3d')
            dim1,dim2 = self.evaluator.k_mesh_dim
            X = np.arange(0,dim1,1)
            Y = np.arange(0,dim2,1)
            X, Y = np.meshgrid(X, Y)
            K_pt_xs = np.array([33,67])   # determined by eyeballing in actual plot
            K_pt_ys = np.array([67,33])   # determined by eyeballing in actual plot
            K_pt_zs = np.array([0,0])
            def update(i = floatslider):
                self.ax.clear()
                evals = self.evaluator.eval_list.cslice[i]
                for Z in evals:
                    Z = np.reshape(Z,(dim1,dim2))
                    surf = self.ax.plot_surface(X, Y, Z, cmap=cm.viridis,linewidth=0, antialiased=False)
                    self.ax.text(K_pt_xs[0], K_pt_ys[0]+10., K_pt_zs[0], 'K point', zdir=None,color='red')
                    self.ax.text(K_pt_xs[1], K_pt_ys[1]+10., K_pt_zs[1], 'K\' point', zdir=None,color='red')
                    self.ax.scatter(K_pt_xs,K_pt_ys,K_pt_zs,color='red',marker='o')
                    self.ax.view_init(elev=5., azim=28.)
                if xlim != None:
                    self.ax.set_xlim(xlim)
                if ylim != None:
                    self.ax.set_ylim(ylim)
                if zlim != None:
                    self.ax.set_zlim(zlim)
        
        # assume the number of bands (including degeneracy) is even
        # assume the the band closest to zero point is ordered 
        elif plot_contour:
            total_num_band = self.num_band * self.spin_degeneracy * self.valley_degeneracy * self.bdg_degeneracy
            top_band_index = int(total_num_band/2)
            bot_band_index = top_band_index-1
            evals = self.evaluator.eval_list.cslice[0]
            eval_top_pre = evals[top_band_index]
            eval_bot_pre = evals[bot_band_index]
            dim1,dim2 = self.evaluator.k_mesh_dim
            eval_top = np.reshape(eval_top_pre,(dim1,dim2))
            eval_bot = np.reshape(eval_bot_pre,(dim1,dim2))

            over_cmap = cm.get_cmap('viridis')
            over_cmap.set_over('r')
            under_cmap = cm.get_cmap('viridis')
            under_cmap.set_under('r')

            self.fig,(ax1, ax2) = plt.subplots(ncols=2,figsize=(10,4))
            im1 = ax1.imshow(eval_bot,origin='lower',aspect='auto',vmax=-self.zero_crossing_threshold,cmap=over_cmap)
            im2 = ax2.imshow(eval_top,origin='lower',aspect='auto',vmin=self.zero_crossing_threshold,cmap=over_cmap)
            ax1.set_title('bottom')
            ax2.set_title('top')
            self.fig.colorbar(im1, ax=ax1)
            self.fig.colorbar(im2, ax=ax2)
            self.ax = [ax1,ax2]

            def update(i = floatslider):
                evals = self.evaluator.eval_list.cslice[i]
                eval_top_pre = evals[top_band_index]
                eval_bot_pre = evals[bot_band_index]
                dim1,dim2 = self.evaluator.k_mesh_dim
                eval_top = np.reshape(eval_top_pre,(dim1,dim2))
                eval_bot = np.reshape(eval_bot_pre,(dim1,dim2))
                im1.set_data(eval_bot)
                im2.set_data(eval_top)
                if xlim != None:
                    self.ax.set_xlim(xlim)
                if ylim != None:
                    self.ax.set_ylim(ylim)
                if zlim != None:
                    self.ax.set_zlim(zlim)
                

        else:
            self.fig, self.ax = plt.subplots(figsize=[4,6])
            def update(i = floatslider):
                self.ax.clear()
                evals = self.evaluator.eval_list.cslice[i]
                total_num_band = self.num_band * self.spin_degeneracy * self.valley_degeneracy * self.bdg_degeneracy
                for eval in evals:
                    self.ax.plot(self.evaluator.k_dist,eval,color='blue')
                self.ax.set_xticks(self.evaluator.k_node)
                self.ax.set_xticklabels(self.evaluator.path_points)
                if xlim != None:
                    self.ax.set_xlim(xlim)
                else:
                    self.ax.set_xlim(self.evaluator.k_node[0],self.evaluator.k_node[-1])
                if ylim != None:
                    self.ax.set_ylim(ylim)
                if zlim != None:
                    self.ax.set_zlim(zlim)
        
        widgets.interact(update,i = floatslider)

    def plot_parameter_dict(self,Δs,μs,plot_3D=False,plot_contour=False,xlim=None,ylim=None,zlim=None):

        Δ_floatslider = self.construct_floatslider(Δs,param_name='Δ')
        μ_floatslider = self.construct_floatslider(μs,param_name='μ')

        if plot_3D:
            self.fig = plt.figure()
            self.ax = self.fig.gca(projection='3d')
            dim1,dim2 = self.evaluator.k_mesh_dim
            X = np.arange(0,dim1,1)
            Y = np.arange(0,dim2,1)
            X, Y = np.meshgrid(X, Y)
            K_pt_xs = np.array([33,67])
            K_pt_ys = np.array([67,33])
            K_pt_zs = np.array([0,0])
            def update(Δ = Δ_floatslider, μ=μ_floatslider):
                self.ax.clear()
                evals=self.evaluator.eval_dict[(Δ,μ)]
                for Z in evals:
                    Z = np.reshape(Z,(dim1,dim2))
                    surf = self.ax.plot_surface(X, Y, Z, cmap=cm.viridis,linewidth=0, antialiased=False)
                    # self.ax.text(K_pt_xs[0], K_pt_ys[0]+10., K_pt_zs[0], 'K point', zdir=None,color='red')
                    # self.ax.text(K_pt_xs[1], K_pt_ys[1]+10., K_pt_zs[1], 'K\' point', zdir=None,color='red')
                    # self.ax.scatter(K_pt_xs,K_pt_ys,K_pt_zs,color='red',marker='o')
                    # self.ax.view_init(elev=5., azim=28.)
                    self.ax.view_init(elev=0., azim=28.)

                if xlim != None:
                    self.ax.set_xlim(xlim)
                if ylim != None:
                    self.ax.set_ylim(ylim)
                if zlim != None:
                    self.ax.set_zlim(zlim)
        
        # assume the number of bands (including degeneracy) is even
        # assume the the band closest to zero point is ordered 
        elif plot_contour:
            
            total_num_band = self.num_band * self.spin_degeneracy * self.valley_degeneracy * self.bdg_degeneracy
            top_band_index = int(total_num_band/2)
            bot_band_index = top_band_index-1
            first_key = list(self.evaluator.eval_dict.keys())[0]
            evals = self.evaluator.eval_dict[first_key]
            eval_top_pre = evals[top_band_index]
            eval_bot_pre = evals[bot_band_index]
            dim1,dim2 = self.evaluator.k_mesh_dim
            eval_top = np.reshape(eval_top_pre,(dim1,dim2))
            eval_bot = np.reshape(eval_bot_pre,(dim1,dim2))

            bot_cmap = cp.copy(cm.get_cmap('viridis'))
            bot_cmap.set_over('r')
            bot_cmap.set_under('w')
            top_cmap = cp.copy(cm.get_cmap('viridis'))
            top_cmap.set_under('r')
            top_cmap.set_over('w')

            self.fig,(ax1, ax2) = plt.subplots(ncols=2,figsize=(10,4))
            im1 = ax1.imshow(eval_bot,origin='lower',aspect='auto',vmax=-self.zero_crossing_threshold,\
                            vmin=eval_bot.min(),cmap=bot_cmap)
            im2 = ax2.imshow(eval_top,origin='lower',aspect='auto',vmin=self.zero_crossing_threshold,\
                            vmax=eval_top.max(),cmap=top_cmap)
            ax1.set_title('bottom')
            ax2.set_title('top')
            self.fig.colorbar(im1, ax=ax1)
            self.fig.colorbar(im2, ax=ax2)
            self.ax = [ax1,ax2]

            def update(Δ = Δ_floatslider, μ=μ_floatslider):
                evals=self.evaluator.eval_dict[(Δ,μ)]
                eval_top_pre = evals[top_band_index]
                eval_bot_pre = evals[bot_band_index]
                dim1,dim2 = self.evaluator.k_mesh_dim
                eval_top = np.reshape(eval_top_pre,(dim1,dim2))
                eval_bot = np.reshape(eval_bot_pre,(dim1,dim2))

                im1.set_data(eval_bot)
                im2.set_data(eval_top)
                vmax = eval_bot.max()*(1+self.zero_crossing_threshold)
                im1.set_clim(vmin=eval_bot.min(),vmax=vmax)
                vmin = eval_top.min()*(1+self.zero_crossing_threshold)
                im2.set_clim(vmax=eval_top.max(),vmin=vmin)
                if xlim != None:
                    self.ax.set_xlim(xlim)
                if ylim != None:
                    self.ax.set_ylim(ylim)
                if zlim != None:
                    self.ax.set_zlim(zlim)


        else:
            self.fig, self.ax = plt.subplots(figsize=[4,6])
            def update(Δ = Δ_floatslider, μ=μ_floatslider):
                self.ax.clear()
                evals=self.evaluator.eval_dict[(Δ,μ)]
                total_num_band = self.num_band * self.spin_degeneracy * self.valley_degeneracy * self.bdg_degeneracy
                for eval in evals:
                    self.ax.plot(self.evaluator.k_dist,eval,color='blue')
                self.ax.set_xticks(self.evaluator.k_node)
                self.ax.set_xticklabels(self.evaluator.path_points)
                if xlim != None:
                    self.ax.set_xlim(xlim)
                else:
                    self.ax.set_xlim(self.evaluator.k_node[0],self.evaluator.k_node[-1])
                if ylim != None:
                    self.ax.set_ylim(ylim)
                if zlim != None:
                    self.ax.set_zlim(zlim)
        
        interact(update,Δ = Δ_floatslider, μ = μ_floatslider)

    def get_fig_ax(self):

        fig = self.fig
        ax = self.ax
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

    def plot_BZ_boundary(self,ax):
        # angle = np.pi
        angle = 0
        R = np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        factor,factor = self.evaluator.k_mesh_dim

        K = R*[[1./3.], [2./3.]]*factor
        K_minus = R*[[-1./3.], [-2./3.]]*factor
        M = R*[[0.5], [0.5]]*factor
        M_minus = R*[[-0.5], [-0.5]]*factor
        Kp = R*[[2./3.], [1./3.]]*factor
        Kp_minus = R*[[-2./3.], [-1./3.]]*factor
        Kpp = R*[[1./3.],[-1./3.]]*factor
        Kpp_minus = R*[[-1./3.],[1./3.]]*factor
        Γ = R*[[0],[0]]*factor

        pt_list = [K,Kp,Kpp,K_minus,Kp_minus,Kpp_minus,K]
        x,y = zip(*pt_list)
        x = np.reshape(np.array(list(x)),len(pt_list))
        y = np.reshape(np.array(list(y)),len(pt_list))

        y_sheered=y-x*np.sin(np.pi*30/180)
        x_sheered=x*np.cos(np.pi*30/180)
        
        ax.plot(x_sheered,y_sheered,color='white',linewidth=1)

    def process_evals(self,evals,N,d=4,plot_diff=False):
        total_num_band = self.num_band * self.spin_degeneracy * self.valley_degeneracy * self.bdg_degeneracy
        bot_band_index = int(total_num_band/2)-1
        eval_bot = evals[bot_band_index]
        eval_bot = np.reshape(eval_bot,(N,N))    
        
        if plot_diff:
            top_band_index = int(total_num_band/2)
            eval_top = evals[top_band_index]
            eval_top = np.reshape(eval_top,(N,N)) 
            eval_diff = np.abs(eval_top - eval_bot)
            eval_final = np.tile(eval_diff,(d,d))
        else:
            eval_final = np.tile(eval_bot,(d,d))
        return eval_final

    def generate_XY_mesh(self,N,d=4):
        # sheer to righten
        X = Y= np.linspace(-N*d/2,N*d/2,N*d)
        X_mid,Y_mid = np.meshgrid(X,Y)
        Y_mesh=Y_mid-X_mid*np.sin(np.pi*30/180)
        X_mesh=X_mid*np.cos(np.pi*30/180)
        return X_mesh,Y_mesh

    # generic plot contour method; for plotting the bottom band
    # atuomatically re-express in standard k coordinates and tile the Brillouin zone
    def contourf(self,evals,Δ,μ,plot_diff=False,vmax=-0.00005,vmin=-0.012,set_ticks=None,fontsize=20,save=False,\
        folder_name_add_string="",file_name_add_string=""):
    
        d = 4 # duplicate number (for tiling)
        N,N = self.evaluator.k_mesh_dim
        eval_final = self.process_evals(evals,N=N,d=d,plot_diff=plot_diff)
        X_mesh,Y_mesh = self.generate_XY_mesh(N=N,d=d)
        
        fig,ax1 = plt.subplots(figsize=[5,4])
        
        levels = np.linspace(vmin, vmax, 100)
        if plot_diff:
            over_cmap = cm.get_cmap('viridis_r')
        else:
            over_cmap = cm.get_cmap('viridis')
        over_cmap.set_over('r')
        # plot contour
        im1 = ax1.contourf(X_mesh,Y_mesh,eval_final,origin='lower',aspect='auto',levels=levels,cmap=over_cmap,vmin=vmin,vmax=vmax)
        #plot boundary of BZ
        self.plot_BZ_boundary(ax1)
        # plot nodes
        if plot_diff:
            ax1.contourf(X_mesh,Y_mesh,eval_final,origin='lower',aspect='auto',levels=[0.1*vmin,vmin],colors='r')
        else:
            ax1.contourf(X_mesh,Y_mesh,eval_final,origin='lower',aspect='auto',levels=[vmax,0.1*vmax],colors='r')

        ax1.set_title('Δ='+str(Δ)+',μ='+str(μ),loc='center')
        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])
        plt.xlabel('$k_x$', fontsize=fontsize)
        plt.ylabel('$k_y$', fontsize=fontsize,rotation=0,labelpad=20)
        cbar = fig.colorbar(im1, ax=ax1)
        cbar.ax.tick_params(labelsize=fontsize)

        if set_ticks != None:
            tick_min, tick_max, num_tick = set_ticks
            color_tick_list = np.linspace(tick_min,tick_max,num_tick)
            cbar.set_ticks(color_tick_list)

        factor = 0.75
        plt.xlim([-N*factor,N*factor])
        plt.ylim([-N*factor,N*factor])

        if save:
            import os
            if not os.path.exists('Data ('+self.feature_mode+')'+folder_name_add_string):
                os.makedirs('Data ('+self.feature_mode+')'+folder_name_add_string)
            plt.savefig('Data ('+self.feature_mode+')'+folder_name_add_string+'/Contour plot (Δ='+str(Δ)+',μ='+str(μ)+')'\
                +file_name_add_string+'.png',bbox_inches='tight',dpi=200)

        return fig,ax1

    
    """
    this is a wrapper method for the _plot3D_ private method. This method contains useful reusing parameters for TBG and MLG.
    style: for either TBG or MLG
    zticks (array): an array of zticks. If zticks==[], will use the default for the style. If zticks==None, will use the default of matplotlib.

    """
    def plot3D(self,evals,Δ,μ,style,N,plot_diff=False,show_title=True,elev=None,fontsize=20,save=False,\
        zbound=None,zticks=[],folder_name_add_string="",file_name_add_string=""):

        if elev==None:
            elev = 3

        if style=='tbg':
            unit_factor = 1000
            if zbound==None:
                zbound = [-0.01*unit_factor,0.01*unit_factor]
            if zticks==[]:
                zticks = [-0.01*unit_factor,-0.005*unit_factor,0,0.005*unit_factor,0.01*unit_factor]
            fig,ax = self._plot3D_(evals,Δ,μ,N,plot_diff=plot_diff,show_title=show_title,\
                unit_factor=unit_factor,zlabel="E (meV)",zbound=zbound,zticks=zticks,elev=elev,fontsize=fontsize)
        elif style=='mlg':
            if zticks==[]:
                zticks = [-3,0,3]
            fig,ax = self._plot3D_(evals,Δ,μ,N,plot_diff=plot_diff,show_title=show_title,\
                unit_factor=1,zlabel="E (a.u.)",zbound=zbound,zticks=zticks,elev=elev,fontsize=fontsize)
        else:
            Logger.raiseException('Invalid style.',exception=ValueError)

        if save:
            import os
            if not os.path.exists('Data ('+self.feature_mode+')'+folder_name_add_string):
                os.makedirs('Data ('+self.feature_mode+')'+folder_name_add_string)
            plt.savefig('Data ('+self.feature_mode+')'+folder_name_add_string+'/3D plot (Δ='+str(Δ)+',μ='+str(μ)+')'\
                +file_name_add_string+'.png',bbox_inches='tight',dpi=200)

        plt.show()

        return fig,ax




    def _plot3D_(self,evals,Δ,μ,N,plot_diff=False,show_title=True,zlabel="",unit_factor = 1000,zticks=None,zbound=None,elev=3,fontsize=20):
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        dim1,dim2 = N,N
        X = np.arange(0,dim1,1)
        Y = np.arange(0,dim2,1)
        X, Y = np.meshgrid(X, Y)

        if show_title:
            ax.set_title('Δ='+str(Δ)+',μ='+str(μ),loc='right')
        else:
            print('Δ='+str(Δ)+',μ='+str(μ))

        if plot_diff:
            i = int(len(evals)/2-1)
            j = int(len(evals)/2)
            diff = np.abs(evals[j] - evals[i])
            diff = np.reshape(diff,(dim1,dim2))
            surf = ax.plot_surface(X, Y, diff, cmap=cm.viridis,linewidth=0, antialiased=False)

            ref = np.zeros([dim1,dim2])
            ref_surf = ax.plot_surface(X, Y, ref, cmap=cm.viridis,linewidth=0, antialiased=False)
            ax.view_init(elev=-3, azim=59.)
        else: 
            for Z in evals:
                Z = np.reshape(Z,(dim1,dim2))
                surf = ax.plot_surface(X, Y, Z*unit_factor, cmap=cm.viridis,linewidth=0, antialiased=False)
            ax.view_init(elev=elev, azim=59.)
    #   
        if zbound!=None:
            ax.set_zbound(zbound)
        ax.set_xticks([])
        ax.set_yticks([])
        if zticks!=None:
            ax.set_zticks(zticks)
        for t in ax.zaxis.get_major_ticks(): 
            t.label.set_fontsize(fontsize)
        ax.set_zlabel(zlabel)

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        return fig,ax


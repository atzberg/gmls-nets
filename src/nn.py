"""
  .. image:: overview.png 

  PyTorch implementation of GMLS-Nets.  Module for neural networks for 
  processing scattered data sets using Generalized Moving Least Squares (GMLS).
 
  If you find these codes or methods helpful for your project, please cite:

  |   @article{trask_patel_gross_atzberger_GMLS_Nets_2019,
  |     title={GMLS-Nets: A framework for learning from unstructured data},
  |     author={Nathaniel Trask, Ravi G. Patel, Ben J. Gross, Paul J. Atzberger},
  |     journal={arXiv:1909.05371},  
  |     month={September},
  |     year={2019},
  |     url={https://arxiv.org/abs/1909.05371}
  |   }

"""

# Authors: B.J. Gross and P.J. Atzberger
# Website: http://atzberger.org/

import torch;
import torch.nn as nn;
import torchvision;
import torchvision.transforms as transforms;

import numpy as np;

import scipy.spatial as spatial # used for finding neighbors within distance $\delta$

from collections import OrderedDict;

import pickle as p;

import pdb;

import time;

# ====================================
# Custom Functions
# ====================================
class MapToPoly_Function(torch.autograd.Function):
  r"""
  This layer processes a collection of scattered data points consisting of a collection 
  of values :math:`u_j` at points :math:`x_j`.  For a collection of target points 
  :math:`x_i`, local least-squares problems are solved for obtaining a local representation 
  of the data over a polynomial space.  The layer outputs a collection of polynomial 
  coefficients :math:`c(x_i)` at each point and the collection of target points :math:`x_i`.
  """
  @staticmethod
  def weight_one_minus_r(z1,z2,params):  
    r"""Weight function :math:`\omega(x_j,x_i) = \left(1 - r/\epsilon\right)^{\bar{p}}_+.` 

        Args:
          z1 (Tensor): The first point.  Tensor of shape [1,num_dims].
          z2 (Tensor): The second point.    Tensor of shape [1,num_dims].
          params (dict): The parameters are 'p' for decay power and 'epsilon' for support size.

        Returns:
          Tensor: The weight evaluation over points.
     """
    epsilon = params['epsilon']; p = params['p'];
    r = torch.sqrt(torch.sum(torch.pow(z1 - z2,2),1));          
      
    diff    = torch.clamp(1 - (r/epsilon),min=0);
    eval    = torch.pow(diff,p);
    return eval;

  @staticmethod
  def get_num_polys(porder,num_dim=None):
    r""" Returns the number of polynomials of given porder. """
    if num_dim == 1:
      num_polys = porder + 1;
    elif num_dim == 2:
      num_polys = int((porder + 2)*(porder + 1)/2);
    elif num_dim == 3:
      num_polys = 0;
      for beta in range(0,porder + 1):
        num_polys += int((porder - beta + 2)*(porder - beta + 1)/2);
    else:
      raise Exception("Number of dimensions not implemented currently. \n  num_dim = %d."%num_dim);
  
    return num_polys;

  @staticmethod
  def eval_poly(pts_x,pts_x2_i0,c_star_i0,porder,flag_verbose):
    r""" Evaluates the polynomials locally around a target point xi given coefficients c. """
    # Evaluates the polynomial locally (this helps to assess the current fit).
    # Implemented for 1D, 2D, and 3D.  
    #
    # 2D:
    # Computes Taylor Polynomials over x and y.
    # T_{k1,k2}(x1,x2) = (1.0/(k1 + k2)!)*(x1 - x01)^{k1}*(x2 - x02)^{k2}.        
    # of terms is N = (porder + 1)*(porder + 2)/2.
    #
    # WARNING: Note the role of factorials and orthogonality here.  The Taylor 
    # expansion/polynomial formulation is not ideal and can give ill-conditioning.  
    # It would be better to use orthogonal polynomials or other bases.
    #
    num_dim = pts_x.shape[1];
    if num_dim == 1:
      II = 0;
      alpha_factorial = 1.0;
      eval_p = torch.zeros(pts_x.shape[0],device=c_star_i0.device);
      for alpha in np.arange(0,porder + 1):
        if alpha >= 2:
          alpha_factorial *= alpha;
        if flag_verbose > 1: print("alpha = " + str(alpha)); print("k = " + str(k));
        # for now, (x - x_*)^k, but ideally use orthogonal polynomials
        base_poly = torch.pow(pts_x[:,0] - pts_x2_i0[0],alpha);
        base_poly = base_poly/alpha_factorial;
        eval_p   += c_star_i0[II]*base_poly;
        II       += 1;
    elif num_dim == 2:
      II = 0;
      alpha_factorial = 1.0;
      eval_p = torch.zeros(pts_x.shape[0],device=c_star_i0.device);
      for alpha in np.arange(0,porder + 1):
        if alpha >= 2:
          alpha_factorial *= alpha;
        for k in np.arange(0,alpha + 1):
          if flag_verbose > 1: print("alpha = " + str(alpha)); print("k = " + str(k));
          # for now, (x - x_*)^k, but ideally use orthogonal polynomials
          base_poly = torch.pow(pts_x[:,0] - pts_x2_i0[0],alpha - k);
          # for now, (x - x_*)^k, but ideally use orthogonal polynomials
          base_poly = base_poly*torch.pow(pts_x[:,1] - pts_x2_i0[1],k);
          base_poly = base_poly/alpha_factorial;
          eval_p   += c_star_i0[II]*base_poly;
          II       += 1;
    elif num_dim == 3: # caution, below gives initial results, but should be more fully validated
      II = 0;
      alpha_factorial = 1.0;
      eval_p = torch.zeros(pts_x.shape[0],device=c_star_i0.device);
      for beta in np.arange(0,porder + 1):
        base_poly = torch.pow(pts_x[:,2] - pts_x2_i0[2],beta);
        for alpha in np.arange(0,porder - beta + 1):
          if alpha >= 2:
            alpha_factorial *= alpha;
          for k in np.arange(0,alpha + 1):
            if flag_verbose > 1: print("alpha = " + str(alpha)); print("k = " + str(k));
            # for now, (x - x_*)^k, but ideally use orthogonal polynomials
            base_poly = base_poly*torch.pow(pts_x[:,0] - pts_x2_i0[0],alpha - k);
            base_poly = base_poly*torch.pow(pts_x[:,1] - pts_x2_i0[1],k);
            base_poly = base_poly/alpha_factorial;
            eval_p   += c_star_i0[II]*base_poly;
            II       += 1;
    else:
      raise Exception("Number of dimensions not implemented currently. \n  num_dim = %d."%num_dim);

    return eval_p;

  @staticmethod
  def generate_mapping(weight_func,weight_func_params,
                       porder,epsilon,
                       pts_x1,pts_x2,
                       tree_points=None,device=None,
                       flag_verbose=0):
    r""" Generates for caching the data for the mapping from field values (uj,xj) :math:`\rightarrow` (ci,xi).  
    This help optimize codes and speed up later calculations that are done repeatedly."""
    if device is None:
      device = torch.device('cpu');
      
    map_data = {};
  
    num_dim = pts_x1.shape[1];
  
    if pts_x2 is None:
      pts_x2 = pts_x1;
	
    pts_x1 = pts_x1.to(device);
    pts_x2 = pts_x2.to(device);      

    pts_x1_numpy = None; pts_x2_numpy = None; 
    if tree_points is None: # build kd-tree of points for neighbor listing
      if pts_x1_numpy is None: pts_x1_numpy = pts_x1.cpu().numpy();
      tree_points = spatial.cKDTree(pts_x1_numpy);  

    # Maps from u(x_j) on $x_j \in \mathcal{S}^1$ to a 
    # polynomial representations in overlapping regions $\Omega_i$ at locations
    # around points $x_i \in \mathcal{S}^2$.
    # These two sample sets need not be the same allowing mappings between point locations.
    # Computes polynomials over x and y.
    # Number of terms in 2D is num_polys = (porder + 1)*(porder + 2)/2.
    num_pts1 = pts_x1.shape[0]; num_pts2 = pts_x2.shape[0];
    num_polys = MapToPoly_Function.get_num_polys(porder,num_dim);
    if flag_verbose > 0: 
      print("num_polys = " + str(num_polys));
      
    M = torch.zeros((num_pts2,num_polys,num_polys),device=device); # assemble matrix at each grid-point
    M_inv = torch.zeros((num_pts2,num_polys,num_polys),device=device); # assemble matrix at each grid-point
    
    #svd_U = torch.zeros((num_pts2,num_polys,num_polys)); # assemble matrix at each grid-point
    #svd_S = torch.zeros((num_pts2,num_polys,num_polys)); # assemble matrix at each grid-point
    #svd_V = torch.zeros((num_pts2,num_polys,num_polys)); # assemble matrix at each grid-point
    
    vec_rij = torch.zeros((num_pts2,num_polys,num_pts1),device=device); # @optimize: ideally should be sparse matrix.

    # build up the batch of linear systems for each target point
    for i in np.arange(0,num_pts2): # loop over the points $x_i$
        
      if (flag_verbose > 0) & (i % 100 == 0): print("i = " + str(i) + " : num_pts2 = " + str(num_pts2));

      if pts_x2_numpy is None: pts_x2_numpy = pts_x2.cpu().numpy();
      indices_xj_i = tree_points.query_ball_point(pts_x2_numpy[i,:], epsilon); # find all points with distance 
                                                                               # less than epsilon from xi.

      for j in indices_xj_i: # @optimize later to use only local points, and where weights are non-zero.
          
        if flag_verbose > 1: print("j = " + str(j));
        
        vec_p_j = torch.zeros(num_polys,device=device);
        w_ij    = weight_func(pts_x1[j,:].unsqueeze(0), pts_x2[i,:].unsqueeze(0), weight_func_params);  # can optimize for sub-lists outer-product          

        # Computes Taylor Polynomials over x,y,z.
        #
				# 2D Case:
        #   T_{k1,k2}(x1,x2) = (1.0/(k1 + k2)!)*(x1 - x01)^{k1}*(x2 - x02)^{k2}.
        #   number of terms is N = (porder + 1)*(porder + 2)/2.
        #   computes polynomials over x and y.
        #
        # WARNING: The monomial basis is non-ideal and can lead to ill-conditioned linear algebra.
        # This ultimately should be generalized in the future to other bases, ideally orthogonal, 
        # which would help both with efficiency and conditioning of the linear algebra.
        #
        if num_dim == 1:
          # number of terms is N = porder + 1.
          II = 0;
          for alpha in np.arange(0,porder + 1):
            if flag_verbose > 1: print("alpha = " + str(alpha)); print("k = " + str(k));
            # for now, (x - x_*)^k, but ideally use orthogonal polynomials
            vec_p_j[II] = torch.pow(pts_x1[j,0] - pts_x2[i,0], alpha);
            II += 1;
        elif num_dim == 2:
          # number of terms is N = (porder + 1)*(porder + 2)/2.
          II = 0;
          for alpha in np.arange(0,porder + 1):
            for k in np.arange(0,alpha + 1):
              if flag_verbose > 1: print("alpha = " + str(alpha)); print("k = " + str(k));
              # for now, (x - x_*)^k, but ideally use orthogonal polynomials
              vec_p_j[II] = torch.pow(pts_x1[j,0] - pts_x2[i,0], alpha - k);
              vec_p_j[II] = vec_p_j[II]*torch.pow(pts_x1[j,1] - pts_x2[i,1], k);
              II += 1;
        elif num_dim == 3:
          # number of terms is N = sum_{alpha_3 = 0}^porder [(porder - alpha_3+ 1)*(porder - alpha_3 + 2)/2.
          II = 0;
          for beta in np.arange(0,porder + 1):
            vec_p_j[II] = torch.pow(pts_x1[j,2] - pts_x2[i,2],beta);
            for alpha in np.arange(0,porder - beta + 1):
              for k in np.arange(0,alpha + 1):
                if flag_verbose > 1:
                  print("beta = " + str(beta)); print("alpha = " + str(alpha)); print("k = " + str(k));
                # for now, (x - x_*)^k, but ideally use orthogonal polynomials
                vec_p_j[II] = vec_p_j[II]*torch.pow(pts_x1[j,0] - pts_x2[i,0],alpha - k);
                vec_p_j[II] = vec_p_j[II]*torch.pow(pts_x1[j,1] - pts_x2[i,1],k);
                II += 1;
                          
        # add contributions to the M(x_i) and r(x_i) terms
        # r += (w_ij*u[j])*vec_p_j;
        vec_rij[i,:,j] = w_ij*vec_p_j;
        M[i,:,:]      += torch.ger(vec_p_j,vec_p_j)*w_ij;  # outer-product of vectors (build match of matrices)

      # Compute the SVD of M for purposes of computing the pseudo-inverse (for solving least-squares problem).
      # Note: M is always symmetric positive semi-definite, so U and V should be transposes of each other
      # and sigma^2 are the eigenvalues squared.  This simplifies some expressions.
      
      U,S,V = torch.svd(M[i,:,:]); # M = U*SS*V^T, note SS = diag(S)
      threshold_nonzero = 1e-9; # threshold for the largest singular value to consider being non-zero.
      I_nonzero         = (S > threshold_nonzero);
      S_inv             = 0.0*S;
      S_inv[I_nonzero]  = 1.0/S[I_nonzero];
      SS_inv            = torch.diag(S_inv);
      M_inv[i,:,:]      = torch.matmul(V,torch.matmul(SS_inv,U.t())); # pseudo-inverse of M^{-1} = V*S^{-1}*U^T

    # Save the linear system information for the least-squares problem at each target point $xi$.
    map_data['M'] = M;
    map_data['M_inv'] = M_inv;
    map_data['vec_rij'] = vec_rij;
    
    return map_data;   

  @staticmethod
  def get_poly_1D_u(u, porder, weight_func, weight_func_params,
                    pts_x1, epsilon = None, pts_x2 = None, cached_data=None,
                    tree_points = None, device=None, flag_verbose = 0):
      r""" Compute the polynomial coefficients in the case of a scalar field.  Would not typically call directly, used for internal purposes. """

      # We assume that all inputs are pytorch tensors
      # Assumes:
      #  pts_x1.size = [num_pts,num_dim] 
      #  pts_x2.size = [num_pts,num_dim]
      # 
      # @optimize: Should cache the points and neighbor lists... then using torch.solve, torch.ger.
      #            Should vectorize all of the for-loop operations via Lambdifying polynomial evals.
      #            Should avoid numpy calculations, maybe cache numpy copy of data if needed to avoid .cpu() transfer calls.
      #            Use batching over points to do solves, then GPU parallizable and faster.
      #      
      if device is None:
        device = torch.device('cpu'); # default cpu device
  
      if (u.dim() > 1):
        print("u.dim = " + str(u.dim()));
        print("u.shape = " + str(u.shape));
        raise Exception("Assumes input with dimension == 1.");
      
      if (cached_data is None) or ('map_data' not in cached_data) or (cached_data['map_data'] is None):
        generate_mapping = MapToPoly_Function.generate_mapping;
        
        if pts_x2 is None:
          pts_x2 = pts_x1;
          
        map_data = generate_mapping(weight_func,weight_func_params,
                                    porder,epsilon,
                                    pts_x1,pts_x2,tree_points,device);
        
        if cached_data is not None:
          cached_data['map_data'] = map_data;
          
      else:
        map_data = cached_data['map_data']; # use cached data
      
      if flag_verbose > 0: 
        print("num_pts1 = " + str(num_pts1) + ", num_pts2 = " + str(num_pts2));

      if epsilon is None:
        raise Exception('The epsilon ball size to use around xi must be specified.')
                
      # Maps from u(x_j) on $x_j \in \mathcal{S}^1$ to a 
      # polynomial representations in overlapping regions $\Omega_i$ at locations
      # around points $x_i \in \mathcal{S}^2$.
      
      # These two sample sets need not be the same allowing mappings between point sets.
      # Computes polynomials over x and y.
      # For 2D case, number of terms is num_polys = (porder + 1)*(porder + 2)/2.

      #c_star[:,i] = np.linalg.solve(np_M,np_r); # "c^*(x_i) = M^{-1}*r."
      vec_rij = map_data['vec_rij'];
      M_inv   = map_data['M_inv'];

      r_all   = torch.matmul(vec_rij,u);
      c_star  = torch.bmm(M_inv,r_all.unsqueeze(2)); # perform batch matric-vector multiplications
      c_star  = c_star.squeeze(2); # convert to list of vectors
      
      output  = c_star;
      output  = output.float(); # Map to float type for GPU / PyTorch Module compatibilities.
      
      return output, pts_x2;

  @staticmethod
  def forward(ctx, input, porder, weight_func, weight_func_params,
              pts_x1, epsilon = None, pts_x2 = None, cached_data=None,
              tree_points = None, device = None, flag_verbose = 0):
      r"""

      For a field u specified at points xj, performs the mapping to coefficients c at points xi, (uj,xj) :math:`\rightarrow` (ci,xi).

      Args:
        input (Tensor): The input field data uj.
        porder (int): Order of the basis to use (polynomial degree).
        weight_func (function): Weight function to use.
        weight_func_params (dict): Weight function parameters.
        pts_x1 (Tensor): The collection of domain points :math:`x_j`.
        epsilon (float): The :math:`\epsilon`-neighborhood size to use to sort points (should be compatible with choice of weight_func_params).
        pts_x2 (Tensor): The collection of target points :math:`x_i`.
        cache_data (dict): Stored data to help speed up repeated calculations.
        tree_points (dict): Stored data to help speed up repeated calculations.
        device (torch.device): Device on which to perform calculations (GPU or other, default is CPU).
        flag_verbose (int): Level of reporting on progress during the calculations.
        
      Returns: 
        tuple of (ci,xi): The coefficient values ci at the target points xi.  The target points xi. 
      
      """
      if device is None:
        device = torch.device('cpu');

      ctx.atz_name = 'MapToPoly_Function';

      ctx.save_for_backward(input,pts_x1,pts_x2);

      ctx.atz_porder = porder;
      ctx.atz_weight_func = weight_func;
      ctx.atz_weight_func_params = weight_func_params;

      get_poly_1D_u = MapToPoly_Function.get_poly_1D_u;
      get_num_polys = MapToPoly_Function.get_num_polys;
      
      input_dim = input.dim();
      if input_dim >= 1: # compute c_star in batches
          pts_x1_numpy = None;
          pts_x2_numpy = None; 
          
          if pts_x2 is None:
            pts_x2 = pts_x1;
          
          # reshape the data to handle as a batch [batch_size, uj_data_size]
          # We assume u is input in the form [I,k,xj], u_I(k,xj), the index I is arbitrary.
          u         = input;
          
          if input_dim == 2: # need to unsqueeze, so 2D we are mapping
                             # [k,xj] --> [I,k,xj] --> [II,c] --> [I,k,xi,c] --> [k,xi,c] 
            u = u.unsqueeze(0); # u(k,xj) assumed in our calculations here
          
          if input_dim == 1: # need to unsqueeze, so 1D we are mapping
                             # [xj] --> [I,k,xj] --> [II,c] --> [I,k,xi,c] --> [xi,c] 
            u = u.unsqueeze(0); # u(k,xj) assumed in our calculations here
            u = u.unsqueeze(0); # u(k,xj) assumed in our calculations here      
            
          u_num_dim = u.dim();            
          size_nm1  = 1;
          for d in range(u_num_dim - 1): 
            size_nm1 *= u.shape[d];

          uu = u.contiguous().view((size_nm1,u.shape[-1]));

          # compute the sizes of c_star and number of points
          num_dim   = pts_x1.shape[1];
          num_polys = get_num_polys(porder,num_dim);
          num_pts2  = pts_x2.shape[0];

          # output needs to be of size [batch_size, xi_data_size, num_polys]
          output = torch.zeros((uu.shape[0],num_pts2,num_polys),device=device); # will reshape at the end

          # loop over the batches and compute the c_star in each case
          if cached_data is None:
            cached_data = {}; # create empty, which can be computed first time to store data.
            
          if tree_points is None:
            if pts_x1_numpy is None: pts_x1_numpy = pts_x1.cpu().numpy();
            tree_points = spatial.cKDTree(pts_x1_numpy);
                              
          for k in range(uu.shape[0]):
            uuu = uu[k,:];
            out, pts_x2 = get_poly_1D_u(uuu,porder,weight_func,weight_func_params,
                                        pts_x1,epsilon,pts_x2,cached_data,
                                        tree_points,flag_verbose);
            output[k,:,:] = out;

          # final output should be [*, xi_data_size, num_polys], where * is the original sizes
          # for indices [i1,i2,...in,k_channel,xi_data,c_poly_coeff].
          output = output.view(*u.shape[0:u_num_dim-1],num_pts2,num_polys);

          if input_dim == 2: # 2D special case we just return k, xi, c (otherwise feed input 3D [I,k,u(xj)] I=1,k=1).
            output = output.squeeze(0);
          
          if input_dim == 1: # 1D special case we just return xi, c (otherwise feed input 3D [I,k,u(xj)] I=1,k=1).
            output = output.squeeze(0);
            output = output.squeeze(0);          
                          
      else:
        print("input.dim = " + str(input.dim()));
        print("input.shape = " + str(input.shape));
        raise Exception("input tensor dimension not yet supported, only dim = 1 and dim = 3 currently.");

      ctx.atz_cached_data = cached_data;
      
      pts_x2_clone = pts_x2.clone();
      return output, pts_x2_clone;
      
  @staticmethod
  def backward(ctx,grad_output,grad_pts_x2):
      r""" Consider a field u specified at points xj and the mapping to coefficients c at points xi, (uj,xj) --> (ci,xi).
      Computes the gradient of the mapping for backward propagation.
      """
  
      flag_time_it = False;
      if flag_time_it:
        time_1 = time.time();
              
      input,pts_x1,pts_x2 = ctx.saved_tensors;
      
      porder = ctx.atz_porder;
      weight_func = ctx.atz_weight_func;
      weight_func_params = ctx.atz_weight_func_params;      
      cached_data = ctx.atz_cached_data;

      #grad_input = grad_weight_func = grad_weight_func_params = None;
      grad_uj = None;

      # we only compute the gradient in x_i, if it is requested (for efficiency)
      if ctx.needs_input_grad[0]: # derivative in uj
        map_data = cached_data['map_data']; # use cached data
        
        vec_rij = map_data['vec_rij'];
        M_inv   = map_data['M_inv'];

        # c_i = M_{i}^{-1} r_i^T u
        # dF/du = dF/dc*dc/du,
        #
        # We can express this using dF/uj = sum_i dF/dci*dci/duj
        #
        # grad_output = dF/dc,   grad_input = dF/du
        #
        # [grad_input]_j = sum_i dF/ci*dci/duj.
        #
        # In practice, we have both batch and channel indices so
        # grad_output.shape = [batchI,channelI,i,compK]
        # grad_output[batchI,channelI,i,compK] = F(batchI,channelI) with respect to ci[compK](batchI,channelI).
        #
        # grad_input[batchI,channelI,j] =
        #
        # We use matrix broadcasting to get this outcome in practice.
        #

        # @optimize can optimize, since uj only contributes non-zero to a few ci's... and could try to use sparse matrix multiplications.
        A1 = torch.bmm(M_inv,vec_rij); # dci/du, grad = grad[i,compK,j]
        A2 = A1.unsqueeze(0).unsqueeze(0); # match grad_output tensor rank, for grad[batchI,channelI,i,compK,j]
        A3 = grad_output.unsqueeze(4); # grad_output[batchI,channelI,i,compK,j]
        A4 = A3*A2; # elementwise multiplication
        A5 = torch.sum(A4,3); # contract on index compK
        A6 = torch.sum(A5,2); # contract on index i

        grad_uj = A6;
                
      else:
        msg_str = "Requested a currently un-implemented gradient for this map: \n";
        msg_str += "ctx.needs_input_grad = \n" + str(ctx.needs_input_grad); 
        raise Exception(msg_str);

      if flag_time_it:
        msg = 'MapToPoly_Function->backward():';
        msg += 'elapsed_time = %.4e'%(time.time() - time_1);
        print(msg);
    
      return grad_uj,None,None,None,None,None,None,None,None,None,None; # since no trainable parts for these components of map


class MaxPoolOverPoints_Function(torch.autograd.Function):
  r"""Applies a max-pooling operation to obtain values :math:`v_i = \max_{j \in \mathcal{N}_i(\epsilon)} \{u_j\}.` """  
  # @optimize: Should cache the points and neighbor lists.
  #            Should avoid numpy calculations, maybe cache numpy copy of data if needed to avoid .cpu() transfer calls.
  #            Use batching over points to do solves, then GPU parallizable and faster.
  @staticmethod
  def forward(ctx,input,pts_x1,epsilon=None,pts_x2=None,
              indices_xj_i_cache=None,tree_points=None,
              flag_verbose=0):
      r"""Compute max pool operation from values at points (uj,xj) to obtain (vi,xi). 

          Args:
            input (Tensor): The uj values at the location of points xj.
            pts_x1 (Tensor): The collection of domain points :math:`x_j`.
            epsilon (float): The :math:`\epsilon`-neighborhood size to use to sort points (should be compatible with choice of weight_func_params).
            pts_x2 (Tensor): The collection of target points :math:`x_i`.
            tree_points (dict): Stored data to help speed up repeated calculations.
            flag_verbose (int): Level of reporting on progress during the calculations.
            
          Returns:
            tuple: The collection ui at target points (same size as uj in the non-j indices).  The collection xi of target points. Tuple of form (ui,xi).
            
          Note:
            We assume that all inputs are pytorch tensors with pts_x1.shape = [num_pts,num_dim] and similarly for pts_x2.

      """

      ctx.atz_name = 'MaxPoolOverPoints_Function';
      
      ctx.save_for_backward(input,pts_x1,pts_x2);
              
      u = input.clone(); # map input values u(xj) at xj to max value in epsilon neighborhood to u(xi) at xi points.

    	# Assumes that input is of size [k1,k2,...,kn,j], where k1,...,kn are any indices.
    	# We perform maxing over batch over all non-indices in j.
      # We reshape tensor to the form [*,j] where one index in *=index(k1,...,kn).     
      u_num_dim = u.dim();            
      size_nm1  = 1;
      for d in range(u_num_dim - 1): 
        size_nm1 *= u.shape[d];

      uj = u.contiguous().view((size_nm1,u.shape[-1]));  # reshape so indices --> [I,j], I = index(k1,...,kn).

      # reshaped        
      if pts_x2 is None:
        pts_x2 = pts_x1;

      pts_x1_numpy = pts_x1.cpu().numpy(); pts_x2_numpy = pts_x2.cpu().numpy(); # move to cpu to get numpy data
      pts_x1 = pts_x1.to(input.device); pts_x2 = pts_x2.to(input.device); # push back to GPU [@optimize later]        
      num_pts1 = pts_x1.size()[0]; num_pts2 = pts_x2.size()[0];
      if flag_verbose > 0: 
        print("num_pts1 = " + str(num_pts1) + ", num_pts2 = " + str(num_pts2));

      if epsilon is None:
        raise Exception('The epsilon ball size to use around xi must be specified.');
      
      ctx.atz_epsilon = epsilon;

      if indices_xj_i_cache is None:
        flag_need_indices_xj_i = True;
      else:
        flag_need_indices_xj_i = False;

      if flag_need_indices_xj_i and tree_points is None: # build kd-tree of points for neighbor listing            
        tree_points = spatial.cKDTree(pts_x1_numpy);
                  
      ctx.atz_tree_points = tree_points;
      ctx.indices_xj_i_cache = indices_xj_i_cache;

      # Maps from u(x_j) on $x_j \in \mathcal{S}^1$ to a u(x_i) giving max values in epsilon neighborhoods.
      # @optimize by caching these data structure for re-use later
      ui = torch.zeros(size_nm1,num_pts2,requires_grad=False,device=input.device);
      ui_argmax_j = torch.zeros(size_nm1,num_pts2,dtype=torch.int64,requires_grad=False,device=input.device);
      # assumes array of form [*,num_pts2], will be reshaped to match uj, [*,num_pts2].
        
      for i in np.arange(0,num_pts2): # loop over the points $x_i$
        if flag_verbose > 1: print("i = " + str(i) + " : num_pts2 = " + str(num_pts2));

        # find all points distance epsilon from xi
        if flag_need_indices_xj_i:
          indices_xj_i = tree_points.query_ball_point(pts_x2_numpy[i,:], epsilon);
          indices_xj_i = torch.Tensor(indices_xj_i).long();
          indices_xj_i.to(uj.device);
        else:
          indices_xj_i = indices_xj_i_cache[i,:];  # @optimize should consider replacing with better data structures

        # take max over neighborhood. Assumes for now that ui is scalar.
        uuj      = uj[:,indices_xj_i]; 
        qq       = torch.max(uuj,dim=-1,keepdim=True);
        ui[:,i]  = qq[0].squeeze(-1); # store max value
        jj       = qq[1].squeeze(-1); # store index of max value
        ui_argmax_j[:,i] = indices_xj_i[jj]; # store global index of the max value
                  
      # reshape the tensor from ui[I,i] to the form uui[k1,k2,...kn,i]
      uui = ui.view(*u.shape[0:u_num_dim-1],num_pts2);
      uui_argmax_j = ui_argmax_j.view(*u.shape[0:u_num_dim-1],num_pts2);
      ctx.atz_uui_argmax_j = uui_argmax_j; # save for gradient calculation

      output = uui; # for now, we assume for now that ui is scalar array of size [num_pts2]
      output = output.to(input.device);

      return output, pts_x2.clone();

  @staticmethod
  def backward(ctx,grad_output,grad_pts_x2):
      r"""Compute gradients of the max pool operations from values at points (uj,xj) --> (max_ui,xi). """

      flag_time_it = False;
      if flag_time_it:
        time_11 = time.time();
      
      # Compute df/dx from df/dy using the Chain Rule df/dx = df/dx*dy/dx.
      # Compute the gradient with respect to inputs, dz/dx.
      #
      # Consider z = f(g(x)), where we refer to x as the inputs and y = g(x) as outputs.
      # If we know dz/dy, we would like to compute dz/dx.  This will follow from the chain-rule
      # as dz/dx = (dz/dy)*(dy/dx).  We call dz/dy the gradient with respect to output and we call
      # dy/dx the gradient with respect to input.
      #
      # Note: the grad_output can be larger than the size of the input vector if we include in our 
      # definition of gradient_input the derivatives with respect to weights.  Should think of everything
      # input as tilde_x = [x,weights,bias,etc...], then grad_output = dz/dtilde_x.
              
      input,pts_x1,pts_x2 = ctx.saved_tensors;
      
      uui_argmax_j = ctx.atz_uui_argmax_j;

      #grad_input = grad_weight_func = grad_weight_func_params = None;
      grad_input = None;

      # We only compute the gradient in xi, if it is requested (for efficiency)
      # stubs for later possible use, but not needed for now
      if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
        msg_str = "Currently requested a non-trainable gradient for this map: \n";
        msg_str += "ctx.needs_input_grad = \n" + str(ctx.needs_input_grad); 
        raise Exception(msg_str);

      if ctx.needs_input_grad[0]:            
        # Compute dL/duj = (dL/dvi)*(dvi/duj), here vi = uui.
        # For the max-pool case, notice that dvi/duj is non-zero only when the index uj 
        # was the maximum value in the neighborhood of vi.  Notice subtle issue with
        # right and left derivatives being different, so max is not differentiable for ties.
        # We use the right derivative lim_h (q(x + h) - q(x))/h, here.

        # We assume that uj.size = [k1,k2,...,kn,j], ui.size = [k1,k2,...,kn,i].
        # These are reshaped so that uuj.size = [I,j] and uui.size = [I,i].
        input_dim    = input.dim();
        size_uj      = input.size();
        size_uj_nm1  = np.prod(size_uj[0:input_dim-1]); # exclude last index size
        
        #ss_grad_input  = input.new_zeros(size_uj_nm1,size_uj[-1]); # to store dL/duj, [I,j] indexing.
        ss_grad_output = grad_output.contiguous().view((size_uj_nm1,grad_output.shape[-1])); # reshape so index [I,i].
        ss_uui_argmax_j = uui_argmax_j.contiguous().view((size_uj_nm1,grad_output.shape[-1])); # reshape so index [I,i].

        # assign the entries k_i = argmax_{j in Omega_i} uj, reshaped so [*,j] = val[*,j].
        flag_method = 'method1';
        if flag_method == 'method1':

          flag_time_it = False;
          if flag_time_it:
            time_0 = time.time();

          I = torch.arange(0,size_uj_nm1,dtype=torch.int64,device=input.device);
          vec_ones = torch.ones(grad_output.shape[-1],dtype=torch.int64,device=input.device);
          II = torch.ger(I.float(),vec_ones.float()); # careful int --> float conv
          II = II.flatten();
          JJ = ss_uui_argmax_j.flatten();
          IJ_indices1 = torch.stack((II,JJ.float())).long();
                      
          i_index  = torch.arange(0,grad_output.shape[-1],dtype=torch.int64,device=input.device);
          vec_ones = torch.ones(size_uj_nm1,dtype=torch.int64,device=input.device);
          KK       = torch.ger(vec_ones.float(),i_index.float()); # careful int --> float conv
          KK       = KK.flatten();
          IJ_indices2 = torch.stack((II,KK)).long();

          # We aim to compute dL/duj = dL/d\bar{u}_i*d\bar{u}_i/duj.
          #
          # This is done efficiently by constructing a sparse matrix using how \bar{u}_i
          # depends on the uj.  Sometimes the same uj contributes multiple times to
          # a given \bar{u}_i entry, so we add together those contributions, as would
          # occur in an explicit multiplication of the terms above for dL/duj.
          # This is acheived efficiently using the .add() for sparse tensors in PyTorch.
          
          # We construct entries of the sparse matrix and coelesce them (add repeats).
          vals = ss_grad_output[IJ_indices2[0,:],IJ_indices2[1,:]]; # @optimize, maybe just flatten
          N1 = size_uj_nm1; N2 = size_uj[-1]; sz = torch.Size([N1,N2]);
          ss_grad_input = torch.sparse.FloatTensor(IJ_indices1,vals,sz).coalesce().to_dense();

          if flag_time_it:
            time_1 = time.time();
            
            print("time: backward(): compute ss_grad_input = %.4e sec"%(time_1 - time_0));            

        elif flag_method == 'method2':
          II      = torch.arange(0,size_uj_nm1,dtype=torch.int64);
          i_index = torch.arange(0,grad_output.shape[-1],dtype=torch.int64);
        
          # @optimize by vectorizing this calculation
          for I in II:
            for j in range(0,i_index.shape[0]):
              ss_grad_input[I,ss_uui_argmax_j[I,j]] += ss_grad_output[I,i_index[j]];
            
        else:
          raise Exception("flag_method type not recognized.\n flag_method = %s"%flag_method);    
          
        # reshape
        grad_input = ss_grad_input.view(*size_uj[0:input_dim - 1],size_uj[-1]);

      if flag_time_it:
        msg = 'atzGMLS_MaxPool2D_Function->backward(): ';
        msg += 'elapsed_time = %.4e'%(time.time() - time_11);
        print(msg);  
      
      return grad_input,None,None,None,None,None,None; # since no trainable parts for components of this map

class ExtractFromTuple_Function(torch.autograd.Function):
  r"""Extracts from a tuple of outputs one of the components."""
    
  @staticmethod
  def forward(ctx,input,index):
    r"""Extracts tuple entry with the specified index."""
    ctx.atz_name = 'ExtractFromTuple_Function';
    
    extracted = input[index];
    output = extracted.clone(); # clone added for safety
    
    return output;
    
  @staticmethod
  def backward(ctx,grad_output): # number grad's needs to match outputs of forward
    r"""Computes gradient of the extraction."""

    raise Exception('This backward is not implemented, since PyTorch automatically handled this in the past.');
    return None,None;
        
# ====================================
# Custom Modules
# ====================================
class PdbSetTraceLayer(nn.Module):
  r"""Allows for placing break-points within the call sequence of layers using pdb.set_trace().  Helpful for debugging networks."""

  def __init__(self):
    r"""Initialization (currently nothing to do, but call super-class)."""
    super(PdbSetTraceLayer, self).__init__()
            
  def forward(self, input):            
    r"""Executes a PDB breakpoint inside of a running network to help with debugging."""
    out = input.clone();  # added clone to avoid .grad_fn overwrite
    pdb.set_trace();
    return out;

class ExtractFromTuple(nn.Module): 
  r"""Extracts from a tuple of outputs one of the components."""
     
  def __init__(self,index=0):
    r"""Initializes the index to extract."""
    super(ExtractFromTuple, self).__init__()
    self.index     = index;  

  def forward(self, input):
    r"""Extracts the tuple entry with the specified index."""
    extracted = input[self.index];
    extracted_clone = extracted.clone(); # cloned to avoid overwrite of .grad_fn
    return extracted_clone;

class ReshapeLayer(nn.Module):
  r"""Performs reshaping of a tensor output within a network."""
 
  def __init__(self,reshape,permute=None):
    r"""Initializes the reshaping form to use followed by the indexing permulation to apply."""
    super(ReshapeLayer, self).__init__()
    self.reshape = reshape; 
    self.permute = permute;
    
  def forward(self, input):        
    r"""Reshapes the tensor followed by applying a permutation to the indexing."""
    reshape = self.reshape;
    permute = self.permute;  
    A = input.contiguous();
    out = A.view(*reshape); 
    if permute is not None:
      out = out.permute(*permute);
    return out;

class PermuteLayer(nn.Module):
  r"""Performs permutation of indices of a tensor output within a network."""
   
  def __init__(self,permute=None):
    r"""Initializes the indexing permuation to apply to tensors."""
    super(PermuteLayer, self).__init__()
    self.permute = permute;
    
  def forward(self, input):            
    r"""Applies and indexing permuation to the input tensor."""
    permute = self.permute;
    input_clone = input.clone(); # adding clone to avoid .grad_fn overwrites
    out = input_clone.permute(*permute);
    return out;

class MLP_Pointwise(nn.Module):
  r"""Creates a collection of multilayer perceptrons (MLPs) for each output channel.
  The MLPs are then applied at each target point xi.
  """

  def create_mlp_unit(self,layer_sizes,unit_name='',flag_bias=True):
    r"""Creates an instance of an MLP with specified layer sizes. """
    layer_dict = OrderedDict();
    NN = len(layer_sizes);
    for i in range(NN - 1):
      key_str = unit_name + ':hidden_layer_%.4d'%(i + 1);
      layer_dict[key_str] = nn.Linear(layer_sizes[i], layer_sizes[i+1],bias=flag_bias);
      if i < NN - 2: # last layer should be linear
        key_str = unit_name + ':relu_%.4d'%(i + 1);
        layer_dict[key_str] = nn.ReLU();
        
    mlp_unit = nn.Sequential(layer_dict); # uses ordered dictionary to create network

    return mlp_unit;
  
  def __init__(self,layer_sizes,channels_in=1,channels_out=1,flag_bias=True,flag_verbose=0):
    r"""Initializes the structure of the pointwise MLP module with layer sizes, number input channels, number of output channels.

        Args:
          layer_sizes (list): The number of hidden units in each layer.
          channels_in (int): The number of input channels.
          channels_out (int): The number of output channels.
          flag_bias (bool): If the MLP should include the additive bias b added into layers.
          flag_verbose (int): The level of messages generated on progress of the calculation.

    """
    super(MLP_Pointwise, self).__init__();

    self.layer_sizes   = layer_sizes; 
    self.flag_bias     = flag_bias;
    self.depth         = len(layer_sizes); 

    self.channels_in   = channels_in;
    self.channels_out  = channels_out;
        
    # create intermediate layers
    mlp_list = nn.ModuleList();
    layer_sizes_unit = layer_sizes.copy(); # we use inputs k*c to cross channels in practice in our unit MLPs
    layer_sizes_unit[0] = layer_sizes_unit[0]*channels_in; # modify the input to have proper size combined k*c
    for ell in range(channels_out):
      mlp_unit = self.create_mlp_unit(layer_sizes_unit,'unit_ell_%.4d'%ell,flag_bias=flag_bias);
      mlp_list.append(mlp_unit);

    self.mlp_list = mlp_list;  
      
  def forward(self, input, params = None): 
    r"""Applies the specified MLP pointwise to the collection of input data to produce pointwise entries of the output channels."""
    #
    # Assumes the tensor has the form [i1,i2,...in,k,c], the last two indices are the
    # channel index k, and the coefficient index c, combine for ease of use, but can reshape.
    # We collapse input tensor with indexing [i1,i2,...in,k,c] to a [I,k*c] tensor, where
    # I is general index, k is channel, and c are coefficient index.
    #
    s = input.shape;
    num_dim = input.dim();

    if (s[-2] != self.channels_in) or (s[-1] != self.layer_sizes[0]): # check correct sized inputs
      print("input.shape = " + str(input.shape));
      raise Exception("MLP assumes an input tensor of size [*,%d,%d]"%(self.channels_in,self.layer_sizes[0]));
    
    calc_size1 = 1.0;
    for d in range(num_dim-2):
      calc_size1 *= s[d];
    calc_size1 = int(calc_size1);
    
    x = input.contiguous().view(calc_size1,s[num_dim-2]*s[num_dim-1]); # shape input to have indexing [I,k*NN + c]

    if params is None:
      output = torch.zeros((self.channels_out,x.shape[0]),device=input.device); # shape [ell,*]
      for ell in range(self.channels_out):
        mlp_q = self.mlp_list[ell];  
        output[ell,:] = mlp_q.forward(x).squeeze(-1); # reduce from [N,1] to [N]

      s = input.shape;
      output = output.view(self.channels_out,*s[0:num_dim-2]); # shape to have index [ell,i1,i2,...,in]
      nn = output.dim();       
      p_ind = np.arange(nn) + 1;
      p_ind[nn-1] = 0;
      p_ind = tuple(p_ind);
      output = output.permute(p_ind); # [*,ell] indexing of final shape
    else:
      raise Exception("Not yet implemented for setting parameters.");

    return output; # [*,ell] indexing of final shape

  def to(self, device):
    r"""Moves data to GPU or other specified device."""
    super(MLP_Pointwise, self).to(device);
    for ell in range(self.channels_out):
      mlp_q = self.mlp_list[ell];
      mlp_q.to(device);      
    return self;


class MLP1(nn.Module):
  r"""Creates a multilayer perceptron (MLP). """

  def __init__(self, layer_sizes, flag_bias = True, flag_verbose=0):
    r"""Initializes MLP and specified layer sizes."""
    super(MLP1, self).__init__();

    self.layer_sizes   = layer_sizes; 
    self.flag_bias     = flag_bias;
    self.depth         = len(layer_sizes); 
    
    # create intermediate layers
    layer_dict = OrderedDict();
    NN = len(layer_sizes);
    for i in range(NN - 1):
      key_str = 'hidden_layer_%.4d'%(i + 1);
      layer_dict[key_str] = nn.Linear(layer_sizes[i], layer_sizes[i+1],bias=flag_bias);
      if i < NN - 2: # last layer should be linear
        key_str = 'relu_%.4d'%(i + 1);
        layer_dict[key_str] = nn.ReLU();
        
    self.layers = nn.Sequential(layer_dict); # uses ordered dictionary to create network
          
  def forward(self, input, params = None): 
    r"""Applies the MLP to the input data.
     
        Args:
          input (Tensor): The coefficient channel data organized as one stacked 
          vector of size Nc*M, where Nc is number of channels and M is number of 
          coefficients per channel.

        Returns:
          Tensor: The evaluation of the network.  Returns tensor of size [batch,1].

    """
    # evaluate network with specified layers
    if params is None:
      eval = self.layers.forward(input);
    else:
      raise Exception("Not yet implemented for setting parameters.");
      
    return eval;

  def to(self, device):
    r"""Moves data to GPU or other specified device."""
    super(MLP1, self).to(device);
    self.layers = self.layers.to(device);
    return self;

class MapToPoly(nn.Module):
  r"""
  This layer processes a collection of scattered data points consisting of a collection 
  of values :math:`u_j` at points :math:`x_j`.  For a collection of target points 
  :math:`x_i`, local least-squares problems are solved for obtaining a local representation 
  of the data over a polynomial space.  The layer outputs a collection of polynomial 
  coefficients :math:`c(x_i)` at each point and the collection of target points :math:`x_i`.
  """
  def __init__(self, porder, weight_func, weight_func_params, pts_x1, 
               epsilon = None,pts_x2 = None,tree_points = None,
               device = None,flag_verbose = 0,**extra_params):
      r"""Initializes the layer for mapping between field data uj at points xj to the 
          local polynomial reconstruction represented by coefficients ci at target points xi.

          Args:
            porder (int): Order of the basis to use.  For polynomial basis is the degree.
            weight_func (func): Weight function to use.
            weight_func_params (dict): Weight function parameters.
            pts_x1 (Tensor): The collection of domain points :math:`x_j`.
            epsilon (float): The :math:`\epsilon`-neighborhood size to use to sort points (should be compatible with choice of weight_func_params).
            pts_x2 (Tensor): The collection of target points :math:`x_i`.
            tree_points (dict): Stored data to help speed up repeated calculations.
            device: Device on which to perform calculations (GPU or other, default is CPU).
            flag_verbose (int): Level of reporting on progress during the calculations.
            **extra_params: Extra parameters allowing for specifying layer name and caching mode.            

      """
      super(MapToPoly, self).__init__();

      self.flag_verbose = flag_verbose;

      if device is None:
        device = torch.device('cpu');

      self.device = device;
      
      if 'name' in extra_params:
        self.name = extra_params['name'];
      else:
        self.name = "default_name";
  
      if 'flag_cache_mode' in extra_params:
        flag_cache_mode = extra_params['flag_cache_mode'];
      else:
        flag_cache_mode = 'generate1';
        
      if flag_cache_mode == 'generate1':  # setup from scratch
        self.porder = porder;
        self.weight_func = weight_func;
        self.weight_func_params = weight_func_params;

        self.pts_x1 = pts_x1;
        self.pts_x2 = pts_x2;

        self.pts_x1_numpy = None;
        self.pts_x2_numpy = None;        
      
        if self.pts_x2 is None:
          self.pts_x2 = pts_x1;

        self.epsilon = epsilon;
      
        if tree_points is None: # build kd-tree of points for neighbor listing
          if self.pts_x1_numpy is None: self.pts_x1_numpy = pts_x1.cpu().numpy();
          self.tree_points = spatial.cKDTree(self.pts_x1_numpy);

        if device is None:
          device = torch.device('cpu');

        self.device = device;
                  
        self.cached_data = {}; # create empty cache for storing data          
        generate_mapping = MapToPoly_Function.generate_mapping;
        self.cached_data['map_data'] = generate_mapping(self.weight_func,self.weight_func_params,
                                                        self.porder,self.epsilon, 
                                                        self.pts_x1,self.pts_x2,
                                                        self.tree_points,self.device,
                                                        self.flag_verbose);
        
      elif flag_cache_mode == 'load_from_file': # setup by loading data from cache file
          
        if 'cache_filename' in extra_params:
          cache_filename = extra_params['cache_filename'];
        else:
          raise Exception('No cache_filename specified.');

        self.load_cache_data(cache_filename); # load data from file
          
      else:
        print("flag_cache_mode = " + str(flag_cache_mode));
        raise Exception('flag_cache_mode is invalid.');

  def save_cache_data(self,cache_filename):
    r"""Save needed matrices and related data to .pickle for later cached use. (Warning: prototype codes here currently and not tested)."""
    # collect the data to save  
    d = {};
    d['porder']  = self.porder;      
    d['epsilon'] = self.epsilon;
    if self.pts_x1_numpy is None: self.pts_x1_numpy = pts_x1.cpu().numpy();
    d['pts_x1'] = self.pts_x1_numpy;
    if self.pts_x2_numpy is None: self.pts_x2_numpy = pts_x2.cpu().numpy();            
    d['pts_x2'] = self.pts_x2_numpy;
    d['weight_func_str'] = str(self.weight_func);
    d['weight_func_params'] = self.weight_func_params;
    d['version'] = __version__;  # Module version      
    d['cached_data'] = self.cached_data;

    # write the data to disk
    f = open(cache_filename,'wb');
    p.dump(d,f); # load the data from file
    f.close();

  def load_cache_data(self,cache_filename):
      r"""Load the needed matrices and related data from .pickle. (Warning: prototype codes here currently and not tested)."""
      f = open(cache_filename,'rb');
      d = p.load(f); # load the data from file
      f.close();

      print(d.keys())
      self.porder = d['porder'];
      self.epsilon = d['epsilon'];

      self.weight_func = d['weight_func_str'];
      self.weight_func_params = d['weight_func_params'];

      self.pts_x1 = torch.from_numpy(d['pts_x1']).to(device);
      self.pts_x2 = torch.from_numpy(d['pts_x2']).to(device);

      self.pts_x1_numpy = d['pts_x1'];
      self.pts_x2_numpy = d['pts_x2'];

      if self.pts_x2 is None:
        self.pts_x2 = pts_x1;

      # build kd-tree of points for neighbor listing
      if self.pts_x1_numpy is None: self.pts_x1_numpy = pts_x1.cpu().numpy();
      self.tree_points = spatial.cKDTree(self.pts_x1_numpy);

      self.cached_data = d['cached_data'];
    
  def eval_poly(self,pts_x,pts_x2_i0,c_star_i0,porder=None,flag_verbose=None):
    r"""Evaluates the polynomial reconstruction around a given target point pts_x2_i0."""
    if porder is None:
      porder = self.porder;
    if flag_verbose is None:
      flag_verbose = self.flag_verbose;
    MapToPoly_Function.eval_poly(pts_x,pts_x2_i0,c_star_i0,porder,flag_verbose);
      
  def forward(self, input): # define the action of this layer
      r"""For a field u specified at points xj, performs the mapping to coefficients c at points xi, (uj,xj) :math:`\rightarrow` (ci,xi)."""
      flag_time_it = False;
      if flag_time_it:
        time_1 = time.time();
        
      # We evaluate the action of the function, backward will be called automatically when computing gradients.
      uj = input;
      output = MapToPoly_Function.apply(uj,self.porder,
                                            self.weight_func,self.weight_func_params,
                                            self.pts_x1,self.epsilon,self.pts_x2,
                                            self.cached_data,self.tree_points,self.device,
                                            self.flag_verbose);
      if flag_time_it:
        msg = 'MapToPoly->forward(): ';
        msg += 'elapsed_time = %.4e'%(time.time() - time_1);
        print(msg);
        
      return output;
  
  def extra_repr(self):
      r"""Displays information associated with this module."""
      # Display some extra information about this layer.
      return 'porder={}, weight_func={}, weight_func_params={}, pts_x1={}, pts_x2={}'.format(
          self.porder, self.weight_func, self.weight_func_params, self.pts_x1.shape, self.pts_x2.shape
      );

  def to(self, device):
    r"""Moves data to GPU or other specified device."""
    super(MapToPoly,self).to(device);
    self.pts_x1 = self.pts_x1.to(device);
    self.pts_x2 = self.pts_x2.to(device);
    return self;  

class MaxPoolOverPoints(nn.Module):
  r"""Applies a max-pooling operation to obtain values :math:`v_i = \max_{j \in \mathcal{N}_i(\epsilon)} \{u_j\}.` """  
  def __init__(self,pts_x1,epsilon=None,pts_x2=None,
               indices_xj_i_cache=None,tree_points=None,
               device=None,flag_verbose=0,**extra_params):
      r"""Setup of max-pooling operation.

          Args:
            pts_x1 (Tensor): The collection of domain points :math:`x_j`.  We assume size [num_pts,num_dim].
            epsilon (float): The :math:`\epsilon`-neighborhood size to use to sort points (should be compatible with choice of weight_func_params).
            pts_x2 (Tensor): The collection of target points :math:`x_i`.
            indices_xj_i_cache (dict): Stored data to help speed up repeated calculations.
            tree_points (dict): Stored data to help speed up repeated calculations.
            device: Device on which to perform calculations (GPU or other, default is CPU).
            flag_verbose (int): Level of reporting on progress during the calculations.
            **extra_params (dict): Extra parameters allowing for specifying layer name and caching mode. 
       
      """      
      super(MaxPoolOverPoints,self).__init__();

      self.flag_verbose = flag_verbose;

      if device is None:
        device = torch.device('cpu');

      self.device = device;
      
      if 'name' in extra_params:
        self.name = extra_params['name'];
      else:
        self.name = "default_name";
  
      if 'flag_cache_mode' in extra_params:
        flag_cache_mode = extra_params['flag_cache_mode'];
      else:
        flag_cache_mode = 'generate1';

      if flag_cache_mode == 'generate1':  # setup from scratch
        self.pts_x1 = pts_x1;
        self.pts_x2 = pts_x2;

        self.pts_x1_numpy = None;
        self.pts_x2_numpy = None;        
      
        if self.pts_x2 is None:
          self.pts_x2 = pts_x1;

        self.epsilon = epsilon;
      
        if tree_points is None: # build kd-tree of points for neighbor listing
          if self.pts_x1_numpy is None: self.pts_x1_numpy = pts_x1.cpu().numpy();
          self.tree_points = spatial.cKDTree(self.pts_x1_numpy);

        if indices_xj_i_cache is None:
          self.indices_xj_i_cache = None; # cache the neighbor lists around each xi
        else:
          self.indices_xj_i_cache = indices_xj_i_cache;

        if device is None:
          device = torch.device('cpu');

        self.device = device;
                  
        self.cached_data = {}; # create empty cache for storing data
        
      elif flag_cache_mode == 'load_from_file': # setup by loading data from cache file
          
        if 'cache_filename' in extra_params:
          cache_filename = extra_params['cache_filename'];
        else:
          raise Exception('No cache_filename specified.');

        self.load_cache_data(cache_filename); # load data from file
          
      else:
        print("flag_cache_mode = " + str(flag_cache_mode));
        raise Exception('flag_cache_mode is invalid.');

  def save_cache_data(self,cache_filename):
    r"""Save data to .pickle file for caching. (Warning: Prototype placeholder code.)"""      
    # collect the data to save  
    d = {};
    d['epsilon'] = self.epsilon;
    if self.pts_x1_numpy is None: self.pts_x1_numpy = pts_x1.cpu().numpy();
    d['pts_x1'] = self.pts_x1_numpy;
    if self.pts_x2_numpy is None: self.pts_x2_numpy = pts_x2.cpu().numpy();            
    d['pts_x2'] = self.pts_x2_numpy;
    d['version'] = __version__;  # Module version      
    d['cached_data'] = self.cached_data;

    # write the data to disk
    f = open(cache_filename,'wb');
    p.dump(d,f); # load the data from file
    f.close();

  def load_cache_data(self,cache_filename):
      r"""Load data to .pickle file for caching. (Warning: Prototype placeholder code.)"""      
      f = open(cache_filename,'rb');
      d = p.load(f); # load the data from file
      f.close();

      print(d.keys())
      self.epsilon = d['epsilon'];

      self.pts_x1 = torch.from_numpy(d['pts_x1']).to(device);
      self.pts_x2 = torch.from_numpy(d['pts_x2']).to(device);

      self.pts_x1_numpy = d['pts_x1'];
      self.pts_x2_numpy = d['pts_x2'];

      if self.pts_x2 is None:
        self.pts_x2 = pts_x1;

      # build kd-tree of points for neighbor listing
      if self.pts_x1_numpy is None: self.pts_x1_numpy = pts_x1.cpu().numpy();
      self.tree_points = spatial.cKDTree(self.pts_x1_numpy);

      self.cached_data = d['cached_data'];
            
  def forward(self, input): # define the action of this layer
      r"""Applies a max-pooling operation to obtain values :math:`v_i = \max_{j \in \mathcal{N}_i(\epsilon)} \{u_j\}.` 

          Args:
            input (Tensor): The collection uj of field values at the points xj.

          Returns:
            Tensor: The collection of field values vi at the target points xi.

      """      
      flag_time_it = False;
      if flag_time_it:
        time_1 = time.time();
      
      uj = input;
      output = MaxPoolOverPoints_Function.apply(uj,self.pts_x1,self.epsilon,self.pts_x2,
                                                self.indices_xj_i_cache,self.tree_points,
                                                self.flag_verbose);
      
      if flag_time_it:
        msg = 'MaxPoolOverPoints->forward(): ';
        msg += 'elapsed_time = %.4e'%(time.time() - time_1);
        print(msg);

      return output;

  def extra_repr(self):
      r"""Displays information associated with this module."""
      return 'pts_x1={}, pts_x2={}'.format(self.pts_x1.shape, self.pts_x2.shape);

  def to(self, device):
    r"""Moves data to GPU or other specified device."""
    super(MaxPoolOverPoints, self).to(device);
    self.pts_x1 = self.pts_x1.to(device);
    self.pts_x2 = self.pts_x2.to(device);
    return self;

class GMLS_Layer(nn.Module):
  r"""The GMLS-Layer processes scattered data by using Generalized Moving Least 
  Squares (GMLS) to construct a local reconstruction of the data (here polynomials).  
  This is represented by coefficients that are mapped to approximate the action of 
  linear or non-linear operators on the input field.

  As depicted above, the architecture processes a collection of input channels 
  into intermediate coefficient channels.  The coefficient channels are 
  then collectively mapped to output channels.  The mappings can be any unit 
  for which back-propagation can be performed.  This includes linear 
  layers or non-linear maps based on multilayer perceptrons (MLPs).

  Examples:

    Here is a typical way to construct a GMLS-Layer.  This is done in 
    the following stages.

    ``(i)`` Construct the scattered data locations xj, xi at which processing will occur.  Here, we create points in 2D.

    >>> xj = torch.randn((100,2),device=device); xi = torch.randn((100,2),device=device);
 
    ``(ii)`` Construct the mapping unit that will be applied pointwise.  Here we create an MLP 
    with Nc input coefficient channels and channels_out output channels.

    >>> layer_sizes = [];
    >>> num_input   = Nc*num_polys; # number of channels (NC) X number polynomials (num_polys) (cross-channel coupling allowed)
    >>> num_depth   = 4; num_hidden  = 100; channels_out = 16; # depth, width, number of output filters 
    >>> layer_sizes.append(num_polys);
    >>> for k in range(num_depth):
    >>>   layer_sizes.append(num_hidden);
    >>> layer_sizes.append(1); # a single unit always gives scalar output, we then use channels_out units.    
    >>> mlp_q_map1 = gmlsnets_pytorch.nn.MLP_Pointwise(layer_sizes,channels_out=channels_out);

    ``(iii)`` Create the GMLS-Layer using these components.

    >>> weight_func1 = gmlsnets_pytorch.nn.MapToPoly_Function.weight_one_minus_r; 
    >>> weight_func_params = {'epsilon':1e-3,'p'=4};
    >>> gmls_layer_params = {
              'flag_case':'standard','porder':4,'Nc':3,
              'mlp_q1':mlp_q_map1,
              'pts_x1':xj,'pts_x2':xi,'epsilon':1e-3,
              'weight_func1':weight_func1,'weight_func1_params':weight_func1_params,
              'device':device,'flag_verbose':0
              };
    >>> gmls_layer=gmlsnets_pytorch.nn.GMLS_Layer(**gmls_layer_params);

    Here is an example of how a GMLS-Layer and other modules in this package
    can be used to process scattered data.  This could be part of a larger 
    neural network in practice (see example codes for more information).  For instance,
        
    >>> layer1 = nn.Sequential(gmls_layer, # produces output tuple of tensors (ci,xi) with shapes ([batch,ci,xi],[xi]).
                               #PdbSetTraceLayer(),
                               ExtractFromTuple(index=0), # from output keep only the ui part and discard the xi part.
                               #PdbSetTraceLayer(),
                               PermuteLayer((0,2,1)) # organize indexing to be [batch,xi,ci], for further processing.
                               ).to(device);

    You can uncomment the PdbSetTraceLayer() to get breakpoints for state information and tensor shapes during processing.
    The PermuteLayer() changes the order of the indexing.  Also can use ReshapeLayer() to reshape the tensors, which is 
    especially useful for processing data related to CNNs.

    Much of the construction can be further simplified by writing a few wrapper classes for your most common use cases.

  More information also can be found in the example codes directory.
  """  
  def __init__(self, flag_case, porder, pts_x1, epsilon, weight_func, weight_func_params,
               mlp_q = None,pts_x2 = None, device = None, flag_verbose = 0):
    r"""
        Initializes the GMLS layer.  

        Args:
          flag_case (str): Flag for the type of architecture to use (default is 'standard').
          porder (int): Order of the basis to use (polynomial degree).
          pts_x1 (Tensor): The collection of domain points :math:`x_j`.
          epsilon (float): The :math:`\epsilon`-neighborhood size to use to sort points (should be compatible with choice of weight_func_params).
          weight_func (func): Weight function to use.
          weight_func_params (dict): Weight function parameters.
          mlp_q (module): Mapping q unit for computing :math:`q(c)`, where c are the coefficients.
          pts_x2 (Tensor): The collection of target points :math:`x_i`.
          device: Device on which to perform calculations (GPU or other, default is CPU).
          flag_verbose (int): Level of reporting on progress during the calculations.
    """
    super(GMLS_Layer, self).__init__();

    if flag_case is None:
      self.flag_case = 'standard';
    else:
      self.flag_case = flag_case;
    
    if device is None:
      device = torch.device('cpu');
    
    self.device = device;

    if self.flag_case == 'standard':
      tree_points = None;
      self.MapToPoly_1 = MapToPoly(porder, weight_func, weight_func_params, 
                                                                      pts_x1, epsilon, pts_x2, tree_points,
                                                                      device, flag_verbose);
      
      if mlp_q is None: # if not specified then create some default custom layers
        raise Exception("Need to specify the mlp_q module for mapping coefficients to output.");    
      else: # in this case initialized outside
        self.mlp_q = mlp_q;

    else:
      print("flag_case = " + str(flag_case));        
      print("self.flag_case = " + str(self.flag_case));
      raise Exception('flag_case not valid.');
      
  def forward(self, input):
    r"""Computes GMLS-Layer processing scattered data input field uj to obtain output field vi.

        Args: 
          input (Tensor): Input channels uj organized in the shape [batch,xj,uj].    

        Returns:
          tuple: The output channels and point locations (vi,xi).  The field vi = q(ci).

    """
    if self.flag_case == 'standard':

      map_output     = self.MapToPoly_1.forward(input); 
      c_star_i       = map_output[0];
      pts_x2         = map_output[1];
      
      # MLP should apply across all channels and coefficients (coeff capture spatial, like kernel)
      fc_input = c_star_i.permute((0,2,1,3)); # we organize as [batchI,ptsI,channelsI,coeffI]

      # We assume MLP can process channelI*Nc + coeffI.
      # We assume output of out = fc, has shape [batchI,ptsI,channelsNew]
      # Outside routines can reshape that into an nD array again for structure samples or use over scattered samples.
      q_of_c_star_i  = self.mlp_q.forward(fc_input);
                                                                     
      pts_x2_p = None; # currently returns None to simplify back-prop and debugging, but could just return the pts_x2.
      
      return_vals = q_of_c_star_i, pts_x2_p;

    return return_vals;

  def to(self, device):
    r"""Moves data to GPU or other specified device."""
    super(GMLS_Layer, self).to(device);
    self.MapToPoly_1 = self.MapToPoly_1.to(device);
    self.mlp_q = self.mlp_q.to(device);
    return self;


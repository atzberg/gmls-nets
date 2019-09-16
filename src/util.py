"""
  Collection of utility routines.
"""

# Authors: B.J. Gross and P.J. Atzberger
# Website: http://atzberger.org/

import torch;
import torch.nn as nn;
import numpy as np;

import pdb
import time

#******************************************
# Custom Functions
#******************************************
class PeriodicPad2dFunc(torch.autograd.Function):
    """Performs periodic padding/tiling of a 2d lattice."""

    @staticmethod
    def forward(ctx, input, pad=None,flag_coords=False):
        """
        Periodically tiles the input into a 2d array for 3x3 repeat pattern.  
        This allows for a quick way for handling periodic boundary conditions 
        on the units cell.
        
        Args:
          input (Tensor): A tensor of size [nbatch,nchannel,nx,ny] or [nx,ny].
          pad (float): The pad value to use.
          flag_coord (bool): If beginning components are coordinates (x,y,z,...,u).

        We then adjust (x,y,u) --> (x + i, y + j, u) for image (i,j).          
        
        Returns:
          output (Tensor): A tensor of the same size [nbatch,nchannel,3*nx,3*ny] or [3*nx,3*ny].
        
        """
        # We use alternative by concatenating the arrays together to tile.
        # Process Tensor inputs of the shape with [nbatch,nchannel,nx,ny].
        # We also allow the case of [nx,ny] for single input.               
        nd = input.dim();
        if (nd > 4):
          raise Exception('Expects tensor that has number of dimensions <= 4 (dim = {}).'.format(nd));
        
        if (nd < 2):
          raise Exception('Expects tensor that has at least number of dimensions >= 2 (dim = {}).'.format(nd));            
        
        a  = input;
        if (nd == 2):
          # add extra dimensions so we can process all tensors the same way
          a = input.unsqueeze(0).unsqueeze(0); # size --> [1,1,nx,ny]          

        w1 = w2 = a;
        aa = torch.cat([w1,a,w2],dim=2);
        h1 = h2 = aa;
        output = torch.cat([h1,aa,h2],dim=3);

        if flag_coords: # indicates (x,y,u) input, so need
                        # to adjust x + i-1 and y + j-1, i,j = 0,1,2 to extend.
          coordI1 = 0; coordI2 = 1; #x and y components
          N1 = output.shape[2]//3; # block size
          N2 = output.shape[3]//3;
          for j in range(0,3):
            blockI2 = np.arange(N2*j,N2*(j + 1),dtype=int);
            output[:,coordI1,:,blockI2] += (j - 1);

          for i in range(0,3):
            blockI1 = np.arange(N1*i,N1*(i + 1),dtype=int);
            output[:,coordI2,blockI1,:] += (i - 1);              
        
        if (nd == 2): # if only [nx,ny] then squeeze our extra dimensions
          output = output.squeeze(0).squeeze(0);        
        
        ctx.pad = pad;
        ctx.size = input.size();
        ctx.numel = input.numel();
        ctx.num_tiles = 3;
                
        return output;

    @staticmethod
    def backward(ctx, grad_output):
        r"""Compute df/dx from df/dy using the Chain Rule df/dx = df/dx*dy/dx.
            For periodic padding we use df/dx = sum_{y_i ~ x} df/dy_i, where 
            the y_i ~ x are all points y_i that are equivalent to x under the periodic extension.
        """              
        num_tiles = ctx.num_tiles;
        
        nd = input.dim();
        if (nd > 4):
          raise Exception('Expects tensor that has number of dimensions <= 4 (dim = {}).'.format(nd));
        
        if (nd < 2):
          raise Exception('Expects tensor that has at least number of dimensions >= 2 (dim = {}).'.format(nd));                    
        
        b = grad_output; # short-hand for grad_output (size of output)
        if (nd == 2):
          # add extra dimensions so we can process all tensors the same way
          b = b.unsqueeze(0).unsqueeze(0); # size --> [1,1,nx,ny]          
                                       
        # construct indices to contract the periodic images in each dimension
        ind_r = torch.zeros(ctx.size,dtype=torch.int64);
        torch.arange(0, ctx.size[2]*num_tiles, out=ind_r);
        ind_r = ind_r.fmod(ctx.size[2]); # repeat indices number of rows [0,1,..nrow-1,0,...nrow-1].
        ind_r = ind_r.view(-1);

        ind_c = torch.zeros(ctx.size,dtype=torch.int64);
        torch.arange(0, ctx.size[3]*num_tiles, out=ind_c);
        ind_c = ind_c.fmod(ctx.size[3]); # repeat indices number of cols [0,1,..ncol-1,0,...ncol-1].
        ind_c = ind_c.view(-1);

        c = b.new_zeros(ctx.size[0],ctx.size[1],ctx.size[2],ctx.size[3]*num_tiles);

        c = c.index_add(2,ind_r,grad_output); # add the rows together to start contracting periodicity        

        d = b.new_zeros(a.transpose(2,3).size());
        d = d.index_add(2,ind_c,c.t()); # add the cols together to contract periodicity
        grad_input = d.transpose(2,3); # Note, this includes contraction of grad_output already, so is df/dx.
                
        if (nd == 2): # if only [nx,ny] then squeeze out extra dimensions
          output = grad_input.squeeze(0).squeeze(0);

        return grad_input;
            
#------------------
class ExtractUnitCell2dFunc(torch.autograd.Function):
    r"""Extracts the 2d unit cell from periodic tiling."""
    
    @staticmethod
    def forward(ctx, input, pad=None):
        r"""Extracts the 2d unit cell from periodic tiling.
        
        Args:
          input (Tensor): Tensor of size [nbatch,nchannel,nx,ny] or [nx,ny].
          
        Returns:
          Tensor: The Tensor of size [nbatch,nchannel,nx/3,ny/3] or [nx/3,ny/3].
                
        """ 
        num_tiles = 3;
        ctx.num_tiles = num_tiles;
        
        nd = input.dim();
        if (nd > 4):
          raise Exception('Expects tensor that has number of dimensions <= 4 (dim = {}).'.format(nd));
        
        if (nd < 2):
          raise Exception('Expects tensor that has at least number of dimensions >= 2 (dim = {}).'.format(nd));                    
        
        a = input; # short-hand 
        if (nd == 2):
          # add extra dimensions so we can process all tensors the same way
          a = a.unsqueeze(0).unsqueeze(0); # size --> [1,1,nx,ny]
        
        chunks_col = torch.chunk(a,num_tiles,dim=2);
        aa = torch.chunk(chunks_col[1],num_tiles,dim=3);  # extract middle of middle
        
        output = aa[1]; # choose middle one (assumes num_tiles==3)
                        
        if (nd == 2): # if only [nx,ny] then squeeze out extra dimensions
          output = output.squeeze(0).squeeze(0);
        
        return output;

    @staticmethod
    def backward(ctx, grad_output):
        r""" Compute df/dx from df/dy using the Chain Rule df/dx = df/dx*dy/dx.
        For periodic padding we use df/dx = sum_{y_i ~ x} df/dy_i, where 
        the y_i ~ x are all points y_i that are equivalent to x under the periodic extension. 
        
        For extracting from periodic tiling the unit cell, the derivatives dy/dx are zero 
        unless x is within the unit cell.  The block matrix is dy/dx = [[Z,Z,Z],[Z,W,Z],[Z,Z,Z]],
        where W is the derivative values in the unit cell (df/dy) and Z is the zero matrix.
        """        
        num_tiles = ctx.num_tiles;
        
        nd = grad_output.dim();
        if (nd > 4):
          raise Exception('Expects tensor that has number of dimensions <= 4 (dim = {}).'.format(nd));
        
        if (nd < 2):
          raise Exception('Expects tensor that has at least number of dimensions >= 2 (dim = {}).'.format(nd));                            

        W = grad_output; # short-hand for grad_output (size of output)
        if (nd == 2):
          # add extra dimensions so we can process all tensors the same way
          W = W.unsqueeze(0).unsqueeze(0); # size --> [1,1,nx,ny]          
                        
        s = W.size();
        Z = grad_output.new_zeros(s[0],s[1],s[2],s[3]);
        A1 = torch.cat([Z,Z,Z],dim=3);
        B1 = torch.cat([Z,W,Z],dim=3);
        bb = torch.cat([A1,B1,A1],dim=2);
        
        grad_input = bb; 
                
        if (nd == 2): # if only [nx,ny] then squeeze out extra dimensions
          grad_input = grad_input.squeeze(0).squeeze(0);
        
        return grad_input;

#******************************************
# Custom Modules
#******************************************
class PeriodicPad2d(nn.Module):
    def __init__(self,flag_coords=False):
        r"""Setup for computing the periodic tiling."""
        super(PeriodicPad2d, self).__init__()
        self.flag_coords = flag_coords;

    def forward(self, input):
        r"""Compute the periodic padding of the input. """
        return PeriodicPad2dFunc.apply(input,None,self.flag_coords);

    def extra_repr(self):
        r"""Displays some of the information associated with the module. """
        return 'PeriodicPad2d: (no internal parameters)';

class ExtractUnitCell2d(nn.Module):
    def __init__(self):
        super(ExtractUnitCell2d, self).__init__()

    def forward(self, input):
        r"""Computes the periodic padding of the input."""
        return ExtractUnitCell2dFunc.apply(input);

    def extra_repr(self):
        r"""Displays some of the information associated with the module. """
        return 'ExtractUnitCell2d: (no internal parameters)';


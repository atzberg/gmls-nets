"""
  Collection of codes for generating some training data sets.
"""

# Authors: B.J. Gross and P.J. Atzberger
# Website: http://atzberger.org/

import torch;
import numpy as np;
import pdb;

class diffOp1(torch.utils.data.Dataset):
  r"""
  Generates samples of the form :math:`(u^{[i]},f^{[i]})` where :math:`f^{[i]} = L[u^{[i]}]`, 
  where :math:`i` denotes the index of the sample.

  Stores data samples in the form :math:`(u,f)`.

  The samples of u are represented as a tensor of size [nsamples,nchannels,nx]
  and sample of f as a tensor of size [nsamples,nchannels,nx].
    
  Note:
    For now, please use nx that is odd.  In this initial implementation, we use a 
    method based on conjugated flips with formula for the odd case which is slightly
    simpler than other case.
    
  """
  def flipForFFT(self,u_k_part):
    r"""We flip as :math:`f_k = f_{N-k}`.  Notice that only :math:`0,\ldots,N-1` entries 
    stored.  This is useful for constructing real-valued function representations
    from random coefficients.  Real-valued function requires :math:`conj(f_k) = f_{N-k}`.
    We can use this flip to construct from random coefficients the term
    :math:`u_k = f_k + conj(flip(f_k))`,  then above constraint is satisfied.

    Args:
      a (Tensor): 1d array to flip.      
      
    Returns:
      Tensor: The flipped tensors symmetric under conjucation.      
    """
    nx = self.nx;        
    uu = u_k_part[:,:,nx:0:-1];
    vv = u_k_part[:,:,0];
    vv = np.expand_dims(vv,2);
    uu_k_flip = np.concatenate([vv,uu],2);     
    
    return uu_k_flip;

  def getComplex(self,a,b):
    j = np.complex(0,1); # create complex number (or use 1j).
    c = a + j*b;
    return c;

  def getRealImag(self,c):
    a = np.real(c);
    b = np.imag(c);
    return a,b;

  def computeLSymbol_ux(self):
    r"""Compute associated Fourier symbols for use under DFT for the operator L[u]."""
    nx = self.nx;
    vec_k1 = torch.zeros(nx);
    vec_k1_pp = torch.zeros(nx);    
    vec_k_sq = torch.zeros(nx);
    L_symbol_real = torch.zeros(nx,dtype=torch.float32);
    L_symbol_imag = torch.zeros(nx,dtype=torch.float32);
    two_pi = 2.0*np.pi;
    #two_pi_i = two_pi*1j; # $2\pi{i}$, 1j = sqrt(-1)
    for i in range(0,nx):
      vec_k1[i] = i;
      if (vec_k1[i] < nx/2):
        vec_k1_p = vec_k1[i];
      else:
        vec_k1_p = vec_k1[i] - nx;
      vec_k1_pp[i] = vec_k1_p;      
      L_symbol_real[i] = 0.0;
      L_symbol_imag[i] = two_pi*vec_k1_p;

      L_hat = self.getComplex(L_symbol_real.numpy(),L_symbol_imag.numpy());

    return L_hat, vec_k1_pp;

  def computeLSymbol_uxx(self):
    r"""Compute associated Fourier symbols for use under DFT for the operator L[u]."""
    nx = self.nx;
    vec_k1 = torch.zeros(nx);
    vec_k1_pp = torch.zeros(nx);    
    vec_k_sq = torch.zeros(nx);
    L_symbol_real = torch.zeros(nx,dtype=torch.float32);
    L_symbol_imag = torch.zeros(nx,dtype=torch.float32);
    neg_four_pi_sq = -4.0*np.pi*np.pi;    
    for i in range(0,nx):      
      vec_k1[i] = i;      
      vec_k_sq[i] = vec_k1[i]*vec_k1[i];
      if (vec_k1[i] < nx/2):
        vec_k1_p = vec_k1[i];
      else:
        vec_k1_p = vec_k1[i] - nx;
      vec_k1_pp[i] = vec_k1_p;
      vec_k_p_sq = vec_k1_p*vec_k1_p;
      L_symbol_real[i] = neg_four_pi_sq*vec_k_p_sq;
      L_symbol_imag[i] = 0.0;
        
      L_hat = self.getComplex(L_symbol_real.numpy(),L_symbol_imag.numpy());

    return L_hat, vec_k1_pp;

  def computeCoeffActionL(self,u_hat,L_hat):
    r"""Computes the action of operator L used for data generation in Fourier space."""
    u_k_real, u_k_imag = self.getRealImag(u_hat);
    L_symbol_real, L_symbol_imag = self.getRealImag(L_hat);

    f_k_real = L_symbol_real*u_k_real - L_symbol_imag*u_k_imag; #broadcast will distr over copies of u.    
    f_k_imag = L_symbol_real*u_k_imag + L_symbol_imag*u_k_real;

    # Generate samples u and f using ifft
    f_hat = self.getComplex(f_k_real,f_k_imag);

    return f_hat;

  def computeActionL(self,u,L_hat):
    r"""Computes the action of operator L used for data generation."""    
    raise Exception('Currently this routine not debugged, need to test first.')
    
    if flag_verbose > 0:
      print("computeActionL(): WARNING: Not yet fully tested.");
    
    # perform FFT to get u_hat
    u_hat = np.fft.fft(u);

    # compute action of L_hat
    f_hat = self.computeCoeffActionL(u_hat,L_hat);

    # compute inverse FFT to get f
    f = np.fft.ifft(f_hat);

    return f;

  def __init__(self,op_type='uxx',op_params=None,
               gen_mode='exp1',gen_params={'alpha1':0.1},
               num_samples=int(1e4),nchannels=1,nx=15,               
               flag_verbose=0, **extra_params):
    r"""Setup for data generation.
    
        Args:
          op_type (str): The differential operator to sample.
          op_params (dict): The operator parameters.
          gen_mode (str): The mode for the data generator.
          gen_params (dict): The parameters for the given generator.
          num_samples (int): The number of samples to generate.
          nchannels (int): The number of channels.
          nx (int): The number of input sample points. 
          flag_verbose (int): Level of reporting during calculations.
          extra_params (dict): Extra parameters for the sampler.
                                
        For extra_params we have:
          noise_factor (float): The amount of noise to add to samples.
          scale_factor (float): A factor to scale magnitude of the samples.
          flagComputeL (bool): If the fourier symbol of operator should be computed.
          
        For generator modes we have:
          gen_mode == 'exp1': 
            alpha1 (float): The decay rate.          

        Note:
          For now, please use only nx that is odd.  In this initial implementation, we use a 
          method based on conjugated flips with formula for the odd case which is slightly
          simpler than other case.
    """
    super(diffOp1, self).__init__();
    
    if flag_verbose > 0:
      print("Generating the data samples which can take some time.");
      print("num_samples = %d"%num_samples);

    self.op_type=op_type;
    self.op_params=op_params;

    self.gen_mode=gen_mode;
    self.gen_params=gen_params;
    
    self.num_samples=num_samples;
    self.nchannels=nchannels;
    self.nx=nx; 

    if (nx % 2 == 0):
      msg = "Not allowed yet to use nx that is even. ";
      msg += "For now, please just use nx that is odd given the flips currently used."
      raise Exception(msg);
    
    noise_factor=0;scale_factor=1.0;flagComputeL=False; # default values
    if 'noise_factor' in extra_params:
      noise_factor = extra_params['noise_factor'];

    if 'scale_factor' in extra_params:        
      scale_factor = extra_params['scale_factor'];

    if 'flagComputeL' in extra_params:        
      flagComputeL = extra_params['flagComputeL'];

    # Generate for the operator the Fourier symbols
    if self.op_type == 'ux' or self.op_type == 'u*ux' or self.op_type == 'ux*ux':
      L_hat, vec_k1_pp = self.computeLSymbol_ux();
    elif self.op_type == 'uxx' or self.op_type == 'u*uxx' or self.op_type == 'uxx*uxx':
      L_hat, vec_k1_pp = self.computeLSymbol_uxx();
    else:
      raise Exception("Unkonwn operator type.");

    if (flagComputeL):
      L_i        = np.fft.ifft(L_hat);
      self.L_hat = L_hat;
      self.L_i   = L_i;
      u          = np.zeros(nx);
      i0         = int(nx/2);      
      u[i0]      = 1.0;
      self.G_i   = self.computeActionL(u);

    # Generate random input function (want real-valued)
    # conj(u_k) = u_{N -k} needs to hold.
    u_k_real = np.random.randn(num_samples,nchannels,nx);
    u_k_imag = np.random.randn(num_samples,nchannels,nx);

    # scale modes to make smooth
    if gen_mode=='exp1':
      alpha1 = gen_params['alpha1'];
      factor_k = scale_factor*np.exp(-alpha1*vec_k1_pp**2);
      factor_k = factor_k.numpy();
    else:
      raise Exception("Generation mode not recognized.");
    
    u_k_real = u_k_real*factor_k; # broadcast will apply over last two dimensions
    u_k_imag = u_k_imag*factor_k; # broadcast will apply over last two dimensions    
    
    flag_debug = False;
    if flag_debug:
      if flag_verbose > 0:
        print("WARNING: debugging mode on.");
        
      u_k_real = 0.0*u_k_real;
      u_k_imag = 0.0*u_k_imag;
    
      u_k_real[0,0,1] = nx;
      u_k_imag[0,0,1] = 0;
    
      u_k_real[1,0,1] = 0;
      u_k_imag[1,0,1] = nx;
    
      u_k_real[2,0,1] = nx;
      u_k_imag[2,0,1] = nx;

    # flip modes for constructing rep of real-valued function    
    u_k_real_flip = self.flipForFFT(u_k_real);
    u_k_imag_flip = self.flipForFFT(u_k_imag);
    
    u_k_real_p = 0.5*u_k_real + 0.5*u_k_real_flip; # make conjugate conj(u_k) = u_{N -k}
    u_k_imag_p = 0.5*u_k_imag - 0.5*u_k_imag_flip; # make conjugate conj(u_k) = u_{N -k}
        
    u_k_real_p = torch.from_numpy(u_k_real_p);
    u_k_imag_p = torch.from_numpy(u_k_imag_p);
    
    u_k_real_p = u_k_real_p.type(torch.float32);
    u_k_imag_p = u_k_imag_p.type(torch.float32);
                    
    u_hat = self.getComplex(u_k_real_p.numpy(),u_k_imag_p.numpy());
        
    f_hat = self.computeCoeffActionL(u_hat,L_hat);
    f_hat = f_hat; # target operator relation for PDEs later is Lu = -f, so f = -Lu.
        
    # Generate samples u and f, in 2d using ifft2.    
    # ifft2 is broadcast over last two indices
    # perform inverse DFT to get u and f.
    u_i = np.fft.ifft(u_hat);
    f_i = np.fft.ifft(f_hat);
    
    if self.op_type == 'u*ux':
      f_i = u_i*f_i;
    elif self.op_type == 'ux*ux':
      f_i = f_i*f_i; 
    elif self.op_type == 'u*uxx':
      f_i = u_i*f_i;
    elif self.op_type == 'uxx*uxx':
      f_i = f_i*f_i;
        
    self.samples_X = torch.from_numpy(np.real(u_i)).type(torch.float32); # only grab real part
    self.samples_Y = torch.from_numpy(np.real(f_i)).type(torch.float32);
    
    if noise_factor > 0:
      self.samples_Y += noise_factor*torch.randn(*self.samples_Y.shape);
        
  def __len__(self):    
    return self.samples_X.size()[0];

  def __getitem__(self,index):
    return self.samples_X[index],self.samples_Y[index];

  def to(self,device):    
    self.samples_X = self.samples_X.to(device);
    self.samples_Y = self.samples_Y.to(device);
    
    return self;

class diffOp2(torch.utils.data.Dataset):
  r"""
  Generates samples of the form :math:`(u^{[i]},f^{[i]})` where :math:`f^{[i]} = L[u^{[i]}]`, 
  where :math:`i` denotes the index of the sample.

  Stores data samples in the form :math:`(u,f)`.

  The samples of u are represented as a tensor of size [nsamples,nchannels,nx]
  and sample of f as a tensor of size [nsamples,nchannels,nx].
   
  Note:
    For now, please use nx that is odd.  In this initial implementation, we use a 
    method based on conjugated flips with formula for the odd case which is slightly
    simpler than other case.
    
  """
  def flipForFFT(self,u_k_part):
    r"""We flip as :math:`f_k = f_{N-k}`.  Notice that only :math:`0,\ldots,N-1` entries 
    stored.  This is useful for constructing real-valued function representations
    from random coefficients.  Real-valued function requires :math:`conj(f_k) = f_{N-k}`.
    We can use this flip to construct from random coefficients the term
    :math:`u_k = f_k + conj(flip(f_k))`,  then above constraint is satisfied.

    Args:
      a (Tensor): 1d array to flip.      
      
    Returns:
      Tensor: The flipped tensors symmetric under conjucation.      
    """
    nx = self.nx;ny = self.ny;
    
    u_k_part_row0 = u_k_part[:,:,0,:];
    u_k_part_row0 = np.expand_dims(u_k_part_row0,2);
    u_k_part_ex = np.concatenate([u_k_part,u_k_part_row0],2);

    u_k_part_col0 = u_k_part_ex[:,:,:,0];
    u_k_part_col0 = np.expand_dims(u_k_part_col0,3);
    u_k_part_ex = np.concatenate([u_k_part_ex,u_k_part_col0],3);

    u_k_part_ex_flip = np.flip(u_k_part_ex,2);
    u_k_part_ex_flip = np.flip(u_k_part_ex_flip,3);

    u_k_part_flip = np.delete(u_k_part_ex_flip,nx,2);
    u_k_part_flip = np.delete(u_k_part_flip,ny,3);

    return u_k_part_flip;

  def getComplex(self,a,b):
    j = np.complex(0,1); # create complex number (or use 1j).
    c = a + j*b;
    return c;

  def getRealImag(self,c):
    a = np.real(c);
    b = np.imag(c);
    return a,b;

  def computeLSymbol_laplacian_u(self):
    r"""Compute associated Fourier symbols for use under DFT for the operator L[u]."""
    num_dim = 1;nx=self.nx;ny=self.ny;
    vec_k1 = torch.zeros(nx,ny);
    vec_k2 = torch.zeros(nx,ny);    
    vec_k1_pp = torch.zeros(nx,ny);
    vec_k2_pp = torch.zeros(nx,ny);
    vec_k_sq = torch.zeros(nx,ny);
    L_symbol_real = torch.zeros(nx,ny,dtype=torch.float32);
    L_symbol_imag = torch.zeros(nx,ny,dtype=torch.float32);
    neg_four_pi_sq = -4.0*np.pi*np.pi;
    for i in range(0,nx):
      for j in range(0,ny):
        vec_k1[i,j] = i;
        vec_k2[i,j] = j;
        vec_k_sq[i,j] = vec_k1[i,j]*vec_k1[i,j] + vec_k2[i,j]*vec_k2[i,j];
        if (vec_k1[i,j] < nx/2):
          vec_k1_p = vec_k1[i,j];
        else:
          vec_k1_p = vec_k1[i,j] - nx;  
        if (vec_k2[i,j] < ny/2):
          vec_k2_p = vec_k2[i,j];
        else:
          vec_k2_p = vec_k2[i,j] - ny; 
        vec_k1_pp[i,j] = vec_k1_p;
        vec_k2_pp[i,j] = vec_k2_p;
        vec_k_p_sq = vec_k1_p*vec_k1_p + vec_k2_p*vec_k2_p;
        L_symbol_real[i,j] = neg_four_pi_sq*vec_k_p_sq;
        L_symbol_imag[i,j] = 0.0;
        
    L_hat = self.getComplex(L_symbol_real.numpy(),L_symbol_imag.numpy());
        
    return L_hat, vec_k1_pp, vec_k2_pp;

  def computeLSymbol_grad_u(self):
    r"""Compute associated Fourier symbols for use under DFT for the operator L[u]."""
    num_dim = 2;nx=self.nx;ny=self.ny;
    vec_k1 = torch.zeros(nx,ny);
    vec_k2 = torch.zeros(nx,ny);    
    vec_k1_pp = torch.zeros(nx,ny);
    vec_k2_pp = torch.zeros(nx,ny);
    vec_k_sq = torch.zeros(nx,ny);
    L_symbol_real = torch.zeros(num_dim,nx,ny,dtype=torch.float32);
    L_symbol_imag = torch.zeros(num_dim,nx,ny,dtype=torch.float32);
    two_pi = 2.0*np.pi;
    #two_pi_i = two_pi*1j; # $2\pi{i}$, 1j = sqrt(-1)
    for i in range(0,nx):
      for j in range(0,ny):
        vec_k1[i,j] = i;
        vec_k2[i,j] = j;
        vec_k_sq[i,j] = vec_k1[i,j]*vec_k1[i,j] + vec_k2[i,j]*vec_k2[i,j];
        if (vec_k1[i,j] < nx/2):
          vec_k1_p = vec_k1[i,j];
        else:
          vec_k1_p = vec_k1[i,j] - nx;  
        if (vec_k2[i,j] < ny/2):
          vec_k2_p = vec_k2[i,j];
        else:
          vec_k2_p = vec_k2[i,j] - ny; 
        vec_k1_pp[i,j] = vec_k1_p;
        vec_k2_pp[i,j] = vec_k2_p;
        vec_k_p_sq = vec_k1_p*vec_k1_p + vec_k2_p*vec_k2_p;
        L_symbol_real[0,i,j] = 0.0;
        L_symbol_imag[0,i,j] = two_pi*vec_k1_p;
        L_symbol_real[1,i,j] = 0.0;
        L_symbol_imag[1,i,j] = two_pi*vec_k2_p;
        
    L_hat_0 = self.getComplex(L_symbol_real[0,:,:].numpy(),L_symbol_imag[0,:,:].numpy());
    L_hat_1 = self.getComplex(L_symbol_real[1,:,:].numpy(),L_symbol_imag[1,:,:].numpy());

    L_hat   = np.stack((L_hat_0,L_hat_1));
       
    return L_hat, vec_k1_pp, vec_k2_pp;

  def computeCoeffActionL(self,u_hat,L_hat):
    r"""Computes the action of operator L used for data generation in Fourier space."""
    u_k_real, u_k_imag = self.getRealImag(u_hat);    
    L_symbol_real, L_symbol_imag = self.getRealImag(L_hat);
    
    f_k_real = L_symbol_real*u_k_real - L_symbol_imag*u_k_imag; #broadcast will distr over copies of u.
    #f_k_real = -1.0*f_k_real;
    f_k_imag = L_symbol_real*u_k_imag + L_symbol_imag*u_k_real;
    #f_k_imag = -1.0*f_k_imag;
    
    # Generate samples u and f using ifft2.    
    f_hat = self.getComplex(f_k_real,f_k_imag);
        
    return f_hat;

  def computeActionL(self,u,L_hat):
    r"""Computes the action of operator L used for data generation."""    
    raise Exception('Currently this routine not debugged, need to test first.')
    
    # perform FFT to get u_hat
    u_hat = np.fft.fft2(u);

    # compute action of L_hat
    f_hat = self.computeCoeffActionL(u_hat,L_hat);

    # compute inverse FFT to get f
    f = np.fft.ifft2(f_hat)

    return f;
                
  def __init__(self,op_type=r'\Delta{u}',op_params=None,
               gen_mode='exp1',gen_params={'alpha1':0.1},
               num_samples=int(1e4),nchannels=1,nx=15,ny=15,
               flag_verbose=0, **extra_params):
    r"""Setup for data generation.
    
        Args:
          op_type (str): The differential operator to sample.
          op_params (dict): The operator parameters.
          gen_mode (str): The mode for the data generator.
          gen_params (dict): The parameters for the given generator.
          num_samples (int): The number of samples to generate.
          nchannels (int): The number of channels.
          nx (int): The number of input sample points in x-direction.
          ny (int): The number of input sample points in y- direction.
          flag_verbose (int): Level of reporting during calculations.
          extra_params (dict): Extra parameters for the sampler.
                                
        For extra_params we have:
          noise_factor (float): The amount of noise to add to samples.
          scale_factor (float): A factor to scale magnitude of the samples.
          flagComputeL (bool): If the fourier symbol of operator should be computed.
          
        For generator modes we have:
          gen_mode == 'exp1': 
            alpha1 (float): The decay rate.          

        Note:
          For now, please use only nx that is odd.  In this initial implementation, we use a 
          method based on conjugated flips with formula for the odd case which is slightly
          simpler than other case.
    """
    if flag_verbose > 0:
      print("Generating the data samples which can take some time.");
      print("num_samples = %d"%num_samples);

    self.op_type=op_type;
    self.op_params=op_params;

    self.gen_mode=gen_mode;
    self.gen_params=gen_params;
    
    self.num_samples=num_samples;
    self.nchannels=nchannels;
    self.nx=nx; self.ny=ny;

    if (nx % 2 == 0) or (ny % 2 == 0) or (nx != ny): # may be able to relax nx != ny (just for safety)
      msg = "Not allowed yet to use nx,ny that are even or unequal. ";
      msg += "For now, please just use nx,ny that is odd given the flips currently used."
      raise Exception(msg);
    
    noise_factor=0;scale_factor=1.0;flagComputeL=False; # default values
    if 'noise_factor' in extra_params:
      noise_factor = extra_params['noise_factor'];

    if 'scale_factor' in extra_params:        
      scale_factor = extra_params['scale_factor'];

    if 'flagComputeL' in extra_params:        
      flagComputeL = extra_params['flagComputeL'];

    # Generate for the operator the Fourier symbols
    flag_vv = 'null';
    if self.op_type == r'\grad{u}' or self.op_type == r'u\grad{u}' or self.op_type == r'\grad{u}\cdot\grad{u}':
      L_hat, vec_k1_pp, vec_k2_pp = self.computeLSymbol_grad_u();
      flag_vv = 'vector2';
    elif self.op_type == r'\Delta{u}' or self.op_type == r'u\Delta{u}' or self.op_type == r'\Delta{u}*\Delta{u}':
      L_hat, vec_k1_pp, vec_k2_pp = self.computeLSymbol_laplacian_u();
      flag_vv = 'scalar';
    else:
      raise Exception("Unknown operator type.");
             
    if (flagComputeL):
      raise Exception("Currently not yet supported, the flagComputeL.");
      L_i = np.fft.ifft2(L_hat);
      self.L_hat = L_hat;
      self.L_i   = L_i;
      u          = np.zeros(nx,ny);
      i0         = int(nx/2);
      j0         = int(ny/2);
      u[i0,j0]   = 1.0;
      self.G_i   = self.computeActionL(u);

    # Generate random input function (want real-valued)
    # conj(u_k) = u_{N -k} needs to hold.
    u_k_real = np.random.randn(num_samples,nchannels,nx,ny);
    u_k_imag = np.random.randn(num_samples,nchannels,nx,ny);
    
    # scale modes to make smooth
    if gen_mode=='exp1':
      alpha1 = gen_params['alpha1'];
      factor_k = scale_factor*np.exp(-alpha1*(vec_k1_pp**2 + vec_k2_pp**2));
      factor_k = factor_k.numpy();
    else:
      raise Exception("Generation mode not recognized.");
       
    u_k_real = u_k_real*factor_k; # broadcast will apply over last two dimensions
    u_k_imag = u_k_imag*factor_k; # broadcast will apply over last two dimensions

    # flip modes for constructing rep of real-valued function
    u_k_real_flip = self.flipForFFT(u_k_real);
    u_k_imag_flip = self.flipForFFT(u_k_imag);
    
    u_k_real = 0.5*u_k_real + 0.5*u_k_real_flip; # make conjugate conj(u_k) = u_{N -k}
    u_k_imag = 0.5*u_k_imag - 0.5*u_k_imag_flip; # make conjugate conj(u_k) = u_{N -k}
        
    u_k_real = torch.from_numpy(u_k_real);
    u_k_imag = torch.from_numpy(u_k_imag);
    
    u_k_real = u_k_real.type(torch.float32);
    u_k_imag = u_k_imag.type(torch.float32);
                    
    u_hat = self.getComplex(u_k_real.numpy(),u_k_imag.numpy());
    if flag_vv == 'scalar':
      f_hat = self.computeCoeffActionL(u_hat,L_hat);
    elif flag_vv == 'vector2':
      f_hat_0 = self.computeCoeffActionL(u_hat,L_hat[0,:,:]);
      f_hat_1 = self.computeCoeffActionL(u_hat,L_hat[1,:,:]);
      f_hat   = np.concatenate((f_hat_0,f_hat_1),-3);
    else:
      raise Exception("Unkonwn operator type.");

    # Generate samples u and f using ifft2.
    # ifft2 is broadcast over last two indices
    # perform inverse DFT to get u and f
    u_i = np.fft.ifft2(u_hat);
    if flag_vv == 'scalar':
      f_i = np.fft.ifft2(f_hat);
    elif flag_vv == 'vector2':
      f_i_0 = np.fft.ifft2(f_hat[:,0,:,:]);
      f_i_1 = np.fft.ifft2(f_hat[:,1,:,:]);
      f_i   = np.stack((f_i_0,f_i_1),-3);
    else:
      raise Exception("Unkonwn operator type.");

    if self.op_type == r'\grad{u}':
      f_i = f_i; # nothing to do.
    elif self.op_type == r'u\grad{u}':
      f_i = u_i*f_i; # matches up by broadcast rules
    elif self.op_type == r'\grad{u}\cdot\grad{u}':
      f_i = np.sum(f_i**2,1); # sum on axis for channels, [batch,channel,nx,ny].
      f_i = np.expand_dims(f_i,1); # keep in form [batch,1,nx,ny]
    elif self.op_type == r'\Delta{u}':
      f_i = f_i;  # nothing to do.
    elif self.op_type == r'u\Delta{u}':
      f_i = u_i*f_i; 
    elif self.op_type == r'\Delta{u}*\Delta{u}':
      f_i = f_i**2; 
    else:
      raise Exception("Unkonwn operator type.");

    self.samples_X = torch.from_numpy(np.real(u_i)).type(torch.float32); # only grab real part
    self.samples_Y = torch.from_numpy(np.real(f_i)).type(torch.float32);

    if noise_factor > 0:
      self.samples_Y += noise_factor*torch.randn(*self.samples_Y.shape);

  def __len__(self):
    return self.samples_X.size()[0];

  def __getitem__(self,index):
    return self.samples_X[index],self.samples_Y[index];

  def to(self,device):
    self.samples_X = self.samples_X.to(device);
    self.samples_Y = self.samples_Y.to(device);
    return self;


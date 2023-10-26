"""
  Collection of routines helpful for visualizing results and generating figures. 
""" 

# Authors: B.J. Gross and P.J. Atzberger
# Website: http://atzberger.org/

import matplotlib;
import matplotlib as mtpl;
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;

import numpy as np;

#----------------------------
def plot_samples_u_f_1d(u_list,f_list,np_xj,np_xi,rows=4,cols=6,
                       figsize = (20*0.9,10*0.9),title="Data Samples: u, f=L[u]",
                       xlabel='x',ylabel='',
                       left=0.125,bottom=0.1,right=0.9, top=0.94,wspace=0.4,hspace=0.4,
                       fontsize=16,y=1.00,flag_draw=True,**extra_params):      

  r"""Plots a collection of data samples in a panel."""
  # --  
  fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,sharey=False);
  fig.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace);
  plt.suptitle(title, fontsize=fontsize,y=y); 

  # -- 
  I1 = 0; I2 = 0;
  for i1 in range(0,rows):
    for i2 in range(0,cols):          
      ax = axs[i1,i2];  

      if i2 % 2 == 0:
        xx = np_xj[:,0]; yy = u_list[I1].numpy()[0,:];
        #yy = yy.squeeze(0);
        ax.plot(xx,yy,'m.-');
        if i1 == 0:
          ax.set_title('u');
        I1 += 1;
      else:
        xx = np_xi[:,0]; yy = f_list[I2].numpy()[0,:];
        #yy = yy.squeeze(0);
        ax.plot(xx,yy,'r.-');
        if i1 == 0:
          ax.set_title('f');
        I2 += 1;
      
      ax.set_xlabel(xlabel);
      ax.set_ylabel(ylabel);

  if flag_draw:
    plt.draw();

def plot_samples_u_f_fp_1d(u_list,f_target_list,f_pred_list,np_xj,np_xi,rows=4,cols=6,
                           figsize = (20*0.9,10*0.9),title="Data Samples: u, f=L[u]",
                           xlabel='x',ylabel='',
                           left=0.125,bottom=0.1,right=0.9, top=0.94,wspace=0.4,hspace=0.4,
                           fontsize=16,y=1.00,flag_draw=True,**extra_params):      

  r"""Plots a collection of data samples and predictions in a panel."""
  # --  
  fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,sharey=False);
  fig.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace);
  plt.suptitle(title, fontsize=fontsize,y=y); 

  # -- 
  I1 = 0; I2 = 0;
  for i1 in range(0,rows):
    for i2 in range(0,cols):          
      ax = axs[i1,i2];  
    
      if i2 % 2 == 0:
        ax.plot(np_xj[:,0],u_list[I1].numpy()[0,:],'m.-');
        if i1 == 0:
          ax.set_title('u');
        I1 += 1;  
        ax.set_xlabel(xlabel);
        ax.set_ylabel(ylabel);
      else:                
        ax.plot(np_xj[:,0],f_target_list[I2].numpy()[0,:],'r.-');
        ax.plot(np_xj[:,0],f_pred_list[I2].numpy()[0,:],'b.-');          
        if i1 == 0:
          ax.set_title('f');
        I2 += 1;  
        ax.set_xlabel(xlabel);
        ax.set_ylabel(ylabel);
      
  if flag_draw:
    plt.draw();

def plot_dataset_diffOp1(dataset,np_xj=None,np_xi=None,rows=4,cols=6,II=None,
                         figsize=(20*0.9,10*0.9),title="Data Samples: u, f=L[u]",
                         xlabel='x',ylabel='',
                         left=0.125,bottom=0.1,right=0.9, top=0.94,wspace=0.4,hspace=0.4,
                         fontsize=16,y=1.00,flag_draw=True,**extra_params):

  r"""Plots a collection of data samples in a panel."""    
  u_list = []; f_list = []; f_pred_list = []; 
  num_samples = len(dataset);
  for I in np.arange(0,min(num_samples,int(rows*cols/2))):                
    if II is None:
      u_list.append(dataset[I][0].cpu()); # just make plain lists for convenience
      f_list.append(dataset[I][1].cpu()); 
    else:
      u_list.append(dataset[II[I]][0].cpu());
      f_list.append(dataset[II[I]][1].cpu());
      
  # --
  plot_samples_u_f_1d(u_list=u_list,f_list=f_list,np_xj=np_xj,np_xi=np_xi,
                      rows=rows,cols=cols,
                      figsize=figsize,title=title,
                      xlabel=xlabel,ylabel=ylabel,
                      left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace,
                      fontsize=fontsize,y=y,flag_draw=flag_draw,**extra_params);

#----------------------------
def plot_samples_u_f_2d(u_list,f_list,np_xj,np_xi,channelI_u=0,channelI_f=0,rows=4,cols=6,
                       figsize = (20*0.9,10*0.9),title="Data Samples: u, f=L[u]",
                       xlabel='x',ylabel='',
                       left=0.125, bottom=0.1, right=0.9, top=0.94, wspace=0.01, hspace=0.1,
                       fontsize=16,y=1.00,flag_draw=True,**extra_params):      

  r"""Plots a collection of data samples in a panel."""
  # --  
  fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,sharey=False);
  fig.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace);
  plt.suptitle(title, fontsize=fontsize,y=y); 

  # --
  for ax in axs.flatten():
    ax.axis('off');

  # -- 
  I1 = 0; I2 = 0; Ic_u = channelI_u;Ic_f = channelI_f;
  for i1 in range(0,rows):
    for i2 in range(0,cols):          
      ax = axs[i1,i2];  

      if i2 % 2 == 0:
        uu = u_list[I1][Ic_u,:,:];
        ax.imshow(uu,cmap='Blues_r');
        if i1 == 0:
          ax.set_title('u');
        I1 += 1;  
      else:                
        ff = f_list[I2][Ic_f,:,:];
        ax.imshow(ff,cmap='Purples_r');
        if i1 == 0:
          ax.set_title('f');
        I2 += 1;   

  if flag_draw:
    plt.draw();

def plot_samples_u_f_fp_2d(u_list,f_target_list,f_pred_list,np_xj,np_xi,channelI_u=0,channelI_f=0,rows=4,cols=6,
                           figsize = (20*0.9,10*0.9),title="Data Samples: u, f=L[u]",
                           xlabel='x',ylabel='',
                           left=0.125, bottom=0.1, right=0.9, top=0.94, wspace=0.01, hspace=0.1,
                           fontsize=16,y=1.00,flag_draw=True,**extra_params):      

  r"""Plots a collection of data samples and predictions in a panel."""
  # --  
  fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,sharey=False);
  fig.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace);
  plt.suptitle(title, fontsize=fontsize,y=y); 

  # --
  for ax in axs.flatten():
    ax.axis('off');

  # -- 
  Ic_u = channelI_u; Ic_f = channelI_f;
  I1 = 0; I2 = 0; I3 = 0; 
  for i1 in range(0,rows):
    for i2 in range(0,cols):          
      ax = axs[i1,i2];  

      if i2 % 3 == 0:
        uu = u_list[I1][Ic_u,:,:];
        ax.imshow(uu,cmap='Blues_r');
        if i1 == 0:
          ax.set_title('u');
        I1 += 1;  
      elif i2 % 3 == 1: 
        ff = f_pred_list[I2][Ic_f,:,:];
        ax.imshow(ff,cmap='Purples_r');
        if i1 == 0:
          ax.set_title('f:predicted');
        I2 += 1;  
      elif i2 % 3 == 2: 
        ff = f_target_list[I3][Ic_f,:,:];
        ax.imshow(ff,cmap='Purples_r');
        if i1 == 0:
          ax.set_title('f:target');
        I3 += 1;              
      
  if flag_draw:
    plt.draw();

def plot_dataset_diffOp2(dataset,np_xj=None,np_xi=None,channelI_u=0,channelI_f=0,rows=4,cols=6,II=None,
                         figsize=(20*0.9,10*0.9),title="Data Samples: u, f=L[u]",
                         xlabel='x',ylabel='',
                         left=0.125, bottom=0.1, right=0.9, top=0.94, wspace=0.01, hspace=0.1,
                         fontsize=16,y=1.00,flag_draw=True,**extra_params):



  r"""Plots a collection of data samples in a panel."""    
  u_list = []; f_list = []; f_pred_list = []; 
  num_samples = len(dataset);
  for I in np.arange(0,min(num_samples,int(rows*cols/2))):                
    if II is None:
      u_list.append(dataset[I][0].cpu()); # just make plain lists for convenience
      f_list.append(dataset[I][1].cpu()); 
    else:
      u_list.append(dataset[II[I]][0].cpu());
      f_list.append(dataset[II[I]][1].cpu());
      
  # --
  plot_samples_u_f_2d(u_list=u_list,f_list=f_list,np_xj=np_xj,np_xi=np_xi,
                      channelI_u=channelI_u,channelI_f=channelI_f,
                      rows=rows,cols=cols,
                      figsize=figsize,title=title,
                      xlabel=xlabel,ylabel=ylabel,
                      left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace,
                      fontsize=fontsize,y=y,flag_draw=flag_draw,**extra_params);

#----------------------------
def plot_images_in_array(axs,img_arr,label_arr=None,cmap=None, **extra_params):
    r"""Plots an array of images as a collection of panels."""
    
    numSamples = len(img_arr);
    sqrtS      = int(np.sqrt(numSamples));
        
    # Default values
    flag_plot_rect = False;
    list_correct_class = None;
    
    if 'list_correct_class' in extra_params:
      list_correct_class = extra_params['list_correct_class'];
      flag_plot_rect     = True;
            
    if 'flag_plot_rect' in extra_params:
      flag_plot_rect = extra_params['flag_plot_rect'];
               
    I = 0;
    for i in range(0,sqrtS):        
        for j in range(0,sqrtS):
            ax = axs[i][j];
            img = img_arr[I];            
            
            if len(img.shape) >= 3: 
              if img.shape[2] == 1: # For BW case of (Nx,Ny,1) --> (Nx,Ny), RGB has (Nx,Ny,3).
                img = img.squeeze(2);
                                        
            if cmap is not None:
              ax.imshow(img, cmap=cmap);
            else:
              ax.imshow(img);
            
            if label_arr is not None:
              ax.set_title("%s"%label_arr[I]);
            
            ax.set_xticks([]);
            ax.set_yticks([]);
                        
            if flag_plot_rect:
                
              if list_correct_class[I]:
                edge_color = 'g';
              else:
                edge_color = 'r';
                
              # draw a rectangle 
              Nx = img.shape[0]; Ny = img.shape[1];    
              rectangle = mtpl.patches.Rectangle((0,0),Nx-1,Ny-1,
                                                 linewidth=5,edgecolor=edge_color,
                                                 facecolor='none');              
              ax.add_patch(rectangle);
            
            I += 1;

def plot_image_array(img_arr,label_arr=None,title=None,figSize=(18,18),
                     title_yp=0.95,cmap="gray",**extra_params):
    r"""Plots an array of images as a collection of panels."""

    # determine number of images we need to plot
    numSamples = len(img_arr);
    sqrtS      = int(np.sqrt(numSamples));
    rows       = sqrtS;
    cols       = sqrtS;
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figSize);
        
    plot_images_in_array(axs,img_arr,label_arr,cmap=cmap,**extra_params);

    if title is None:
      plt.suptitle("Collection of Images", fontsize=18,y=title_yp);
    else:
      plt.suptitle(title, fontsize=18,y=title_yp);
    
def plot_gridspec_image_array(outer_h,img_arr,label_arr=None,title=None,figSize=(18,18),title_yp=0.95,cmap="gray",title_x=0.0,title_y=1.0,title_fsize=14):    
    r"""Plots an array of images as a collection of panels."""  
    fig = plt.gcf();
    
    sqrtS = int(np.sqrt(len(img_arr)));

    inner = gridspec.GridSpecFromSubplotSpec(sqrtS, sqrtS,
                                             subplot_spec=outer_h, wspace=0.1, hspace=0.1);
    
    # collect the axes
    axs =[];
    I = 0;
    for i in range(0,sqrtS):
      axs_r = [];
      for j in range(0,sqrtS):
        ax = plt.Subplot(fig, inner[I]);
        fig.add_subplot(ax);
        axs_r.append(ax);
        I += 1;    
        
      axs.append(axs_r);

    # plot the images
    plot_images_in_array(axs,img_arr,cmap="gray");
    
    if title is None:      
      a = 1;
    else:
      axs[0][0].text(title_x,title_y,title,fontsize=title_fsize);

#----------------------------
def save_fig(baseFilename,extraLabel,flag_verbose=0,dpi_set=200,flag_pdf=False):
  r"""Saves figures to disk."""
    
  fig = plt.gcf();  
  fig.patch.set_alpha(1.0);
  fig.patch.set_facecolor((1.0,1.0,1.0,1.0));
  
  if flag_pdf:
    saveFilename = '%s%s.pdf'%(baseFilename,extraLabel);
    if flag_verbose > 0:
      print('saveFilename = %s'%saveFilename);
    plt.savefig(saveFilename, format='pdf',dpi=dpi_set,facecolor=(1,1,1,1),alpha=1.0);

  saveFilename = '%s%s.png'%(baseFilename,extraLabel);
  if flag_verbose > 0:
    print('saveFilename = %s'%saveFilename);
  plt.savefig(saveFilename, format='png',dpi=dpi_set,facecolor=(1,1,1,1),alpha=1.0);



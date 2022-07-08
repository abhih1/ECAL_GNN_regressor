#!/usr/bin/env python
# coding: utf-8

# In[1]:




import uproot
import numpy as np
import glob
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from tqdm import tqdm as tqdm


# In[6]:


def crop_EBshower_padded(imgEB, iphi, ieta, window=128):

		assert len(imgEB.shape) == 3, '!! len(imgEB.shape): %d != 3'%len(imgEB.shape)
		assert ieta < imgEB.shape[1], '!! ieta:%d !< imgEB.shape[1]:%d'%(ieta, imgEB.shape[1])
		assert iphi < imgEB.shape[2], '!! iphi:%d !< imgEB.shape[2]:%d'%(iphi, imgEB.shape[2])

		# NOTE: image window here should correspond to the one used in RHAnalyzer
		off = window//2
		ieta = int(ieta)+1 # seed positioned at [15,15]
		iphi = int(iphi)+1 # seed positioned at [15,15]

		# ------------------------------------------------
		# ieta (row) padding
		# ------------------------------------------------
		pad_lo, pad_hi = 0, 0
		# lower padding check
		if ieta >= off:
				ieta_lo = ieta-off
		else:
				pad_lo = abs(ieta-off)
				ieta_lo = 0
		# upper padding check
		if ieta+off <= imgEB.shape[1]:
				ieta_hi = ieta+off
		else:
				pad_hi = abs(ieta+off-imgEB.shape[1])
				ieta_hi = imgEB.shape[1]

		# ------------------------------------------------
		# iphi (col) wrap-around
		# ------------------------------------------------
		# Wrap-around on left side
		if iphi < off:
				diff = off-iphi
				img_crop_ = np.concatenate((imgEB[:, ieta_lo:ieta_hi, -diff:], imgEB[:, ieta_lo:ieta_hi, :iphi+off]), axis=-1)
		# Wrap-around on right side
		elif 360-iphi < off:
				diff = off - (360-iphi)
				img_crop_ = np.concatenate((imgEB[:, ieta_lo:ieta_hi, iphi-off:], imgEB[:, ieta_lo:ieta_hi, :diff]), axis=-1)
		# Nominal case
		else:
				img_crop_ = imgEB[:, ieta_lo:ieta_hi, iphi-off:iphi+off]

		# Add ieta padding if needed
		img_crop = np.pad(img_crop_, ((0,0), (pad_lo, pad_hi), (0,0)), 'constant') # pads with 0
		assert img_crop.shape[1] == window, '!! img_crop.shape[1]:%d != window:%d'%(img_crop.shape[1], window)
		assert img_crop.shape[2] == window, '!! img_crop.shape[2]:%d != window:%d'%(img_crop.shape[2], window)

		return img_crop

def flip_img(X_img):
    X_img = X_img.squeeze()
    #print(X_img.shape)
    Down = X_img[:63,:] #Excluding [63,63]-->highest deposit
    Up= X_img[64:128,:]
    Down_avg= np.mean(Down.flatten())
    Up_avg= np.mean(Up.flatten())
    #print("Down_avg= %.4f, Up_avg= %.4f"%(Down_avg, Up_avg))
 
    if (Down_avg > Up_avg):
        X_img = np.flipud(X_img)

    Left =  X_img[:,:63]
    Right = X_img[:,64:128]
    Left_avg= np.mean(Left.flatten())
    Right_avg= np.mean(Right.flatten())
   
    if (Left_avg > Right_avg): 
        X_img = np.fliplr(X_img)

    X_img = X_img.reshape(1,128,128)        
    return(X_img)


def crop_EBshower(imgEB, iphi, ieta, window=128):

	# NOTE: image window here should correspond to the one used in RHAnalyzer
	off = window//2
	iphi = int(iphi)+1 # seed positioned at [15,15]
	ieta = int(ieta)+1 # seed positioned at [15,15]
	
	# Wrap-around on left side
	if iphi < off:
			diff = off-iphi
			img_crop = np.concatenate((imgEB[:,ieta-off:ieta+off,-diff:], imgEB[:,ieta-off:ieta+off,:iphi+off]), axis=-1)
	# Wrap-around on right side
	elif 360-iphi < off:
			diff = off - (360-iphi)
			img_crop = np.concatenate((imgEB[:,ieta-off:ieta+off,iphi-off:], imgEB[:,ieta-off:ieta+off,:diff]), axis=-1)
	# Nominal case
	else:
			img_crop = imgEB[:,ieta-off:ieta+off,iphi-off:iphi+off]
	
	return img_crop


# In[66]:


#from mpl_toolkits import mplot3d
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#%matplotlib notebook


def processinputfile(inputfile,processeddir,fidx):
    
    file = uproot.open(inputfile)
    tree = file["fevt/RHTree"]
    tokeep = tree.arrays(tokeep_labels,library="np")
    nevts = tree.num_entries
    
    #tree = file["fevt/RHTree"]
    #tree.show() # get branch details
    
    SC_iphi=tokeep["SC_iphi"]
    SC_ieta=tokeep["SC_ieta"]
    SC_mass=tokeep["SC_mass"]
    EB_energy=tokeep["EB_energy"]
    EB_RHeta=tokeep["EB_RHeta"]
    EB_RHphi=tokeep["EB_RHphi"]
    EB_RHx=tokeep["EB_RHx"]
    EB_RHy=tokeep["EB_RHy"]
    EB_RHz=tokeep["EB_RHz"]
    
    for i in tqdm(range(nevts),desc="processing event in file -->"):
        npho = len(SC_iphi[i])

        for j in range(npho):


            sciphi = SC_iphi[i][j]
            scieta = SC_ieta[i][j]
            scmass = SC_mass[i][j]
            eben = EB_energy[i].reshape(1,170,360)
            ebeta = EB_RHeta[i].reshape(1,170,360)
            ebphi = EB_RHphi[i].reshape(1,170,360)
            ebx = EB_RHx[i].reshape(1,170,360)
            eby = EB_RHy[i].reshape(1,170,360)
            ebz = EB_RHz[i].reshape(1,170,360)

            #plt.imshow(eben[0,:,:])
            #plt.show()


            eb_crop = crop_EBshower_padded(eben, sciphi, scieta)
            rh_eta  = crop_EBshower_padded(ebeta, sciphi, scieta)
            rh_phi = crop_EBshower_padded(ebphi, sciphi, scieta)
            rh_x = crop_EBshower_padded(ebx, sciphi, scieta)
            rh_y = crop_EBshower_padded(eby, sciphi, scieta)
            rh_z = crop_EBshower_padded(ebz, sciphi, scieta)

            rawvers = False
            #rawvers = True
            if rawvers:

                idata = np.stack([eb_crop.flatten(),rh_x.flatten(),rh_y.flatten(),rh_z.flatten(),rh_eta.flatten(),rh_phi.flatten()],axis=1)
                idata = idata[idata[:,0] > 0 ] 
                '''print(idata.shape)
                plt.scatter(idata[:,4],idata[:,5],s=idata[:,0])
                plt.show()'''

            else:
                centered = True
                rw,cl = np.where(np.squeeze(eb_crop,axis=0)>-999)
                if centered:
                    rwmax,clmax = np.where(np.squeeze(eb_crop,axis=0) == np.max(eb_crop))
                    #print(rwmax,clmax)
                    rw  = rw - rwmax
                    cl  = cl - rwmax

                #idata = np.stack([eb_crop.flatten(),rw,cl,rh_x.flatten(),rh_y.flatten(),rh_z.flatten(),rh_eta.flatten(),rh_phi.flatten()],axis=1)
                idata = np.stack([eb_crop.flatten(),rw,cl],axis=1)
                idata = idata[idata[:,0] > 0 ] 
                '''print(idata.shape)
                plt.scatter(idata[:,1],idata[:,2],s=idata[:,0])
                plt.show()'''



            '''plt.imshow(eben[0,:,:],cmap='gray_r')
            plt.show()
            plt.imshow(eb_crop.reshape(128,128),cmap='gray_r')
            plt.show()'''

            torch.save(Data(x = torch.tensor(idata, dtype=torch.float32),
                        pmass = torch.tensor(scmass.reshape(1,1),dtype=torch.float32),
                        pscieta = torch.tensor(scieta.reshape(1,1),dtype=torch.float32),
                        psciphi = torch.tensor(sciphi.reshape(1,1),dtype=torch.float32)),
                        #osp.join(processed_dir, 'datapi_{}_{}.pt'.format(evt,ntk)))
                        #osp.join(processed_dir, 'data_{}_{}_{}.pt'.format(idx,evt,ngun)))
                        osp.join(processeddir, 'data_{}_{}_{}.pt'.format(fidx,i,j)))
                        #'test.pt')

        #if i > 5:
        #print(time.time() - start_time)

        #    break
        #break


# In[69]:



#import time
#start_time = time.time()

###################################################################################################
###
###   MAIN PROGRAM HERE
###
###################################################################################################


path = '/data_CMS/cms/sghosh/ECALGNNDATA/RootFiles/'
dsets = os.listdir(path) 

    
out_dir = '/data_CMS/cms/sghosh/ECALGNNDATA/GRAPHS_ietaiphi/'



tokeep_labels = ["SC_iphi",
                 "SC_ieta",
                 "SC_mass",
                 "EB_energy",
                 "EB_RHeta",
                 "EB_RHphi",
                 "EB_RHx",
                 "EB_RHy",
                 "EB_RHz",
                # "EB_energyT",
                # "EB_energyZ",
                ]


for dset in dsets:
    print("processing file:",path+dset)
    
    processed_dir = out_dir+dset+'/'
    print("saving into:",processed_dir)
    
    if not os.path.exists(processed_dir):
         os.makedirs(processed_dir)
    
    
    filesindir = [path+dset+'/'+i for i in os.listdir(path+dset+'/')] 
    #print(filesindir)
    fidx = 0
    for inp in tqdm(filesindir,desc="processing files in dataset -->"):
        processinputfile(inp,processed_dir,fidx)
        fidx += 1
        #break
    
    
    





import numpy as np

def iso_integral(field,theta,mask,area_in,th_min = 300.0, th_max=400.0, nbin=101):
    # Compute the conditional sum (aka isentropic integral) of "field" binned in terms of the variables theta
    # INPUTS:
    # field: integrating variable (3D - nz,nx,ny)
    # theta: sotring variable (3D - nz,nx,ny)
    # area_in: horizontal grid area (2D - nx,ny)
    # mask: test whether the data point is to be masked (3D field - nz,nx,ny) mask = True or 1: do not include' mask = 0 or False: include
    # th_min, th_max: lowest and highest bin. any data below th_min or above th_max will be included n the first/last bin.
    # nbin: number of equally spaced bins in theta.
    # OUTPUTS: 
    # area (nz,nbin) : conditional sum of area at each level and bin
    # output (nz,nbin) : condition sum of field at each level and bin
        (nz,nx,ny) = field.shape
        dth = (th_max - th_min)/(nbin - 1)
        output = np.zeros((nz,nbin))
        area =  np.zeros((nz,nbin))
        ibin = np.minimum(np.maximum(np.floor((theta - th_min)/dth),0),nbin-2)
        weight = np.maximum(np.minimum((theta - th_min)/dth - ibin, 1.0), 0.0)
        wa2 = weight * area_in * (1-mask)
        wa = (1.0-weight) * area_in * (1-mask)
        wfa = wa * field 
        wfa2 = wa2 * field
        for i in range(nbin-1):
            area[:,i] += np.sum(np.where(ibin == i, wa, 0.0),(1,2))
            output[:,i] += np.sum(np.where(ibin == i, wfa, 0.0),(1,2))
            area[:,i+1] += np.sum(np.where(ibin == i, wa2, 0.0),(1,2))
            output[:,i+1] += np.sum(np.where(ibin == i, wfa2, 0.0),(1,2))
        return output, area
    
def iso_multi_integral(field,theta,mask,area_in,th_min = 300.0, th_max=400.0, nbin=101):
    # Compute multiple conditional sums (aka isentropic integrals) of "field" binned in terms of the variables theta
    # field: integrating variable (4D - nfield nz,nx,ny)
    # theta: sotring variable (3D - nz,nx,ny)
    # area_in: horizontal grid area (2D - nx,ny)
    # mask: test whether the data point is to be masked (3D field - nz,nx,ny) mask = True or 1: do not include' mask = 0 or False: include
    # th_min, th_max: lowest and highest bin. Any data below th_min or above th_max will be included in the first/last bin.
    # nbin: number of equally spaced bins in theta.
    # OUTPUTS: 
    # area (nz,nbin): conditional sum of area at each level and bin
    # output (nfield, nz,nbin): condition sum of field at each level and bin
        (nfield, nz,nx,ny) = field.shape
        dth = (th_max - th_min)/(nbin - 1)
        output = np.zeros((nfield,nz,nbin))
        area =  np.zeros((nz,nbin))
        ibin = np.minimum(np.maximum(np.floor((theta - th_min)/dth),0),nbin-2)
        weight = np.maximum(np.minimum((theta - th_min)/dth - ibin, 1.0), 0.0)
        wa2 = weight * area_in * (1-mask)
        wa = (1.0-weight) * area_in * (1-mask)
        wfa = wa * field 
        wfa2 = wa2 * field
        for i in range(nbin-1):
            area[:,i] += np.sum(np.where(ibin == i, wa, 0.0),(1,2))
            output[:,:,i] += np.sum(np.where(ibin == i, wfa, 0.0),(2,3))
            area[:,i+1] += np.sum(np.where(ibin == i, wa2, 0.0),(1,2))
            output[:,:,i+1] += np.sum(np.where(ibin == i, wfa2, 0.0),(2,3))
        return output, area
    
def iso_average(RHOS,RHO,SZAV,nstart = 0, nend = -1, nslice = -1, dt = 1):
    if (nslice <= 0):
        nslice = RHOS.shape[0]
    Sav = (tiny * np.expand_dims(SZav,2) + np.sum( np.reshape(RHOS[nstart:nend,:,:],(nslice,dt,nk,nbin)),1)) \
                / (tiny+np.sum(np.reshape(area[nstart:nend,:,:],(nslice,dt,nk,nbin)),1))
    return Sav

def iso_multicell_integral(field,theta,mask,area_in,th_min = 300.0, th_max=400.0, nbin=101):
    # Compute multiple conditional sums (aka isentropic intergals) of "field" binned in terms of the variables theta
    # field: integrating variable (4D - nfield nz,nx)
    # theta: sotring variable (3D - nz,nx)
    # area_in: horizontal grid area (1D - nx)
    # mask: test whether the data point is to be masked (3D field - nz,nx,ny) mask = True or 1: do not include' mask = 0 or False: include
    # th_min, th_max: lowest and highest bin. any data below th_min or above th_max will be included n the first/last bin.
    # nbin: number of equally spaced bins in theta.
    # OUTPUTS: 
    # area (nz,nbin) : conditional sum of area at each level and bin
    # output (nfield, nz,nbin) : condition sum of field at each level and bin
        (nfield, nz,ncell) = field.shape
        dth = (th_max - th_min)/(nbin - 1)
        output = np.zeros((nfield,nz,nbin))
        area =  np.zeros((nz,nbin))
        ibin = np.minimum(np.maximum(np.floor((theta - th_min)/dth),0),nbin-2)
        weight = np.maximum(np.minimum((theta - th_min)/dth - ibin, 1.0), 0.0)
        wa2 = weight * area_in * (1-mask)
        wa = (1.0-weight) * area_in * (1-mask)
        wfa = wa * field 
        wfa2 = wa2 * field
        for i in range(nbin-1):
            area[:,i] += np.sum(np.where(ibin == i, wa, 0.0),(1))
            output[:,:,i] += np.sum(np.where(ibin == i, wfa, 0.0),(2))
            area[:,i+1] += np.sum(np.where(ibin == i, wa2, 0.0),(1))
            output[:,:,i+1] += np.sum(np.where(ibin == i, wfa2, 0.0),(2))
        return output, area
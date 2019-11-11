# -------------------------------------------------------
# author        : Johannes Horak
# contact        : johannes.horak@uibk.ac.at
# -------------------------------------------------------
# this script calculates the pertubations to a background
# wind-field as predicted by linear theory.

import getopt
import xarray as xa
import pandas as pd
import sys
import glob as glob
import numpy as np
from math import radians, cos, sin, asin, sqrt
import pyfftw as pyfftw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import lib.utilities as utl

def mwrite(string):
    sys.stdout.write(string)
    sys.stdout.flush()
        
def get_topo(x,topo='witch',a0=1000,a1=20000):
    if topo == 'witch':
        h=a0*(a1**2/(x**2+a1**2))                # topography
    elif topo == 'triangle':
        if np.abs(x) <= a1/2.0:
                k = np.sign(x)
                h = -np.sign(x)*x*2.0*(a0/a1)+a0
        else:
            h = 0.0
    elif topo=="sine":
        h=a0/2.0+a0/2.0*np.sin((np.pi/a1)*(x-a1/2.0))        # sine with the minimum at domain center
    return h

topo_file    = None
zero_borders = False
Nz           = 100
dz           = 200.0
U            = 20.0
N            = 0.01
V            = 0.0
f            = 0.0 #000001
parsonly     = False
topo_a0 = None
topo_a1 = None
# READ COMMAND LINE OPTIONS
try:
    opts, args = getopt.getopt(sys.argv[1:],"h",["topo=","zeroborders","help","nz=","dz=","U=","N=","a0=","a1=","parsonly"])
    if len(opts)==0:
        print("  error occured - no parameters given!")
        sys.exit(1)
except getopt.GetoptError:
    print("  error occured!")
    sys.exit(2)
for opt, arg in opts:
    #print opt," ",arg
    if opt in ("-h","--help"):
        print("  help not yet available!")
        sys.exit()
    elif opt in ("--topo"):
        topo_file=arg
    elif opt in ("--zeroborders"):
        zero_borders=True
    elif opt in ("--nz"):
        Nz = int(arg)
    elif opt in ("--dz"):
        dz = float(arg)
    elif opt in ("--U"):
        U  = float(arg)
    elif opt in ("--N"):
        N  = float(arg)
    elif opt in ("--a1"):
        topo_a1  = float(arg)
    elif opt in ("--a0"):
        topo_a0 = float(arg)

    elif opt in ("--parsonly"):
        parsonly = True

#if topo_file is None:
#    print("  error: no topography file specified!")


print('Nz {:n}'.format(Nz))
print('dz {:n}'.format(dz))
print('U  {:2.0f}'.format(U))
print('N  {:2.4f}'.format(N))

# create the topography


Nx_in = 404
Ny_in = 5
dx    = 2000.0
dy    = 2000.0
N2    = N**2

lonc = 90.0
latc = 0.0

topo_in = np.zeros(Ny_in*Nx_in).reshape(Ny_in,Nx_in)

for nx in range(Nx_in):
    x = (nx-Nx_in/2.0)*dx
    topo_in[:,nx] = get_topo(x,topo='witch',a0=topo_a0,a1=topo_a1)

ddeg = utl.find_angle_for_dx(dx=dx,lonc=lonc,latc=latc)

lon = (np.arange(0,Nx_in,1.0)-np.floor(Nx_in/2.0))*ddeg+lonc
lat = (np.arange(0,Ny_in,1.0)-np.floor(Ny_in/2.0))*ddeg+latc

Lx_in    = dx*Nx_in
Ly_in    = dy*Ny_in
Lz    = dz*Nz
xs_in    = (np.arange(0,Nx_in)-np.floor(Nx_in/2.0))*dx/1000.0
ys_in    = (np.arange(0,Ny_in)-np.floor(Ny_in/2.0))*dy/1000.0
lvls    = (np.arange(0,Nz)+0.5)*dz

filenames    = "{:s}_N{:0.4f}_U{:3.1f}_V{:3.1f}".format('witch',N,U,V)    # filename template for plots

lambda_z     = 2*np.pi*U/N                                        # calculate vertical wavelength
scorer2      = N**2/U**2
k2           = (1/topo_a1)**2
regime_f     = np.sqrt(scorer2/k2)
Fr           = U/(N*topo_a0)
regime       = ''

if regime_f <= 0.1:
    regime="irrotational_flow"
elif regime_f > 0.1 and regime_f <= 1:
    regime="evanescent_flow"
elif regime_f > 1 and regime_f <= 10:
    regime="vert_prop_waves"
elif regime_f > 10:
    regime="hydrostatic_waves"


print("  * some variables")
print('      Froude number           : {:2.2f}'.format(Fr))
print("      half width of topography: {:4.0f}m".format(topo_a1))
print("      vertical wavelength     : {:4.0f}m".format(lambda_z))
print("      scorer parameter        : {:f}".format(np.sqrt(scorer2)))
print("      k parameter             : {:f}".format(np.sqrt(k2)))
print("      regime parameter        : {:f}".format(regime_f))
print("      regime                  : {:s}".format(regime))

if parsonly:
    sys.exit(1)



print("  * not forcing topography to zero at boundaries")
topo     = topo_in.copy()
topo_mod = topo_in.copy()

Nx = Nx_in
Ny = Ny_in
Lx = Lx_in
Ly = Ly_in

xs = xs_in
ys = ys_in

topo_fft         = np.fft.fft2(topo_mod)
topo_fft_shifted = np.fft.fftshift(topo_fft)




# set the wave numbers
# luckily numpy can do the for us. The coefficient 2*pi comes from the
# Fourier transformation applied in Smith 2003:
# Smith, 2003 - A linear upslope time-delay model for orographic precipitation
fx_raw = np.fft.fftfreq(Nx,dx)
fy_raw = np.fft.fftfreq(Ny,dy)

k_raw = 2*np.pi*fx_raw
l_raw = 2*np.pi*fy_raw

'''
# ----------------------------------------------------------------------
# Here's the code to calculate the wave-numbers manually:
# This will eventually be removed from the source
# ----------------------------------------------------------------------
dkx=2.0*np.pi/Lx
dky=2.0*np.pi/Ly

k_raw=np.empty(Nx)
l_raw=np.empty(Ny)

Ncxh=Nx/2+1
Ncyh=Ny/2+1

for i in range(0,Nx):
    if i<Ncxh:
        k_raw[i]=dkx*(i)
    else:
        k_raw[i]=-dkx*(Nx+1-i)

for i in range(0,Ny):
    if i<Ncyh:
        l_raw[i]=dky*(i)
    else:
        l_raw[i]=-dky*(Ny+1-i)

k_raw=k_raw
l_raw=l_raw
'''


# find position closest to what vertical wavelenght is:
# used for later plots
lambda_z_i = -1
for z_i, z in enumerate(lvls):
    if z_i < len(lvls)-1:
        if np.abs(lambda_z-lvls[z_i+1]) > np.abs(lambda_z-z):
            lambda_z_i = z_i
            break
        else:
            pass

print("  * calculating linear solution...")
field_z      = np.zeros(Nx*Ny*Nz).reshape(Nz,Ny,Nx)
field_w_r    = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_u_r    = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_v_r    = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_p_r    = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_eta_r  = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_m_re   = np.empty(Nx*Ny).reshape(Ny,Nx)
field_m_im   = np.empty(Nx*Ny).reshape(Ny,Nx)

field_u_real = np.empty(Nx*Ny*Nz).reshape(Nz,Ny,Nx)
field_w_real = np.empty(Nx*Ny*Nz).reshape(Nz,Ny,Nx)

SMALL_VALUE = 10**-15

for k in range(0,Nz):
    mwrite("     working on z level {:n}/{:n}\r".format(k+1,Nz))

    for i in range(0,Nx):
        for j in range(0,Ny):
            z = dz*(0.5+k)+topo_mod[j,i]                                # topography following coordinate system - seems to work for above zero as well though.
            
            field_z[k,j,i] = z 
            
            kl2  = (k_raw[i]**2+l_raw[j]**2)
            if kl2 < SMALL_VALUE:
                kl2 = SMALL_VALUE
            
            sigma     = (U*k_raw[i]+V*l_raw[j])                         # sigma can be zero, approximate it by a small value if that's the case
            if sigma < SMALL_VALUE:
                sigma = SMALL_VALUE # small value, same as in icar

            m2_nom    = (N2-sigma**2)*kl2                               # equation (A12, nominator)
            m2_denom  = (sigma**2-f**2)                                 # equation (A12, demonimator)

            if m2_denom == 0.:                                           # this shouldn't happen, we check nonetheless
                print("m**2 : denominator was zero")
                print("sigma: ",sigma)
                print("nom  : ",m2_nom)
                print(m2_nom/(10**-15))
                sys.exit(1)
            else:
                m2    = m2_nom/m2_denom

            # Have to look this up again - correct root necessary to correspond to decay or radiation BC
            # See Smith 2004 (a linear theory of orography precipitation)
            if l_raw[j] < SMALL_VALUE:                                                          
                m = np.sign(k_raw[i])*np.sqrt(N2)/U                     # equation (14) in Smith 2004
            elif N2 > sigma**2:                                          
                m = np.sqrt((N2/sigma**2)*kl2)*np.sign(sigma)           # equation (13) in Smith 2004
                    
                if np.isnan(m):                                         # this should not happen, but if it does we quit with an error
                    print("error, not sure how this happened!")
                    print("had m**2 = ",m2)
                    print(" nominator  : ",m2_nom)
                    print(" denominator: ",m2_denom)
                    print("k    :",k_raw[i])
                    print("l    :",l_raw[j])
                    print("sigma: ",sigma)
                    sys.exit(1)
            elif N2 < sigma**2:                
                m = 1j*np.sqrt(kl2)                                     # see text before equation (13) in Smith 2004
            else:
                m = 0.

            eta_r = topo_fft[j,i]*np.exp(1j*m*z)                        # equation (A6) - parcel displacement calculation,
            
            
            if (np.real(m) < SMALL_VALUE) & (np.imag(m) < SMALL_VALUE):
                m = SMALL_VALUE
                
            w_r = 1j*sigma*eta_r
            p_r = (1j*eta_r/m)*(N2-sigma**2)
            
            u_r = -(m*(sigma*k_raw[i])*1j*eta_r)/kl2
            v_r = -(m*(sigma*l_raw[j])*1j*eta_r)/kl2
            
            
            field_m_re[j,i]=np.real(m)
            field_m_im[j,i]=np.imag(m)
            field_eta_r[k,j,i]=eta_r
            field_w_r[k,j,i]=w_r
            field_u_r[k,j,i]=u_r
            field_v_r[k,j,i]=v_r
            field_p_r[k,j,i]=p_r


print("")
print("  * performing inverse fourier transforms")
print('     transforming eta')
field_eta_complex     = np.fft.ifft2(field_eta_r)
field_eta_real        = np.real(field_eta_complex)
print('     transforming u')
field_u_complex        = np.fft.ifft2(field_u_r)
field_u_real        = np.real(field_u_complex)
#field_u_real        = correct_y_axis(field_u_real)
print('     transforming v')
field_v_complex        = np.fft.ifft2(field_v_r)
field_v_real        = np.real(field_v_complex)
print('     transforming w')
field_w_complex        = np.fft.ifft2(field_w_r)
field_w_real        = np.real(field_w_complex)
print('     transforming p')
field_p_complex        = np.fft.ifft2(field_p_r)
field_p_real        = np.real(field_p_complex)
#field_w_real        = correct_y_axis(field_w_real)


print(' maximum updraft  : {:2.1f}'.format(np.max(field_w_real)))
print(' maximum downdraft: {:2.1f}'.format(np.min(field_w_real)))
'''
print "  * setting everything below topography to nan"

for k in range(0,Nz):
    for i in range(0,Nx):
        for j in range(0,Ny):
            if np.isnan(topo[j,i]):
                print i,' ',j,' ',topo[j,i]
            else:
                if topo[j,i] > lvls[k]:
                    field_w_real[k,j,i]=np.nan
                    field_u_real[k,j,i]=np.nan
                    field_v_real[k,j,i]=np.nan
'''
        
print("  * creating and writing output netcdf")

output_ds    = xa.Dataset(
                        data_vars={
                                'par_N':N,
                                'par_U':U,
                                'par_V':V,
                                'par_a0':topo_a0,
                                'par_a1':topo_a1,
                                'par_nz':Nz,
                                'par_dz':dz,
                                'par_Fr':Fr,
                                'par_scorer':np.sqrt(scorer2),
                                'U':(['bottom_top','south_north','west_east'],field_u_real+U),
                                'V':(['bottom_top','south_north','west_east'],field_v_real+V),
                                'w':(['bottom_top','south_north','west_east'],field_w_real),
                                'u':(['bottom_top','south_north','west_east'],field_u_real),
                                'v':(['bottom_top','south_north','west_east'],field_v_real),
                                'p':(['bottom_top','south_north','west_east'],field_p_real),
                                'z':(['bottom_top','south_north','west_east'],field_z),
                                'w_real':(['bottom_top','south_north','west_east'],np.abs(field_w_r)),
                                'pr_re':(['bottom_top','south_north','west_east'],np.real(np.fft.fftshift(field_p_r))),
#                                'pr_im':(['bottom_top','south_north','west_east'],np.imag(np.fft.fftshift(field_p_r))),
                                'eta':(['bottom_top','south_north','west_east'],field_eta_real),
                                'fx':(['west_east'],fx_raw),
                                'k':(['west_east'],k_raw),
                                'l':(['south_north'],l_raw),
#                                'm_re':(['south_north','west_east'],field_m_re),
#                                'm_im':(['south_north','west_east'],field_m_im),
                                'topo':(['south_north','west_east'],topo),
                                'topo_mod':(['south_north','west_east'],topo_mod),
#                                'topo_im':(['south_north','west_east'],np.real(np.fft.fftshift(topo_fft))),
                                'topo_re':(['south_north','west_east'],np.imag(np.fft.fftshift(topo_fft)))
                        },
                        coords={
                                'x':xs_in,
                                'y':ys_in,
                                'bottom_top' :np.arange(0.5,Nz+0.5,1.0)*dz,
                                'south_north':np.arange(0,Ny,1.0),
                                'west_east'  :np.arange(0,Nx,1.0)
                        }
                )
                
output_ds.attrs['info']='generated with calinsol\ninput parameters stored as coordinate and dimensionless variables with prefix par_'

output_ds.to_netcdf("./output.nc",format='NETCDF4')    

# plot some quantities
y_cross=int(np.floor(Ny_in/2))

U_field = (field_u_real+U)[:,y_cross,:]
V_field = (field_v_real+V)[:,y_cross,:]

plt.clf()
plt.pcolormesh(xs,lvls/1000.0,field_w_real[:,y_cross,:],cmap='bwr')
plt.colorbar()
plt.fill_between(xs, 0, topo_mod[y_cross,:]/1000.0,facecolor='white', interpolate=True)
plt.plot(xs,topo_mod[y_cross,:]/1000.0,lw=1,c="black")
plt.title('w field')
plttext=plt.text(xs[0], lvls[-2]/1000.0, "regime: {:s}\nU={:2.1f}m/s\nV={:2.1f}m/s\nN={:0.4f}/s".format(regime,U,V,N),verticalalignment='top')
plttext.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black'))
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
plt.grid()
#plt.xlim(50,150)
plt.savefig("./plot_{:s}_w_field.png".format(filenames),dpi=300)

plt.close()

plt.clf()
plt.pcolormesh(xs,lvls/1000.0,U_field,cmap='Blues')
plt.colorbar()
plt.fill_between(xs, 0, topo_mod[y_cross,:]/1000.0,facecolor='white', interpolate=True)
plt.plot(xs,topo_mod[y_cross,:]/1000.0,lw=1,c="black")
plt.title('U field')
plttext=plt.text(xs[0], lvls[-2]/1000.0, "regime: {:s}\nU={:2.1f}m/s\nV={:2.1f}m/s\nN={:0.4f}/s".format(regime,U,V,N),verticalalignment='top')
plttext.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black'))
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
plt.grid()
#plt.xlim(50,150)
plt.savefig("./plot_{:s}_U_field.png".format(filenames),dpi=300)
plt.close()

plt.clf()
plt.pcolormesh(xs,lvls/1000.0,field_p_real[:,y_cross,:],cmap='Blues')
plt.colorbar()
plt.fill_between(xs, 0, topo_mod[y_cross,:]/1000.0,facecolor='white', interpolate=True)
plt.plot(xs,topo_mod[y_cross,:]/1000.0,lw=1,c="black")
plt.title('pressure perturbation field')
plttext=plt.text(xs[0], lvls[-2]/1000.0, "regime: {:s}\nU={:2.1f}m/s\nV={:2.1f}m/s\nN={:0.4f}/s".format(regime,U,V,N),verticalalignment='top')
plttext.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black'))
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
plt.grid()
#plt.xlim(50,150)
plt.savefig("./plot_{:s}_p_field.png".format(filenames),dpi=300)
plt.close()

plt.clf()
plt.pcolormesh(xs,lvls/1000.0,V_field,cmap='Blues')
plt.colorbar()
plt.fill_between(xs, 0, topo_mod[y_cross,:]/1000.0,facecolor='white', interpolate=True)
plt.plot(xs,topo_mod[y_cross,:]/1000.0,lw=1,c="black")
plt.title('V field')
plttext=plt.text(xs[0], lvls[-2]/1000.0, "regime: {:s}\nU={:2.1f}m/s\nV={:2.1f}m/s\nN={:0.4f}/s".format(regime,U,V,N),verticalalignment='top')
plttext.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black'))
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
plt.grid()
#plt.xlim(50,150)
plt.savefig("./plot_{:s}_V_field.png".format(filenames),dpi=300)
plt.close()
plt.clf()

z_start = lambda_z_i
while z_start > 2:
    z_start-=2
dnz=int(Nz/15.0)
for z_i in range(z_start,Nz,dnz):
    streamline=(field_eta_real[z_i,y_cross,:]+lvls[z_i])/1000.0
    baseline=np.ones(len(streamline))*lvls[z_i]/1000.0
    plt.fill_between(xs,streamline, baseline, facecolor="none", hatch="|", edgecolor="gray", linewidth=0.0, interpolate=True)
    plt.plot(xs, streamline,color='gray',lw=1)
    plt.plot([xs[0],xs[-1]], [lvls[z_i]/1000.0,lvls[z_i]/1000.0],color='gray',lw=1)
    
#plt.contour(xs,lvls/1000.0,field_eta_real[:,y_cross,:])
plt.fill_between(xs, 0, topo_mod[y_cross,:]/1000.0,facecolor='white', interpolate=True)
plt.plot(xs,topo_mod[y_cross,:]/1000.0,lw=1,c="black")
plt.title('eta field')
plttext=plt.text(xs[0], lvls[-2]/1000.0, "regime: {:s}\nU={:2.1f}m/s\nV={:2.1f}m/s\nN={:0.4f}/s".format(regime,U,V,N),verticalalignment='top')
plttext.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black'))
plt.xlabel('distance (km)')
plt.ylabel('altitude (km)')
#plt.xlim(50,150)
plt.ylim(0,lvls[-1]/1000.0)
plt.savefig("./plot_{:s}_eta_field.png".format(filenames),dpi=300)
plt.close()


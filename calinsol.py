# -------------------------------------------------------
# author		: Johannes Horak
# contact		: johannes.horak@uibk.ac.at
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

def mwrite(string):
	sys.stdout.write(string)
	sys.stdout.flush()

topo_file		= None
zero_borders	= False
Nz				= 20
dz				= 1000.0
U				= 10.0
N				= 0.01
V				= 0.0
f				= 0.0000001#1.454*10**(-4)

# READ COMMAND LINE OPTIONS
try:
	opts, args = getopt.getopt(sys.argv[1:],"h",["topo=","zeroborders","help","nz=","dz=","U=","N="])
	if len(opts)==0:
		print "  error occured - no parameters given!"
		sys.exit(1)
except getopt.GetoptError:
	print "  error occured!"
	sys.exit(2)
for opt, arg in opts:
	#print opt," ",arg
	if opt in ("-h","--help"):
		print "  help not yet available!"
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

if topo_file is None:
	print "  error: no topography file specified!"


in_file	= xa.open_dataset(topo_file)
topo_in	= in_file.HGT_M.values
lon		= in_file.XLONG_M
lat		= in_file.XLAT_M


dx=1000.0
dy=1000.0
N2=N**2


topo_file_name	= topo_file.split('/')[-1].split('.nc')[0]							# extract filename from full path
topo_a1			= float(topo_file_name.split('a1_')[1].split('km')[0])*1000.0		# extract half-width from filename (requires filename to contain a1_)
																					# better way would be to save it in nc-file.

filenames		= "{:s}_N{:0.4f}_U{:3.1f}_V{:3.1f}".format(topo_file_name,N,U,V)		# filename template for plots

lambda_z		= 2*np.pi*U/np.sqrt(N2)										# calculate vertical wavelength
scorer2			= N**2/U**2
k2				= (1/topo_a1)**2
regime_f		= np.sqrt(scorer2/k2)

if regime_f <= 0.1:
	regime="irrotational_flow"
elif regime_f > 0.1 and regime_f <= 1:
	regime="evanescent_flow"
elif regime_f > 1 and regime_f <= 10:
	regime="vert_prop_waves"
elif regime_f > 10:
	regime="hydrostatic_waves"



print "  * some variables"
print "      half width of topography: {:4.0f}m".format(topo_a1)
print "      vertical wavelength     : {:4.0f}m".format(lambda_z)
print "      scorer parameter        : {:f}".format(np.sqrt(scorer2))
print "      k parameter             : {:f}".format(np.sqrt(k2))
print "      regime parameter        : {:f}".format(regime_f)
print "      regime                  : {:s}".format(regime)
Nx_in	= topo_in.shape[1]
Ny_in	= topo_in.shape[0]
Lx_in	= dx*Nx_in
Ly_in	= dy*Ny_in
Lz		= dz*Nz
xs_in	= np.arange(0,Nx_in)*dx/1000.0
ys_in	= np.arange(0,Ny_in)*dy/1000.0
lvls	= (np.arange(0,Nz)+0.5)*dz

# force topography to zero towards the boundaries
# very experimental and actually makes matters worse at the moment
if zero_borders:
	print "  * forcing topography to zero at boundaries"

	maxd=90.0
	mind=100.0

	Nx=Nx_in+200
	Ny=Ny_in+200
	Lx=dx*Nx
	Ly=dy*Ny
	xs	= np.arange(0,Nx)*dx/1000.0
	ys	= np.arange(0,Ny)*dy/1000.0


	topo=np.empty(Ny*Nx).reshape(Ny,Nx)
	topo[100:301,100:301]=topo_in.copy()
	topo_mod = np.empty(Nx*Ny).reshape(Ny,Nx)


	for i in range(0,Nx):
		for j in range(0,Ny):
			ri=i-Nx/2.0
			rj=j-Ny/2.0
			d=np.sqrt(ri**2+rj**2)
			if d>maxd:	
				# find j,i that correspond to the last undampened topography
				lbda	= 1.0-np.sqrt(1+(maxd**2-d**2)/d**2)
				i_c	= int(np.round(ri-lbda*ri+Nx/2.0))
				j_c	= int(np.round(rj-lbda*rj+Ny/2.0))
				x	= d-maxd
				frac= x/mind
				if frac > 1.0:
					frac = 1.0
				#print x,' ',d,' ',maxd
				#print frac
				topo_mod[j,i]=topo[j_c,i_c]-frac*topo[j_c,i_c]
				#topo_mod[j,i]=topo[j_c,i_c]*np.exp(-(d-maxd)/12.0)
				#print i, ' ',j
				#print i_c, ' ',j_c
				#print np.sqrt((i_c-Nx/2.0)**2+(j_c-Ny/2.0)**2)
				#print d
				#print lbda
				#sys.exit(0)
			else:
				topo_mod[j,i]=topo[j,i]			
else:
	print "  * not forcing topography to zero at boundaries"
	topo	 = topo_in.copy()
	topo_mod = topo_in.copy()
	
	Nx = Nx_in
	Ny = Ny_in
	Lx = Lx_in
	Ly = Ly_in
	
	xs = xs_in
	ys = ys_in

topo_fft		 = np.fft.fft2(topo_mod)
topo_fft_shifted = np.fft.fftshift(topo_fft)




# calculate wave numbers
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

fx_raw = np.fft.fftfreq(Nx,dx)
fy_raw = np.fft.fftfreq(Ny,dy)

# find position closest to what vertical wavelenght is:
lambda_z_i = -1
for z_i, z in enumerate(lvls):
	#print z_i, " ",z, " ",(lambda_z-lvls[z_i+1])," ",(lambda_z-z)
	if z_i < len(lvls)-1:
		if np.abs(lambda_z-lvls[z_i+1]) > np.abs(lambda_z-z):
			lambda_z_i = z_i
			break
		else:
			pass

print "  * calculating linear solution..."
field_w_r = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_u_r = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_v_r = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_p_r = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_eta_r = np.empty(Nx*Ny*Nz,dtype=np.complex_).reshape(Nz,Ny,Nx)
field_m_re = np.empty(Nx*Ny).reshape(Ny,Nx)
field_m_im = np.empty(Nx*Ny).reshape(Ny,Nx)

field_u_real = np.empty(Nx*Ny*Nz).reshape(Nz,Ny,Nx)
field_w_real = np.empty(Nx*Ny*Nz).reshape(Nz,Ny,Nx)


for k in range(0,Nz):
	mwrite("     working on z level {:n}/{:n}\r".format(k+1,Nz))
	z = dz*(0.5+k)
	for i in range(0,Nx):
		for j in range(0,Ny):
			# only if different N for different levels:
			
			if z <= 4000:
				N2 = (0.01047)**2
			else:
				N2 = (0.004188)**2
			
			sigma	= (U*k_raw[i]+V*l_raw[j])
			m2_nom	= (N2-sigma**2)*(k_raw[i]**2+l_raw[j]**2)
			m2_denom= (sigma**2-f**2)
			
			if m2_denom == 0:
				m2	= 0
				print "m denominator was zero"
			else:
				m2	= m2_nom/m2_denom

			if m2 == np.inf:
				m = np.inf
			else:
				if N2 > sigma**2:
					if m2 < 0:
						m=1j*np.sqrt(-m2)*np.sign(sigma)
					else:
						m=np.sqrt(m2)*np.sign(sigma)
					if np.isnan(m):
						print "had ",m2
						print "nom: ",m2_nom
						print "denom: ",m2_denom
						print "k=",k_raw[i]
						print "l=",l_raw[j]
						print "sigma=",sigma
						sys.exit(0)
				elif sigma**2 <= N2:
					m=1j*np.sqrt(m2)
			
			eta_r=topo_fft[j,i]*np.exp(1j*m*z)

			if m == 0:
				w_r=1j*sigma*eta_r
				p_r=0 #1j*np.inf
				u_r=0.0
				v_r=0.0
			elif k_raw[i]**2+l_raw[j]**2 == 0:
				w_r=1j*sigma*eta_r
				p_r=(1j*eta_r/m)*(N2-sigma**2)
				if np.isnan(p_r):
					sys.exit(2)
				u_r=-1j*np.inf
				v_r=-1j*np.inf
			elif m == np.inf:
				w_r = np.inf
				u_r = np.inf
				v_r = np.inf
				p_r = 0
			else:
				w_r=1j*sigma*eta_r
				p_r=(1j*eta_r/m)*(N2-sigma**2)
				u_r=-(m*(sigma*k_raw[i]-1j*l_raw[j]*f)*1j*eta_r)/(k_raw[i]**2+l_raw[j]**2)
				v_r=-(m*(sigma*l_raw[j]+1j*k_raw[i]*f)*1j*eta_r)/(k_raw[i]**2+l_raw[j]**2)
			field_m_re[j,i]=np.real(m)
			field_m_im[j,i]=np.imag(m)
			field_eta_r[k,j,i]=eta_r
			field_w_r[k,j,i]=w_r
			field_u_r[k,j,i]=u_r
			field_v_r[k,j,i]=v_r
			field_p_r[k,j,i]=p_r


print ""
print "  * performing inverse fourier transforms"
print '     transforming eta'
field_eta_complex	= np.fft.ifft2(field_eta_r)
field_eta_real		= np.real(field_eta_complex)
print '     transforming u'
field_u_complex		= np.fft.ifft2(field_u_r)
field_u_real		= np.real(field_u_complex)
#field_u_real		= correct_y_axis(field_u_real)
print '     transforming v'
field_v_complex		= np.fft.ifft2(field_v_r)
field_v_real		= np.real(field_v_complex)
print '     transforming w'
field_w_complex		= np.fft.ifft2(field_w_r)
field_w_real		= np.real(field_w_complex)
print '     transforming p'
field_p_complex		= np.fft.ifft2(field_p_r)
field_p_real		= np.real(field_p_complex)
#field_w_real		= correct_y_axis(field_w_real)

	

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
		
print "  * creating and writing output netcdf"
		
output_ds	= xa.Dataset(
							data_vars={
								'U':(['bottom_top','south_north','west_east'],field_u_real+U),
								'V':(['bottom_top','south_north','west_east'],field_v_real+V),
								'w':(['bottom_top','south_north','west_east'],field_w_real),
								'u':(['bottom_top','south_north','west_east'],field_u_real),
								'v':(['bottom_top','south_north','west_east'],field_v_real),
								'p':(['bottom_top','south_north','west_east'],field_p_real),
								'w_real':(['bottom_top','south_north','west_east'],np.abs(field_w_r)),
								'pr_re':(['bottom_top','south_north','west_east'],np.real(np.fft.fftshift(field_p_r))),
								'pr_im':(['bottom_top','south_north','west_east'],np.imag(np.fft.fftshift(field_p_r))),
								'eta':(['bottom_top','south_north','west_east'],field_eta_real),
								'fx':(['west_east'],fx_raw),
								'k':(['west_east'],k_raw),
								'm_re':(['south_north','west_east'],field_m_re),
								'm_im':(['south_north','west_east'],field_m_im),
								'topo':(['south_north','west_east'],topo),
								'topo_mod':(['south_north','west_east'],topo_mod),
								'topo_im':(['south_north','west_east'],np.real(np.fft.fftshift(topo_fft))),
								'topo_re':(['south_north','west_east'],np.imag(np.fft.fftshift(topo_fft)))
								#'lon':(['south_north','west_east'],lon),
								#'lat':(['south_north','west_east'],lat)
							},
							coords={
								'bottom_top':np.arange(0.5,Nz+0.5,1.0)*dz,
								'south_north':np.arange(0,Ny,1.0),
								'west_east':np.arange(0,Nx,1.0)
							}
						)
output_ds.to_netcdf("./output.nc",format='NETCDF4')	

# plot some quantities
y_cross=100

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


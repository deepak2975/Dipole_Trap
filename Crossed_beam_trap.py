import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from sympy import symbols
from shapely.geometry import Polygon
from matplotlib import cm

# Constants
U0 = -1.826e-36 #Dipole factor multiplies by intensity calculated for rubidium-87 for 1064nm dipole trap(Ref:All-optical 87Rb Bose-Einstein condensate apparatus: Construction and operation Dipl. phys. I. L. H. Humbert) page 31 last paragraph
p1 = 1  #Power in the first Beam
p2 = 1  #Power in the second Beam
W1x_0 = 23.09e-6  #Beam waist of the first beam in x direction
W1z_0 = 23.09e-6    #Beam waist of the first beam in z direction
W2x_0 = 76.97e-6    #Beam waist of the second beam in x direction(beam is rotated by 90 so this direction is becomes y)
W2z_0 = 20e-6    #Beam waist of the first beam in z direction
x0 = y0 = z0 = 0
theta =90*np.pi/180  # radians
lambda_laser = 1064e-9  # wavelength in meters
m = 87 * 1.66e-27  # mass of Rb in kg
g = 9.81  # gravity in m/s^2
kb= 1.380649*10**(-23) 

# Rayleigh lengths
xR1 = np.pi * W1x_0**2 / lambda_laser
zR1 = np.pi * W1z_0**2 / lambda_laser
xR2 = np.pi * W2x_0**2 / lambda_laser
zR2 = np.pi * W2z_0**2 / lambda_laser

print(f"Rayleigh range of the first beam in x direction(mm) = {xR1*10**3}")
print(f"Rayleigh range of the first beam in z direction(mm) = {zR1*10**3}")
print(f"Rayleigh range of the second beam in y direction(mm) = {xR2*10**3}")
print(f"Rayleigh range of the second beam in z direction(mm) = {zR2*10**3}")

# Function definition
def U(x, y, z, P1=p1, P2=p2, w1x_0=W1x_0, w1z_0=W1z_0, w2x_0=W2x_0, w2z_0=W2z_0):
    # Calculate beam waists for Beam 1
    w1x = w1x_0 * np.sqrt(1 + ((x - x0)**2) / xR2)
    w1z = w1z_0 * np.sqrt(1 + ((z - z0)**2) / zR2)

    term1 = ((2 * P1 * np.exp(-2 * ((x - x0)**2 / (w1x**2 * (1 + (x - x0)**2 / xR1)) + (z - z0)**2 / (w1z**2 * (1 + (z - z0)**2 / zR1))))) /
             (np.pi * w1x * w1z))

    # Rotate coordinates for Beam 2
    rotated_x = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
    rotated_y = (y - y0) * np.cos(theta) + (x - x0) * np.sin(theta)

    # Calculate beam waists for Beam 2
    w2x = w2x_0 * np.sqrt(1 + (rotated_x**2) / xR2)
    w2z = w2z_0 * np.sqrt(1 + ((z-z0)**2) / zR2)

    # Gaussian beam potential for Beam 2
    term2 = ((2 * P2 * np.exp(-2 * ((rotated_x**2) / (w2x**2 * (1 + rotated_x**2 / xR2)) + (Z**2) / (w2z**2 * (1 + rotated_x**2 / zR2))))) /
            (np.pi * w2x * w2z))

    # Gravity term
    gravity_term = m * g * z

    return (U0 * (term1 + term2) + gravity_term)


# Generate 3D grid
x = np.linspace(-50e-6, 50e-6, 200)  # Zoom in on -100 to 100 microns, 100 points
y = np.linspace(-50e-6, 50e-6, 200)
z = np.linspace(-50e-6, 50e-6, 200)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# 2D slice plots
z_slice = int(len(z) / 2)  # Middle slice along z
y_slice = int(len(y) / 2)  # Middle slice along y
x_slice = int(len(x) / 2)  # Middle slice along x

# Function to find local maxima in an array
def Max(array):
    try:
        Maxima_indices = find_peaks(array)[0]
        # print("Local Maxima indices:", Maxima_indices)
        # print("Local Maxima values:", array[Maxima_indices])
    except Exception as e:
        print(f"Error occured (increase your space to capture peaks like maxima and minima): {e}")
    
    return Maxima_indices

# Function to find local minima in an array
def Min(array):
    Minima_indices = find_peaks(-array)[0]
    # print("Local Minima indices:", Minima_indices)
    # print("Local Minima values:", array[Minima_indices])
    return Minima_indices

def plot_3d():
    # Evaluate the function
    U_values = U(X, Y, Z, p1, p2)

    # U values divided by kb for equivalent temprature 
    U_norm = U_values / kb

    # 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the beam in 3D space
    sc = ax.scatter(X.flatten() * 1e6, Y.flatten() * 1e6, Z.flatten() * 1e6, c=U_norm.flatten() * 1e6, cmap='viridis', s=1)
    plt.colorbar(sc, ax=ax, label='Potential U(microK)')
    
    ax.set_xlabel('X (microns)')
    ax.set_ylabel('Y (microns)')
    ax.set_zlabel('Z (microns)')
    ax.set_title('3D Potential Beam U(x, y, z)')
    plt.show()
plot_3d()


def plot_2d():
    # Evaluate the function
    U_values = U(X, Y, Z, p1, p2)

    # Normalize U values for better contrast in plots
    U_norm = U_values / kb

    # XY Plane (at Z=0)
    U_xy = U_norm[:, :, z_slice]

    # Create a figure for subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # XY Plane Contour Plot (at Z=0)
    axs[0, 0].contourf(x * 1e6, y * 1e6, U_xy.T * 1e6, levels=100, cmap='viridis')
    axs[0, 0].set_title('Potential U in XY Plane (Z=0)')
    axs[0, 0].set_xlabel('X (microns)')
    axs[0, 0].set_ylabel('Y (microns)')
    fig.colorbar(axs[0, 0].contourf(x * 1e6, y * 1e6, U_xy.T * 1e6, levels=100, cmap='viridis'), ax=axs[0, 0], label='Potential U(microK)')

    # YZ Plane Contour Plot (at X=0)
    U_yz = U_norm[x_slice, :, :]
    axs[0, 1].contourf(y * 1e6, z * 1e6, U_yz.T * 1e6, levels=100, cmap='viridis')
    axs[0, 1].set_title('Potential U in YZ Plane (X=0)')
    axs[0, 1].set_xlabel('Y (microns)')
    axs[0, 1].set_ylabel('Z (microns)')
    fig.colorbar(axs[0, 1].contourf(y * 1e6, z * 1e6, U_yz.T * 1e6, levels=100, cmap='viridis'), ax=axs[0, 1], label='Potential U(microK)')

    # ZX Plane Contour Plot (at Y=0)
    U_zx = U_norm[:, y_slice, :]
    axs[1, 0].contourf(x * 1e6, z * 1e6, U_zx.T * 1e6, levels=100, cmap='viridis')
    axs[1, 0].set_title('Potential U in ZX Plane (Y=0)')
    axs[1, 0].set_xlabel('Z (microns)')
    axs[1, 0].set_ylabel('X (microns)')
    fig.colorbar(axs[1, 0].contourf(x * 1e6, z * 1e6, U_zx.T * 1e6, levels=100, cmap='viridis'), ax=axs[1, 0], label='Potential U(microK)')

    # Add contours for each subplot
    contour_lines_xy = axs[0, 0].contour(x * 1e6, y * 1e6, U_xy.T * 1e6, levels=5, colors='black', linewidths=0.7)
    axs[0, 0].clabel(contour_lines_xy, inline=True, fontsize=8)
    
    contour_lines_yz = axs[0, 1].contour(y * 1e6, z * 1e6, U_yz.T * 1e6, levels=5, colors='black', linewidths=0.7)
    axs[0, 1].clabel(contour_lines_yz, inline=True, fontsize=8)
    
    contour_lines_zx = axs[1, 0].contour(z * 1e6, x * 1e6, U_zx.T * 1e6, levels=5, colors='black', linewidths=0.7)
    axs[1, 0].clabel(contour_lines_zx, inline=True, fontsize=8)

    plt.tight_layout()
    plt.show()

    # 3D Surface Plots for subplots
    fig = plt.figure(figsize=(12, 10))

    # XY Plane (3D surface plot)
    x1, y1 = np.meshgrid(x, y)
    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    surface_xy = ax1.plot_surface(x1 * 1e6, y1 * 1e6, U_xy.T * 1e6, cmap="coolwarm_r", edgecolor='none')
    ax1.set_xlabel("X (microns)")
    ax1.set_ylabel("Y (microns)")
    ax1.set_zlabel("Potential U (µK)")
    fig.colorbar(surface_xy, ax=ax1, shrink=0.5, aspect=10, label="Potential (µK)")

    # YZ Plane (3D surface plot)
    y1, z1 = np.meshgrid(y, z)
    ax2 = fig.add_subplot(2, 2, 4, projection='3d')
    surface_yz = ax2.plot_surface(y1 * 1e6, z1 * 1e6, U_yz.T * 1e6, cmap="coolwarm_r", edgecolor='none')
    ax2.set_xlabel("Y (microns)")
    ax2.set_ylabel("Z (microns)")
    ax2.set_zlabel("Potential U (µK)")
    fig.colorbar(surface_yz, ax=ax2, shrink=0.5, aspect=10, label="Potential (µK)")

    # ZX Plane (3D surface plot)
    z1, x1 = np.meshgrid(z, x)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surface_zx = ax3.plot_surface(z1 * 1e6, x1 * 1e6, U_zx.T * 1e6, cmap="coolwarm_r", edgecolor='none')
    ax3.set_xlabel("X (microns)")
    ax3.set_ylabel("Z (microns)")
    ax3.set_zlabel("Potential U (µK)")
    fig.colorbar(surface_zx, ax=ax3, shrink=0.5, aspect=10, label="Potential (µK)")

    plt.tight_layout()
    plt.show()
plot_2d()



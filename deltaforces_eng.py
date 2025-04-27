import numpy as np
import matplotlib.pyplot as plt
import datetime



# Delta parameters
m_R = 0.050  # Mass of a pair of arms in kg
m_E = 0.200  # Mass of the effector in kg
m_W = 0.100  # MAss of a carriage in kg
r_D = 100  # Delta radius in mm
r_B = 90  # Printable radius of the printer in mm
x_E = 0  # x-coordinate of the effector (the effector is moved along y in this calc)
L = 210  # Arm length in mm
a_e = 5  # Effector acceleration in m/s^2
g = 9.8  # Gravitational acceleration in m/s^2

if abs(x_E) >= r_B:
    print("Error, x_E outside the range!")
    quit()

points = 1801
extent = np.sqrt(r_B * r_B - x_E * x_E)
y_range = np.linspace(-extent, extent, points) # Coordinates from edge to edge of the printable area




# Define the distance formula
def distance_from_line(x0, y0, A, B, C):
    """
    Calculates the perpendicular distance from point (x0, y0) to the line Ax + By + C = 0
    """
    return abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)

def calculate_forces(y_E):
    # Coordinates of points A, B, and C (forming the triangle)
    C_x, C_y = 0, r_D
    B_x, B_y = r_D * np.sin(np.pi / 3), -r_D * np.cos(np.pi / 3)
    A_x, A_y = -r_D * np.sin(np.pi / 3), -r_D * np.cos(np.pi / 3)
    
    # Midpoints of the lines (CE, AE, and B)
    M_CE_x, M_CE_y = (C_x + x_E) / 2, (C_y + y_E) / 2
    M_AE_x, M_AE_y = (A_x + x_E) / 2, (A_y + y_E) / 2
    
    # Calculate slope and line BC equation: (C_x - B_x) * (y - B_y) = (C_y - B_y) * (x - B_x)
    slope_BC = (C_y - B_y) / (C_x - B_x) if B_x != C_x else float('inf')  # Avoid division by zero for vertical line
    A_BC = slope_BC
    B_BC = -1
    C_BC = B_y - A_BC * B_x
    
    # Calculate the perpendicular distances from points M_CE, E, and M_AE to line BC
    d_M_CE_BC = distance_from_line(M_CE_x, M_CE_y, A_BC, B_BC, C_BC)
    d_E_BC = distance_from_line(x_E, y_E, A_BC, B_BC, C_BC)
    d_M_AE_BC = distance_from_line(M_AE_x, M_AE_y, A_BC, B_BC, C_BC)
    
    # Calculate the force at point A using the given formula
    F_A = (2 * m_R * d_M_CE_BC + m_E * d_E_BC + m_R * d_M_AE_BC) / (1.5 * r_D)
    
    # Calculate the force at point C using the corrected formula
    F_C = m_W + ((y_E * (2 * m_E + 3 * m_R) + r_D * (m_E + 3 * m_R)) / (3 * r_D))
    
    # Calculate the force at point B by subtracting forces at A and C from the total load
    F_B = m_E + 3 * (m_W + m_R) - F_A - F_C
    
    return F_C

def calculate_dynamic_load(F_C, y):
    """
    Calculates the dynamic load at point C based on the given formula.
    """
    if abs(r_D - y) > L:
        raise ValueError("Invalid value: |r_D - y| cannot exceed L.")
    
    angle = np.arccos((r_D - y) / L)  # Calculate acos((r_D - y) / L)
    cot_angle = 1 / np.tan(angle)  # Calculate cot(angle)
    F_dyn_C = (cot_angle * a_e + g) * F_C  # Dynamic load formula
    return F_dyn_C

def sweep_pos(y_range):
    F_stat = np.zeros(y_range.size)
    F_dyn = np.zeros(y_range.size)
    for y in range(points):
        F_stat[y] = calculate_forces(y_range[y])
        F_dyn[y] = calculate_dynamic_load(F_stat[y], y_range[y])
    return F_stat, F_dyn
# Run the function to calculate loads

forces = sweep_pos(y_range)
plt.plot(y_range, forces[0], label = "Static force")
plt.plot(y_range, forces[1], label = "Dynamic force")
plt.xlabel("y position in mm")
plt.ylabel("Force in N")
plt.legend()
plt.suptitle(("Forces on C carriage for x = " + str(x_E) + " mm"), fontsize = 18)
plt.title(("Effector mass: " + str(m_E * 1000) + "g, carriage mass: " + str(m_W * 1000) + " g, arm pair mass: " + str(m_R * 1000) + " g\nArm length: " + str(L) + " mm, delta radius: " + str(r_D) + " mm, bed diameter: " + str(r_B * 2) + " mm"), fontsize = 10)
plt.tight_layout()
time = datetime.datetime.now()
title = "DeltaForces - " + str(time.month).zfill(2) + "_" + str(time.day).zfill(2) + " - " + str(time.hour).zfill(2) + "_" + str(time.minute).zfill(2) + ".jpg"
print(title)
plt.savefig(title, dpi=300) # Comment if you don't want to save the figure
plt.show()

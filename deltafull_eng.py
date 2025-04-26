import numpy as np
import matplotlib.pyplot as plt
import datetime
# Delta parameters
DELTA_DIAGONAL_ROD = 210  # Arm length in mm
DELTA_RADIUS = 100  # Delta radius in mm
MAX_RADIUS = 90  # Radius for which the delta will be simulated, half of printable diameter, in mm
POINT_COUNT = 4000  # Number of grid points for simulation
P = 2  # Belt pitch in mm
T = 16  # Pulley tooth count
M_SYNCH = 600  # Stepper holding torque in mNm
N = 400  # Stepper's steps per revolution
Z_CONSTANT = 0  # Constant Z coordinate
tol = 0.05  # Stepper step angle tolerance as fraction

# Precalculated adjustment value
adjustment = (tol * P * T) / N

# Constants for forces and errors
m_R, m_E, m_W = 0.1, 0.2, 0.1  # Masses of arm pairs, effector and carriages, in kg

F_friction = 2.1 # Friction force inside the linear guide in N




#
F_drive = m_E / 3 + m_R + m_W
M_drive = F_drive * P * T / 2 / np.pi
e_drive = np.arcsin(M_drive / M_SYNCH) * 4 * P * T / (N * 2 * np.pi)


# A quick function to calculate square
def sq(num):
    return num * num

# A quick function to calculate distance of a point from a line
def distance_from_line(x0, y0, A, B, C):
    return (A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)

# Calculation of forces within a delta
def calculate_forces(r_D, m_R, m_E, m_W, x_E, y_E):
    C_x, C_y = 0, r_D
    B_x, B_y = r_D * np.sin(np.pi / 3), -r_D * np.cos(np.pi / 3)
    A_x, A_y = -r_D * np.sin(np.pi / 3), -r_D * np.cos(np.pi / 3)
    M_CE_x, M_CE_y = (C_x + x_E) / 2, (C_y + y_E) / 2
    M_AE_x, M_AE_y = (A_x + x_E) / 2, (A_y + y_E) / 2
    M_BE_x, M_BE_y = (B_x + x_E) / 2, (B_y + y_E) / 2
    slope_BC = (C_y - B_y) / (C_x - B_x) if B_x != C_x else float('inf')
    A_BC = slope_BC
    B_BC = -1
    C_BC = B_y - A_BC * B_x
    d_M_CE_BC = distance_from_line(M_CE_x, M_CE_y, A_BC, B_BC, C_BC)
    d_E_BC = distance_from_line(x_E, y_E, A_BC, B_BC, C_BC)
    d_M_AE_BC = distance_from_line(M_AE_x, M_AE_y, A_BC, B_BC, C_BC)
    
    F_A = m_W + (2 * m_R * d_M_CE_BC + m_E * d_E_BC + m_R * d_M_AE_BC) / (1.5 * r_D)
    F_C = m_W + ((y_E * (2 * m_E + 3 * m_R) + r_D * (m_E + 3 * m_R)) / (3 * r_D))

    # Calculate slope and line AC equation: (C_x - A_x) * (y - A_y) = (C_y - A_y) * (x - A_x)
    slope_AC = (C_y - A_y) / (C_x - A_x) if A_x != C_x else float('inf')  # Avoid division by zero for vertical line
    A_AC = slope_AC
    B_AC = -1
    C_AC = A_y - A_AC * A_x
    
    # Calculate the perpendicular distances from points M_CE, E, and M_BE to line AC
    d_M_CE_AC = distance_from_line(M_CE_x, M_CE_y, A_AC, B_AC, C_AC)
    d_E_AC = distance_from_line(x_E, y_E, A_AC, B_AC, C_AC)
    d_M_BE_AC = distance_from_line(M_BE_x, M_BE_y, A_AC, B_AC, C_AC)

    F_B = m_W + (2 * m_R * d_M_CE_AC + m_E * d_E_AC + m_R * d_M_BE_AC) / (1.5 * r_D)



    return F_A, F_B, F_C


# Delta Tower positions
SIN_60 = np.sqrt(3) / 2
COS_60 = 0.5

DELTA_TOWER1_X, DELTA_TOWER1_Y = 0.0, DELTA_RADIUS
DELTA_TOWER2_X, DELTA_TOWER2_Y = -SIN_60 * DELTA_RADIUS, -COS_60 * DELTA_RADIUS
DELTA_TOWER3_X, DELTA_TOWER3_Y = SIN_60 * DELTA_RADIUS, -COS_60 * DELTA_RADIUS


# Inverse and Forward kinematics
def inverse(cartesian):
    delta = np.zeros(3)
    delta[0] = np.sqrt(sq(DELTA_DIAGONAL_ROD) - sq(DELTA_TOWER1_X - cartesian[0]) - sq(DELTA_TOWER1_Y - cartesian[1])) + cartesian[2]
    delta[1] = np.sqrt(sq(DELTA_DIAGONAL_ROD) - sq(DELTA_TOWER2_X - cartesian[0]) - sq(DELTA_TOWER2_Y - cartesian[1])) + cartesian[2]
    delta[2] = np.sqrt(sq(DELTA_DIAGONAL_ROD) - sq(DELTA_TOWER3_X - cartesian[0]) - sq(DELTA_TOWER3_Y - cartesian[1])) + cartesian[2]
    return delta

def forward(delta):
    y1, z1 = DELTA_TOWER1_Y, delta[0]
    x2, y2, z2 = DELTA_TOWER2_X, DELTA_TOWER2_Y, delta[1]
    x3, y3, z3 = DELTA_TOWER3_X, DELTA_TOWER3_Y, delta[2]
    
    re = DELTA_DIAGONAL_ROD
    dnm = (y2 - y1) * x3 - (y3 - y1) * x2
    w1 = y1**2 + z1**2
    w2 = x2**2 + y2**2 + z2**2
    w3 = x3**2 + y3**2 + z3**2
    
    a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
    b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2.0
    a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
    b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2.0
    
    a = a1**2 + a2**2 + dnm**2
    b = 2 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm**2)
    c = (b2 - y1 * dnm)**2 + b1**2 + dnm**2 * (z1**2 - re**2)
    
    d = b**2 - 4.0 * a * c
    if d < 0:
        return None  # Non-existing point
    
    cartesian = np.zeros(3)
    cartesian[2] = -0.5 * (b + np.sqrt(d)) / a
    cartesian[0] = (a1 * cartesian[2] + b1) / dnm
    cartesian[1] = (a2 * cartesian[2] + b2) / dnm
    return cartesian



def calculate_distance(point1, point2):
    return np.sqrt(sq(point2[0] - point1[0]) + sq(point2[1] - point1[1]) + sq(point2[2] - point1[2]))

# Grid generation
step_size = 2 * MAX_RADIUS / np.sqrt(POINT_COUNT)
x = np.arange(-MAX_RADIUS, MAX_RADIUS, step_size)
y = np.arange(-MAX_RADIUS, MAX_RADIUS, step_size)
xx, yy = np.meshgrid(x, y)

# Initialize arrays for F_min, F_max, M_min, M_max, ERROR_min, ERROR_max
F_min = {'A': np.zeros_like(xx), 'B': np.zeros_like(xx), 'C': np.zeros_like(xx)}
F_max = {'A': np.zeros_like(xx), 'B': np.zeros_like(xx), 'C': np.zeros_like(xx)}
M_min = {'A': np.zeros_like(xx), 'B': np.zeros_like(xx), 'C': np.zeros_like(xx)}
M_max = {'A': np.zeros_like(xx), 'B': np.zeros_like(xx), 'C': np.zeros_like(xx)}
ERROR_min = {'A': np.zeros_like(xx), 'B': np.zeros_like(xx), 'C': np.zeros_like(xx)}
ERROR_max = {'A': np.zeros_like(xx), 'B': np.zeros_like(xx), 'C': np.zeros_like(xx)}

# Main calculation loop
z = np.zeros_like(xx)  # Initialize result array
min_error, max_error = float('inf'), 0

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        if xx[i, j]**2 + yy[i, j]**2 <= MAX_RADIUS**2:
            point = [xx[i, j], yy[i, j], Z_CONSTANT]
            delta_coordinates = inverse(point)
            # Calculate forces for each of the columns at the point
            F_A, F_B, F_C = calculate_forces(DELTA_RADIUS, m_R, m_E, m_W, xx[i, j], yy[i, j])
            
            # Calculate F_min and F_max for each column
            F_min['A'][i, j] = F_A - F_friction
            F_max['A'][i, j] = F_A + F_friction
            F_min['B'][i, j] = F_B - F_friction
            F_max['B'][i, j] = F_B + F_friction
            F_min['C'][i, j] = F_C - F_friction
            F_max['C'][i, j] = F_C + F_friction

            # Calculate M_min and M_max for each column
            M_min['A'][i, j] = F_min['A'][i, j] * P * T / (2 * np.pi)
            M_max['A'][i, j] = F_max['A'][i, j] * P * T / (2 * np.pi)
            M_min['B'][i, j] = F_min['B'][i, j] * P * T / (2 * np.pi)
            M_max['B'][i, j] = F_max['B'][i, j] * P * T / (2 * np.pi)
            M_min['C'][i, j] = F_min['C'][i, j] * P * T / (2 * np.pi)
            M_max['C'][i, j] = F_max['C'][i, j] * P * T / (2 * np.pi)

            # Calculate ERROR_min and ERROR_max for each column
            ERROR_min['A'][i, j] = np.arcsin(M_min['A'][i, j] / M_SYNCH) * 4 * P * T / (N * 2 * np.pi) - adjustment - e_drive
            ERROR_max['A'][i, j] = np.arcsin(M_max['A'][i, j] / M_SYNCH) * 4 * P * T / (N * 2 * np.pi) + adjustment - e_drive
            ERROR_min['B'][i, j] = np.arcsin(M_min['B'][i, j] / M_SYNCH) * 4 * P * T / (N * 2 * np.pi) - adjustment - e_drive
            ERROR_max['B'][i, j] = np.arcsin(M_max['B'][i, j] / M_SYNCH) * 4 * P * T / (N * 2 * np.pi) + adjustment - e_drive
            ERROR_min['C'][i, j] = np.arcsin(M_min['C'][i, j] / M_SYNCH) * 4 * P * T / (N * 2 * np.pi) - adjustment - e_drive
            ERROR_max['C'][i, j] = np.arcsin(M_max['C'][i, j] / M_SYNCH) * 4 * P * T / (N * 2 * np.pi) + adjustment - e_drive


            # Calculate max distance from the original point (xx[i, j], yy[i, j], Z_CONSTANT)
            max_distance = 0
            for dA in [ERROR_min['A'][i, j], 0, ERROR_max['A'][i, j]]:
                for dB in [ERROR_min['B'][i, j], 0, ERROR_max['B'][i, j]]:
                    for dC in [ERROR_min['C'][i, j], 0, ERROR_max['C'][i, j]]:
                        delta_with_error = delta_coordinates + [dC, dA, dB]
                        #delta_with_error = delta_coordinates + [1, 0, 0]
                        new_point = forward(delta_with_error)
                        if new_point is not None:
                            distance = calculate_distance(point, new_point)
                            max_distance = max(max_distance, distance)

            z[i, j] = max_distance
            min_error = min(min_error, max_distance)
            max_error = max(max_error, max_distance)
        else:
            z[i, j] = np.nan  # Exclude points outside the radius


# Plotting with Matplotlib
plt.figure(figsize=(10, 10))
heatmap = plt.imshow(z, extent=(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS),
                     origin='lower', cmap='jet', interpolation='nearest')
cbar = plt.colorbar(heatmap, label='Effector positioning uncertainty (mm)')


plt.title(f"Delta positioning uncertainty map\n"
          f"Effector positioning uncertainty range: {np.round(min_error, decimals = 6)} - {np.round(max_error, decimals = 6)} mm\n"
          f"Delta radius: {DELTA_RADIUS} mm, Printable area diameter: {MAX_RADIUS*2} mm, Arm length: {DELTA_DIAGONAL_ROD} mm\n"
          f"Effector weight: {m_E*1000} g, Arm pair weight: {m_R*1000} g, Carriage weight: {m_W*1000} g\n"
          f"Carriage friction force: {F_friction} N")
plt.xlabel('X coordinate in mm')
plt.ylabel('Y coordinate in mm')
plt.gca().set_aspect('equal')  # Ensure the plot is square
time = datetime.datetime.now()
title = "DeltaFull - " + str(time.month).zfill(2) + "_" + str(time.day).zfill(2) + " - " + str(time.hour).zfill(2) + "_" + str(time.minute).zfill(2) + ".jpg"
print(title)
plt.savefig(title, dpi=300) # Comment if you don't want to save the figure
plt.show()

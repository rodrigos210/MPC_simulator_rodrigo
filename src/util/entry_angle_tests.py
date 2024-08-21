import numpy as np

# Define points A, B, and C
A = np.array([12, 12, 0])
B = np.array([10, 10, 0])
C = np.array([11, 10, 0])

# Calculate vectors AB and BC
AB = B - A
BC = C - B
BA = A - B

# Calculate the dot product and magnitudes
dot_product = np.dot(AB, BC)
magnitude_AB = np.linalg.norm(AB)
magnitude_BA = np.linalg.norm(BA)
magnitude_BC = np.linalg.norm(BC)

# Calculate the cosine of the angle
cosine_angle = dot_product / (magnitude_AB * magnitude_BC)

# Calculate the angle in radians and degrees
angle_rad = np.arccos(cosine_angle)
angle_deg = np.degrees(angle_rad)
print("Cosine",cosine_angle)
print(f"Angle between AB and BC: {angle_rad:.2f} radians, {angle_deg:.2f} degrees")

import casadi as ca

x_ref = ca.DM([])
x_obstacle = ca.DM([5, 5])
for i in range(40):
    x_ref.append([i/4,i/8])
    pos_delta = x_ref[0:2] - x_obstacle[0:2]
    distance_to_obstacle = ca.norm_2(pos_delta)
    print(distance_to_obstacle)

import numpy as np
import torch


def generate_cylinder_flow_data(num_samples, num_boundary_points, num_residual_points, num_initial_points,
                                cylinder_radius=0.1, domain_size=(-1, 5, -2, 2),
                                time_range=(0, 30), inlet_velocity_range=(0.5, 2.0)):
    x_min, x_max, y_min, y_max = domain_size
    t_min, t_max = time_range
    v_min, v_max = inlet_velocity_range
    cylinder_x, cylinder_y = 0.0, 0.0

    u_bcs_list = []
    y_bcs_list = []
    s_bcs_list = []

    u_res_list = []
    y_res_list = []
    s_res_list = []

    u_ic_list = []
    y_ic_list = []
    s_ic_list = []

    for _ in range(num_samples):
        inlet_velocity = np.random.uniform(v_min, v_max)

        x_ic, y_ic = generate_initial_points(num_initial_points, cylinder_radius, cylinder_x, cylinder_y, domain_size)
        t_ic = np.zeros(len(x_ic))

        u_ic = np.zeros(len(x_ic))
        v_ic = np.zeros(len(x_ic))
        p_ic = np.zeros(len(x_ic))

        dist_to_inlet = np.abs(x_ic - x_min)
        dist_to_outlet = np.abs(x_ic - x_max)
        dist_to_top = np.abs(y_ic - y_max)
        dist_to_bottom = np.abs(y_ic - y_min)
        dist_to_cylinder = np.sqrt((x_ic - cylinder_x) ** 2 + (y_ic - cylinder_y) ** 2)

        boundary_tol = 0.01

        for i in range(len(x_ic)):
            x, y = x_ic[i], y_ic[i]
            dist_cyl = dist_to_cylinder[i]

            if dist_cyl < cylinder_radius + boundary_tol:
                u_ic[i] = 0.0
                v_ic[i] = 0.0

            elif dist_to_inlet[i] < boundary_tol:
                u_ic[i] = inlet_velocity
                v_ic[i] = 0.0

            elif dist_to_outlet[i] < boundary_tol:
                u_ic[i] = inlet_velocity
                v_ic[i] = 0.0

            elif dist_to_top[i] < boundary_tol:
                u_ic[i] = 0.0
                v_ic[i] = 0.0

            elif dist_to_bottom[i] < boundary_tol:
                u_ic[i] = 0.0
                v_ic[i] = 0.0

            else:
                if dist_cyl < cylinder_radius * 2.0:
                    transition_ratio = max(0.0, (dist_cyl - cylinder_radius) / cylinder_radius)
                    u_ic[i] = inlet_velocity * transition_ratio
                else:
                    u_ic[i] = inlet_velocity

        u_ic_input = np.full((len(x_ic), 1), inlet_velocity)
        y_ic_coords = np.column_stack([x_ic, y_ic, t_ic])
        s_ic = np.column_stack([u_ic, v_ic, p_ic])

        u_ic_list.append(u_ic_input)
        y_ic_list.append(y_ic_coords)
        s_ic_list.append(s_ic)


        n_cyl = int(num_boundary_points * 0.3)
        n_inlet = int(num_boundary_points * 0.15)
        n_outlet = int(num_boundary_points * 0.15)
        n_top = int(num_boundary_points * 0.2)
        n_bottom = int(num_boundary_points * 0.2)

        theta = np.random.uniform(0, 2 * np.pi, n_cyl)
        r = cylinder_radius
        x_cyl = cylinder_x + r * np.cos(theta)
        y_cyl = cylinder_y + r * np.sin(theta)
        t_cyl = np.random.uniform(t_min, t_max, n_cyl)

        x_inlet = np.full(n_inlet, x_min)
        y_inlet = np.random.uniform(y_min, y_max, n_inlet)
        t_inlet = np.random.uniform(t_min, t_max, n_inlet)

        x_outlet = np.full(n_outlet, x_max)
        y_outlet = np.random.uniform(y_min, y_max, n_outlet)
        t_outlet = np.random.uniform(t_min, t_max, n_outlet)

        y_top = np.full(n_top, y_max)
        x_top = np.random.uniform(x_min, x_max, n_top)
        t_top = np.random.uniform(t_min, t_max, n_top)

        y_bottom = np.full(n_bottom, y_min)
        x_bottom = np.random.uniform(x_min, x_max, n_bottom)
        t_bottom = np.random.uniform(t_min, t_max, n_bottom)

        x_bcs = np.concatenate([x_cyl, x_inlet, x_outlet, x_top, x_bottom])
        y_bcs = np.concatenate([y_cyl, y_inlet, y_outlet, y_top, y_bottom])
        t_bcs = np.concatenate([t_cyl, t_inlet, t_outlet, t_top, t_bottom])

        u_cyl = np.zeros(n_cyl)
        v_cyl = np.zeros(n_cyl)
        p_cyl = np.zeros(n_cyl)

        u_inlet = np.full(n_inlet, inlet_velocity)
        v_inlet = np.zeros(n_inlet)
        p_inlet = np.zeros(n_inlet)

        u_outlet = np.full(n_outlet, inlet_velocity)
        v_outlet = np.zeros(n_outlet)
        p_outlet = np.zeros(n_outlet)

        u_top = np.zeros(n_top)
        v_top = np.zeros(n_top)
        p_top = np.zeros(n_top)

        u_bottom = np.zeros(n_bottom)
        v_bottom = np.zeros(n_bottom)
        p_bottom = np.zeros(n_bottom)

        u_bcs = np.concatenate([u_cyl, u_inlet, u_outlet, u_top, u_bottom])
        v_bcs = np.concatenate([v_cyl, v_inlet, v_outlet, v_top, v_bottom])
        p_bcs = np.concatenate([p_cyl, p_inlet, p_outlet, p_top, p_bottom])

        u_input = np.full((len(x_bcs), 1), inlet_velocity)

        y_bcs_coords = np.column_stack([x_bcs, y_bcs, t_bcs])

        s_bcs = np.column_stack([u_bcs, v_bcs, p_bcs])

        x_res, y_res, t_res = generate_collocation_points(
            num_residual_points, cylinder_radius, cylinder_x, cylinder_y,
            domain_size, time_range
        )

        u_res_input = np.full((len(x_res), 1), inlet_velocity)
        y_res_coords = np.column_stack([x_res, y_res, t_res])
        s_res = np.zeros((len(x_res), 3))

        u_bcs_list.append(u_input)
        y_bcs_list.append(y_bcs_coords)
        s_bcs_list.append(s_bcs)

        u_res_list.append(u_res_input)
        y_res_list.append(y_res_coords)
        s_res_list.append(s_res)

    u_ic_train = np.vstack(u_ic_list).astype(np.float32)
    y_ic_train = np.vstack(y_ic_list).astype(np.float32)
    s_ic_train = np.vstack(s_ic_list).astype(np.float32)

    u_bcs_train = np.vstack(u_bcs_list).astype(np.float32)
    y_bcs_train = np.vstack(y_bcs_list).astype(np.float32)
    s_bcs_train = np.vstack(s_bcs_list).astype(np.float32)

    u_res_train = np.vstack(u_res_list).astype(np.float32)
    y_res_train = np.vstack(y_res_list).astype(np.float32)
    s_res_train = np.vstack(s_res_list).astype(np.float32)

    return (u_ic_train, y_ic_train, s_ic_train,
            u_bcs_train, y_bcs_train, s_bcs_train,
            u_res_train, y_res_train, s_res_train)


def generate_initial_points(num_points, cylinder_radius, cylinder_x, cylinder_y, domain_size):

    x_min, x_max, y_min, y_max = domain_size

    x_points, y_points = [], []
    cnt = 0

    boundary_ratio = 0.3
    n_boundary = int(num_points * boundary_ratio)
    n_interior = num_points - n_boundary

    boundary_tol = 0.01

    n_inlet = n_boundary // 5
    x_inlet = np.full(n_inlet, x_min)
    y_inlet = np.random.uniform(y_min, y_max, n_inlet)

    n_outlet = n_boundary // 5
    x_outlet = np.full(n_outlet, x_max)
    y_outlet = np.random.uniform(y_min, y_max, n_outlet)

    n_top = n_boundary // 5
    y_top = np.full(n_top, y_max)
    x_top = np.random.uniform(x_min, x_max, n_top)

    n_bottom = n_boundary // 5
    y_bottom = np.full(n_bottom, y_min)
    x_bottom = np.random.uniform(x_min, x_max, n_bottom)

    n_cyl = n_boundary - (n_inlet + n_outlet + n_top + n_bottom)
    theta = np.random.uniform(0, 2 * np.pi, n_cyl)
    x_cyl = cylinder_x + cylinder_radius * np.cos(theta)
    y_cyl = cylinder_y + cylinder_radius * np.sin(theta)

    x_boundary = np.concatenate([x_inlet, x_outlet, x_top, x_bottom, x_cyl])
    y_boundary = np.concatenate([y_inlet, y_outlet, y_top, y_bottom, y_cyl])

    x_points.extend(x_boundary)
    y_points.extend(y_boundary)

    cnt = 0
    while cnt < n_interior:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        dist = np.sqrt((x - cylinder_x) ** 2 + (y - cylinder_y) ** 2)
        if dist > cylinder_radius:
            on_boundary = (
                    abs(x - x_min) < boundary_tol or
                    abs(x - x_max) < boundary_tol or
                    abs(y - y_min) < boundary_tol or
                    abs(y - y_max) < boundary_tol or
                    abs(dist - cylinder_radius) < boundary_tol
            )

            if not on_boundary:
                x_points.append(x)
                y_points.append(y)
                cnt += 1

    return np.array(x_points), np.array(y_points)


def generate_collocation_points(num_points, cylinder_radius, cylinder_x, cylinder_y,
                                           domain_size, time_range):
    x_min, x_max, y_min, y_max = domain_size
    t_min, t_max = time_range

    x_points, y_points, t_points = [], [], []

    n_cylinder_layers = int(num_points * 0.3)

    radius_multipliers = [1.1, 1.3, 1.6, 2.0, 2.5]
    points_per_layer = n_cylinder_layers // len(radius_multipliers)

    for multiplier in radius_multipliers:
        radius = cylinder_radius * multiplier
        for _ in range(points_per_layer):
            theta = np.random.uniform(0, 2 * np.pi)
            x = cylinder_x + radius * np.cos(theta)
            y = cylinder_y + radius * np.sin(theta)
            t = np.random.uniform(t_min, t_max)

            if (x_min <= x <= x_max) and (y_min <= y <= y_max):
                x_points.append(x)
                y_points.append(y)
                t_points.append(t)

    n_wake = int(num_points * 0.25)
    for _ in range(n_wake):
        x = np.random.uniform(cylinder_x + cylinder_radius, cylinder_x + 4 * cylinder_radius)
        y = np.random.uniform(cylinder_y - 2 * cylinder_radius, cylinder_y + 2 * cylinder_radius)
        t = np.random.uniform(t_min * 0.2, t_max)

        dist = np.sqrt((x - cylinder_x) ** 2 + (y - cylinder_y) ** 2)
        if dist > cylinder_radius and (x_min <= x <= x_max) and (y_min <= y <= y_max):
            x_points.append(x)
            y_points.append(y)
            t_points.append(t)

    n_remaining = num_points - len(x_points)
    cnt = 0
    while cnt < n_remaining:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        t = np.random.uniform(t_min, t_max)

        dist = np.sqrt((x - cylinder_x) ** 2 + (y - cylinder_y) ** 2)
        if dist > cylinder_radius:
            x_points.append(x)
            y_points.append(y)
            t_points.append(t)
            cnt += 1

    return np.array(x_points), np.array(y_points), np.array(t_points)


def generate_prediction_data(inlet_velocity, grid_size, cylinder_radius=0.1,
                             domain_size=(-1, 5, -2, 2), time_range=(0, 30)):
    x_min, x_max, y_min, y_max = domain_size
    t_min, t_max = time_range
    cylinder_x, cylinder_y = 0.0, 0.0

    nx, ny, nt = grid_size

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    t = np.linspace(t_min, t_max, nt)

    XX, YY, TT = np.meshgrid(x, y, t, indexing='ij')

    x_flat = XX.flatten()
    y_flat = YY.flatten()
    t_flat = TT.flatten()

    dist_flat = np.sqrt((x_flat - cylinder_x) ** 2 + (y_flat - cylinder_y) ** 2)
    mask_flat = dist_flat > cylinder_radius
    x_flat = x_flat[mask_flat]
    y_flat = y_flat[mask_flat]
    t_flat = t_flat[mask_flat]

    u_pred = np.full((len(x_flat), 1), inlet_velocity).astype(np.float32)
    y_pred = np.column_stack([x_flat, y_flat, t_flat]).astype(np.float32)

    return u_pred, y_pred, x_flat, y_flat, t_flat, x, y, t
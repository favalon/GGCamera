# Use numpy.linalg.norm:
import numpy as np
import matplotlib.pyplot as plt


def line_vector(point_1, point_2):
    dist = np.linalg.norm(point_2 - point_1)
    line = point_2 - point_1
    unit_vector = line / dist

    return unit_vector, dist


def rotation(uv1, uv2, dist1, dist2):
    scale = dist2/dist1

    # uv1 = np.array([1, 0, 0], dtype=np.float64)
    # uv2 = np.array([0, 0, 1], dtype=np.float64)

    v = np.cross(uv1, uv2)
    if np.linalg.norm(v) == 0:
        return np.eye(3), scale
    s = np.linalg.norm(v)
    c = np.dot(uv1, uv2)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s ** 2)

    uv1_proj_r = np.dot(r, uv1)

    return r, scale


def plot(uv1, dist1, uv2, dist2, r, original_line1, original_line2):
    original_point = np.array([0, 0, 0])

    line1 = np.array([original_point, uv1]) * dist1
    line2 = np.array([original_point, uv2]) * dist2

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax1.plot(line1[:, 0], line1[:, 1], line1[:, 2], '.r-', linewidth=2)
    ax1.plot(line2[:, 0], line2[:, 1], line2[:, 2], '.r-', linewidth=2)

    ax1.plot(original_line1[:, 0], original_line1[:, 1], original_line1[:, 2], '.g-', linewidth=2)
    ax1.plot(original_line2[:, 0], original_line2[:, 1], original_line2[:, 2], '.g-', linewidth=2)

    project_line2 = np.array([np.dot(r, original_point), np.dot(r, uv1)])
    ax1.plot(project_line2[:, 0], project_line2[:, 1], project_line2[:, 2], '.b--', linewidth=2)

    plt.show()

    return


def main():
    line1_p1 = np.array([1.0, 3.5, -6.3])
    line1_p2 = np.array([4.5, 1.6, 1.2])

    line2_p1 = np.array([2.0, 1.5, 5.3])
    line2_p2 = np.array([1.5, 3.6, 4.2])

    uv1, dist1 = line_vector(line1_p1, line1_p2)
    uv2, dist2 = line_vector(line2_p1, line2_p2)

    original_line1 = np.array([[1.0, 3.5, -6.3], [4.5, 1.6, 1.2]])
    original_line2 = np.array([[2.0, 1.5, 5.3], [1.5, 3.6, 4.2]])

    rotation_matrix, scale = rotation(uv1, uv2, dist1, dist2)
    plot(uv1, dist1, uv2, dist2, rotation_matrix, original_line1, original_line2)

    return


if __name__ == "__main__":
    main()

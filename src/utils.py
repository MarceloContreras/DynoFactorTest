
import matplotlib.pyplot as plt

def plotPath(trajectory, orientations, ax, label = "Path", color = "blue", linewidth = 2, axis_length=0.1, axis_step=1, no_legend=False):
    """
    Plots a 3D trajectory with orientation axes.
    """
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label= label, color=color, linewidth = linewidth)

    # Plot orientation axes
    for i in range(0, len(trajectory), axis_step):
        pos = trajectory[i]
        R = orientations[i]  # 3x3 rotation matrix

        # Plot local X, Y, Z axes
        for j, color in enumerate(['r', 'g', 'b']):  # X=Red, Y=Green, Z=Blue
            axis_vector = R[:, j] * axis_length
            ax.quiver(pos[0], pos[1], pos[2],
                      axis_vector[0], axis_vector[1], axis_vector[2],
                      color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory with Orientation Axes')
    if not(no_legend):
        ax.legend()
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
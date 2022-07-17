import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from general.save_load import LoadBasic


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, path='local_data/skeleton', fn='test_data.json'):
        self.path = path
        self.fn = fn
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=40, init_func=self.setup_plot, blit=False)

    def setup_plot(self, lim=5):
        """Initial drawing of the scatter plot."""
        x, y, z, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, z, c=c, s=20)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([0, lim])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        skeleton_sequence = LoadBasic.load_basic(fn=self.fn, path=self.path, file_type='json')

        while True:
            for frame, skeleton in enumerate(skeleton_sequence):
                all_skeleton_xyz = np.zeros((4, len(skeleton['Skeleton']), 3))
                for i, points in enumerate(skeleton['Skeleton']):
                    all_skeleton_xyz[0, i, 0] = points['LocalPosition']['x']
                    all_skeleton_xyz[0, i, 1] = points['LocalPosition']['y']
                    all_skeleton_xyz[0, i, 2] = points['LocalPosition']['z']

                    all_skeleton_xyz[1, i, 0] = points['LocalRotation']['x']
                    all_skeleton_xyz[1, i, 1] = points['LocalRotation']['y']
                    all_skeleton_xyz[1, i, 2] = points['LocalRotation']['z']

                    all_skeleton_xyz[2, i, 0] = points['WorldPosition']['x']
                    all_skeleton_xyz[2, i, 1] = points['WorldPosition']['y']
                    all_skeleton_xyz[2, i, 2] = points['WorldPosition']['z']

                    all_skeleton_xyz[3, i, 0] = points['WorldRotation']['x']
                    all_skeleton_xyz[3, i, 1] = points['WorldRotation']['y']
                    all_skeleton_xyz[3, i, 2] = points['WorldRotation']['z']

                s, c = np.ones((len(skeleton['Skeleton']), 2)).T * 30
                tmp = np.c_[all_skeleton_xyz[0, :, 0] + all_skeleton_xyz[2, :, 0],
                            all_skeleton_xyz[0, :, 1] + all_skeleton_xyz[2, :, 1],
                            all_skeleton_xyz[0, :, 2] + all_skeleton_xyz[2, :, 2],
                            s, c]
                tmp = np.c_[all_skeleton_xyz[2, :, 0],
                            all_skeleton_xyz[2, :, 1],
                            all_skeleton_xyz[2, :, 2],
                            s, c]
                yield tmp

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        x = data[:, 0]
        y = data[:, 2]
        z = data[:, 1]
        # Set x and y data...
        self.scat._offsets3d = (x, y, z)
        # Set sizes...
        # self.scat.set_sizes(data[:, 3])
        # # Set colors..
        # self.scat.set_array(data[:, 4])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        # time.sleep(0.1)
        return self.scat,


if __name__ == '__main__':
    a = AnimatedScatter(path='local_data/skeleton', fn='test_data.json')
    plt.show()
    # main()

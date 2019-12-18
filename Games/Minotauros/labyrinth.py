import numpy as np


class Labyrinth:
    """Create a labyrinth given by a boolean array where the paths are True, and the walls are False"""

    def __init__(self, width, height, pathwall_ratio, n_holes):
        """
        :param width: width of the labyrinth
        :param height: height of teh labyrinth
        :param pathwall_ratio: ratio between walkable and not walkable cells
        """
        self.cell_type = {'Wall': 'W',
                          'Path': 'P',
                          'Hole': 'H',
                          'Exit': 'E'}
        self.labyrinth = self.create_labyrinth(width, height, pathwall_ratio, n_holes)
        self.paths = np.zeros(self.labyrinth.shape)
        self.path_num = 0
        self.last_path_label = 1

    def create_labyrinth(self, width, height, pathwall_ratio, n_holes):
        labyrinth = np.random.uniform(0, 1, (width, height))
        labyrinth = labyrinth>pathwall_ratio
        labyrinth = np.where(labyrinth == True, self.cell_type['Path'], self.cell_type['Wall'])
        labyrinth[(np.random.randint(0, width, n_holes), np.random.randint(0, height, n_holes))] = \
            self.cell_type['Hole']
        labyrinth[(np.random.randint(0, width), np.random.randint(0, height))] = self.cell_type['Exit']
        return labyrinth

    def check_path_connection(self):
        "Check that the Path cells are all connected"
        self.paths = np.zeros(self.labyrinth.shape)
        self.path_num = 0
        self.last_path_label = 1
        for i in range(self.labyrinth.shape[0]):
            for j in range(self.labyrinth.shape[1]):
                self.scan(i, j)
        print('Found %d distinct paths' % self.path_num)

    def scan(self, ix, iy):
        if self.labyrinth[ix, iy] == 'P':
            if self.paths[ix, iy] == 0:
                self.paths[ix, iy] = self.last_path_label
                self.path_num += 1
                self.last_path_label += 1
            self.set_adjacent(ix + 1, iy, self.paths[ix, iy])
            # self.set_adjacent(ix - 1, iy, self.paths[ix, iy])
            self.set_adjacent(ix, iy + 1, self.paths[ix, iy])
            # self.set_adjacent(ix, iy - 1, self.paths[ix, iy])

    def set_adjacent(self, ix, iy, path_label):
        if ix < 0 or ix >= self.labyrinth.shape[0] or iy < 0 or iy >= self.labyrinth.shape[1]:
            return
        if self.labyrinth[ix, iy] == 'P':
            if self.paths[ix, iy] == 0:
                self.paths[ix, iy] = path_label
            elif self.paths[ix, iy] != path_label:
                # two paths merges, so we unify the labels
                print(self.paths)
                self.paths[np.where(self.paths == self.paths[ix, iy])] = path_label
                self.path_num -= 1
                print(self.paths)


class Agent:

    def __init__(self, labyrinth):
        self.labyrinth = labyrinth
        self.width, self.height = self.labyrinth.shape
        self._position = None

    @property
    def position(self):
        while self._position is None or self.labyrinth[self._position]==False:
            self._position = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        return self._position

    @position.setter
    def position(self, coord_pair):
        self._position = coord_pair

    def move_up(self):
        ix, iy = self.position
        if ix>0 and self.labyrinth[ix-1, iy]:
            self.position = (ix-1, iy)
        else:
            print('Can not go more up')

    def move_down(self):
        ix, iy = self.position
        if ix<self.height-1 and self.labyrinth[ix+1, iy]:
            self.position = (ix+1, iy)
        else:
            print('Can not go more down')

    def move_left(self):
        ix, iy = self.position
        if iy>0 and self.labyrinth[ix, iy-1]:
            self.position = (ix, iy-1)
        else:
            print('Can not go more left')

    def move_right(self):
        ix, iy = self.position
        if iy<self.width-1 and self.labyrinth[ix, iy+1]:
            self.position = (ix, iy+1)
        else:
            print('Can not go more right')

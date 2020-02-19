import gym
import numpy as np
import copy
import matplotlib.pyplot as plt
from gym import spaces


class PocMan(gym.Env):
    metadata = {'render.modes': ['human']}
    """
    Map code:
     `x`: wall
     `o`: food
     `<|>`: goes to other side
     `R|P|O|B`: ghost
    """

    def __init__(self,
                 observation_type='sparse_scalar',
                 harmless_ghosts=[],
                 ghost_random_move_prob=[0.,0.,0.,0.],
                 wall_place_prob=0,
                 food_place_prob=0.2):
        """
        Observation types:
        sparse_vector: 16 bit of RAM with certain bits zeroed out
        sparse_scalar: sparse_vector but binary encoded
        full_vector: 16 bit of RAM
        full_scalar: full_vector but binary encoded
        full_ascii: ASCII state with the map code defined below
        full_rgb: full_ascii but color-coded for convnets

        Map code:
        `x`: wall
        `o`: food
        `<|>`: goes to other side
        `R|P|O|B`: ghost
        """
        self.INIT_GAME_MAP = [
            ['x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x'],
            ['x',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','x'],
            ['x',' ','x','x',' ','x','x','x',' ','x',' ','x','x','x',' ','x','x',' ','x'],
            ['x','o',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','o','x'],
            ['x',' ','x','x',' ','x',' ','x','x','x','x','x',' ','x',' ','x','x',' ','x'],
            ['x',' ',' ',' ',' ','x',' ',' ',' ','x',' ',' ',' ','x',' ',' ',' ',' ','x'],
            ['x','x','x','x',' ','x','x','x',' ','x',' ','x','x','x',' ','x','x','x','x'],
            ['x','x','x','x',' ','x',' ',' ',' ',' ',' ',' ',' ','x',' ','x','x','x','x'],
            ['x','x','x','x',' ','x',' ','x',' ',' ',' ','x',' ','x',' ','x','x','x','x'],
            ['<',' ',' ',' ',' ','x',' ','x',' ',' ',' ','x',' ','x',' ',' ',' ',' ','>'],
            ['x','x','x','x',' ','x',' ','x','x','x','x','x',' ','x',' ','x','x','x','x'],
            ['x','x','x','x',' ','x',' ',' ',' ',' ',' ',' ',' ','x',' ','x','x','x','x'],
            ['x','x','x','x',' ','x',' ','x','x','x','x','x',' ','x',' ','x','x','x','x'],
            ['x',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','x'],
            ['x',' ','x','x',' ','x','x','x',' ','x',' ','x','x','x',' ','x','x',' ','x'],
            ['x','o',' ','x',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','x',' ','o','x'],
            ['x','x',' ','x',' ','x',' ','x','x','x','x','x',' ','x',' ','x',' ','x','x'],
            ['x',' ',' ',' ',' ','x',' ',' ',' ','x',' ',' ',' ','x',' ',' ',' ',' ','x'],
            ['x',' ','x','x','x','x','x','x',' ','x',' ','x','x','x','x','x','x',' ','x'],
            ['x',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','x'],
            ['x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x','x']
        ]
        self.observation_type = observation_type

        self.INIT_GHOST_POSES = [[7,8],[7,10],[9,8],[9,10]]
        self.GHOST_COLORS = [[255,0,0],[255,127,156],[255,186,0],[0,0,255]]
        self.INIT_PACMAN_POS = [13,9]
        self.ACTION_NAMES = np.array(['NORTH', 'EAST', 'WEST', 'SOUTH'])
        self.CHASE_PROB = 0.75
        self.DEFENSIVE_SLIP = 0.25
        self.NUM_ACTS = 4
        self.NUM_OBS = int(2**16)
        
        self.harmless_ghosts = harmless_ghosts
        self.ghost_random_move_prob = ghost_random_move_prob
        self.wall_place_prob = wall_place_prob
        self.food_place_prob = food_place_prob

        self._height = len(self.INIT_GAME_MAP)
        self._width = len(self.INIT_GAME_MAP[0])

        self.reset() # assigns observation_space too

        self.action_space = spaces.Discrete(4)

        if self.wall_place_prob > 0:
            self.placeWalls()
        self.placeFood()

        

    def step(self, action):
        """
        First move the ghosts, then move PacMan
        Encoder the observation using either the map, the 16-bit vector or the scalar (the binary vector translated into decimal notation)
        """
        self.moveGhosts()
        reward = self.movePacman(action)
        obs = self.select_obs()
        return obs, reward, self.inTerminalState, {}

    def reset(self):
        """
        Reset all variables to defaults
        """
        self.inTerminalState = False
        self.currImmediateReward = -1
        self.foodLeft = 0
        self.powerPillCounter = 0

        self.gameMap = np.array(self.INIT_GAME_MAP)
        self.h, self.w = self.gameMap.shape
        self.ghostPoses = np.array(self.INIT_GHOST_POSES)
        self.ghostDirs = [-1,-1,-1,-1]
        self.pacManPos = self.INIT_PACMAN_POS

        return self.select_obs()


    """
    Helper methods
    """
    def select_obs(self):
        if self.observation_type == 'full_vector':
            obs = self.getCurrentObservation()
            self.observation_space = gym.spaces.Box(
                                            low=0,
                                            high=1,
                                            shape=(16),
                                            dtype=np.uint8,
                                        )
        elif self.observation_type == 'full_scalar':
            obs = self.getCurrentObservation()
            self.observation_space = gym.spaces.Box(
                                            low=0,
                                            high=self.NUM_OBS,
                                            shape=(1),
                                            dtype=np.uint8,
                                        )
        elif self.observation_type == 'sparse_vector':
            obs = self.getCurrentObservationSparse()
            self.observation_space = gym.spaces.Box(
                                            low=0,
                                            high=1,
                                            shape=(16),
                                            dtype=np.uint8,
                                        )
        elif self.observation_type == 'sparse_scalar':
            obs = self.getCurrentObservationSparse()
            self.observation_space = gym.spaces.Box(
                                            low=0,
                                            high=self.NUM_OBS,
                                            shape=(1),
                                            dtype=np.uint8,
                                        )
        elif self.observation_type == 'full_ascii':
            obs = self.gameMap
            self.observation_space = gym.spaces.Box(
                                            low=0,
                                            high=255,
                                            shape=(self._height, self._width),
                                            dtype=np.str,
                                        )
        elif self.observation_type == 'full_rgb':
            obs = self.getCurrentObservationFullImage()
            self.observation_space = gym.spaces.Box(
                                            low=0,
                                            high=255,
                                            shape=(self._height, self._width, 3),
                                            dtype=np.uint8,
                                        )
        return obs

    def getCurrentObservationFullImage(self):
        """
        ASCII map color-coded as 3-d RGB tensor
        """
        screen = 128 * np.ones(self.gameMap.shape).reshape(self.gameMap.shape[0],self.gameMap.shape[1],1).repeat(3,axis=2).astype(int)
        for y,row in enumerate(self.gameMap):
            for x,symbol in enumerate(row):
                if symbol == 'x':
                    screen[y,x,:] = [0,0,0]
                if symbol == '.':
                    screen[y,x,:] = [128,255,128]
                if symbol == 'o':
                    screen[y,x,:] = [0,255,255]
        screen[self.pacManPos[0],self.pacManPos[1],:] = [255,255,0]
        for i,(y,x) in enumerate(self.ghostPoses):
            screen[y,x,:] = self.GHOST_COLORS[i]

        # if len(self.harmless_ghosts):
        #     for i in range(4):
        #         if i not in self.harmless_ghosts:
        #             screen[0,0,:] = self.GHOST_COLORS[i]
        return screen
        

    def placeFood(self):
        """
        For every grid, if it is empty, and w.p. P_food, place some food into that tile
        """
        self.foodLeft = 0
        for y in range(self.h-1):
            for x in range(self.w):
                if self.INIT_GAME_MAP[y][x] == ' ' and (np.random.uniform() < self.food_place_prob) and (y != self.pacManPos[0]) and (x != self.pacManPos[1]) and not (y > 6 and y < 12 and x > 5 and x < 13):
                    self.INIT_GAME_MAP[y][x] = '.'
                    self.foodLeft += 1

    def placeWalls(self):
        """
        For every grid, if it is empty, and w.p. P_walls, place a wall into that tile
        """
        for y in range(self.h-1):
            for x in range(self.w):
                if self.INIT_GAME_MAP[y][x] == ' ' and (np.random.uniform() < self.wall_place_prob) and (y != self.pacManPos[0]) and (x != self.pacManPos[1]) and not (y > 6 and y < 12 and x > 5 and x < 13):
                    self.INIT_GAME_MAP[y][x] = 'x'

    def computeIntFromBinary(self,arr):
        """
        Base 2 -> Base 10
        """
        result = 0
        for i,digit in enumerate(arr):
            if digit:
                result += 2**i
        return result

    def computeFoodManhattanDist(self):
        """
        Find the closest pile of food to the agent wrt L1 distance
        """
        min_dist = 5
        for y_diff in range(-4,4):
            for x_diff in range(-4,4):
                y = self.pacManPos[0]+y_diff
                x = self.pacManPos[1]+x_diff

                if (y > 0 and y < self.h-1) and (x > 0 and x < self.w-1):
                    if self.gameMap[y][x] == '.':
                        min_dist = min(min_dist, abs(y_diff) + abs(x_diff))
        return min_dist

    def computeNewStateInformation(self):
        """
        Compute the reward described in David Silver and Will's papers
        """
        l_reward = -1
        for i,ghost in enumerate(self.ghostPoses):
            if ghost[0] == self.pacManPos[0] and ghost[1] == self.pacManPos[1]:
                if self.powerPillCounter >= 0 or i in self.harmless_ghosts:
                    l_reward += 25
                    self.resetGhost(i)
                else:
                    l_reward -= 50
                    self.inTerminalState = True
        
        if self.gameMap[self.pacManPos[0],self.pacManPos[1]] == '.':
            self.gameMap[self.pacManPos[0],self.pacManPos[1]] = ' '
            l_reward += 100
            self.foodLeft -= 1
            if self.foodLeft == 0:
                self.inTerminalState = True
                l_reward += 1000
        elif self.gameMap[self.pacManPos[0],self.pacManPos[1]] == 'o':
            self.gameMap[self.pacManPos[0],self.pacManPos[1]] = ' '
            self.powerPillCounter = 15
        return l_reward

    def directionalDistance(self,lhs,rhs,dir_):
        """
        semi-distance which preserves the sign of the operation
        """
        if dir_ == 'NORTH':
            return lhs[0] - rhs[0]
        if dir_ == 'EAST':
            return rhs[1] - lhs[1]
        if dir_ == 'SOUTH':
            return rhs[0] - lhs[0]
        if dir_ == 'WEST':
            return lhs[1] - rhs[1]

    def getCurrentObservation(self):
        """
        Get the current state's observation for the agent
        """
        obs = np.zeros(16)
        y,x = self.pacManPos

        obs[0] = int(self.gameMap[y-1][x] == 'x')
        obs[1] = int(self.gameMap[y][x+1] == 'x')
        obs[2] = int(self.gameMap[y+1][x] == 'x')
        obs[3] = int(self.gameMap[y][x-1] == 'x')

        food_manhattan = self.computeFoodManhattanDist()

        if food_manhattan <= 2:
            obs[4] = 1
        elif food_manhattan <= 3:
            obs[5] = 1
        elif food_manhattan <= 4:
            obs[5] = 1

        """
        NORTH
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[0] -= 1
        while tempLoc[0] > 0 and tempLoc[1] < self.w and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[11] = 1
                    break
            if obs[11] == 1:
                break
            if self.gameMap[tempLoc[0],tempLoc[1]] == '.':
                obs[7] = 1
            tempLoc[0] -= 1

        """
        EAST
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[1] += 1
        while tempLoc[0] > 0 and tempLoc[1] < self.w and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[12] = 1
                    break
            if obs[12] == 1:
                break
            if self.gameMap[tempLoc[0],tempLoc[1]] == '.':
                obs[8] = 1
            tempLoc[1] += 1

        """
        SOUTH
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[0] += 1
        while tempLoc[0] < self.h-1 and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[13] = 1
                    break
            if obs[13] == 1:
                break
            if self.gameMap[tempLoc[0],tempLoc[1]] == '.':
                obs[9] = 1
            tempLoc[0] += 1

        """
        WEST
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[1] -= 1
        while tempLoc[1] > 0 and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[14] = 1
                    break
            if obs[14] == 1:
                break
            if self.gameMap[tempLoc[0],tempLoc[1]] == '.':
                obs[10] = 1
            tempLoc[1] -= 1

        obs[15] = int(self.powerPillCounter >= 0)

        if self.observation_type == 'full_vector':
            return obs
        if self.observation_type == 'full_scalar':
            return self.computeIntFromBinary(obs)
    
    def getCurrentObservationSparse(self):
        """
        Get the current state's observation for the agent, in sparse PocMan
        """
        obs = np.zeros(16)
        y,x = self.pacManPos

        obs[0] = int(self.gameMap[y-1][x] == 'x')
        obs[1] = int(self.gameMap[y][x+1] == 'x')
        obs[2] = int(self.gameMap[y+1][x] == 'x')
        obs[3] = int(self.gameMap[y][x-1] == 'x')

        # food_manhattan = self.computeFoodManhattanDist()

        # if food_manhattan <= 2:
        #     obs[4] = 1
        # elif food_manhattan <= 3:
        #     obs[5] = 1
        # elif food_manhattan <= 4:
        #     obs[5] = 1

        """
        NORTH
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[0] -= 1
        while tempLoc[0] > 0 and tempLoc[1] < self.w and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[4] = 1
                    break
            if obs[4] == 1:
                break
            tempLoc[0] -= 1

        """
        EAST
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[1] += 1
        while tempLoc[0] > 0 and tempLoc[1] < self.w and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[5] = 1
            if obs[5] == 1:
                break
            tempLoc[1] += 1

        """
        SOUTH
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[0] += 1
        while tempLoc[0] < self.h-1 and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[6] = 1
                    break
            if obs[6] == 1:
                break
            tempLoc[0] += 1

        """
        WEST
        """
        tempLoc = copy.deepcopy(self.pacManPos)
        tempLoc[1] -= 1
        while tempLoc[1] > 0 and self.gameMap[tempLoc[0],tempLoc[1]] != 'x':
            for ghost in self.ghostPoses:
                if tempLoc[0] == ghost[0] and tempLoc[1] == ghost[1]:
                    obs[7] = 1
                    break
            if obs[7] == 1:
                break
            tempLoc[1] -= 1

        obs[8] = int(self.powerPillCounter >= 0)

        if self.observation_type == 'sparse_vector':
            return obs
        if self.observation_type == 'sparse_scalar':
            return self.computeIntFromBinary(obs)

    def getCurrentObservationAction(self,action):
        """
        Get the current state's observation for the agent with the action
        """
        old_obs_type = self.observation_type
        self.observation_type = 'sparse_vector'
        obs = self.getCurrentObservationSparse()
        self.observation_type = old_obs_type

        if action == 0:
            a_vec = [0,0]
        elif action == 1:
            a_vec = [0,1]
        elif action == 2:
            a_vec = [1,0]
        elif action == 3:
            a_vec = [1,1]
        if self.observation_type == 'sparse_vector':
                return obs + a_vec
        if self.observation_type == 'sparse_scalar':
            return self.computeIntFromBinary(np.concatenate([obs,a_vec]))
    """
    Moves:
    [Ghosts]:
    -Offensive
    -Defensive
    -Random
    """

    def getValidMovements(self,y,x):
        """
        Make a list of valid movements for a given (x,y) position
        """
        moves = []
        if self.gameMap[y+1][x] != 'x':
            moves.append('SOUTH')
        if self.gameMap[y-1][x] != 'x':
            moves.append('NORTH')
        if self.gameMap[y][x-1] != 'x':
            moves.append('WEST')
        if self.gameMap[y][x+1] != 'x':
            moves.append('EAST')
        return np.array(moves)
    
    def makeMove(self,action,position):
        """
        Compute the next timestep position for the input (both for agent and ghosts)
        """
        if action == 'NORTH':
            position[0] -= 1
        if action == 'EAST':
            position[1] += 1
        if action == 'SOUTH':
            position[0] += 1
        if action == 'WEST':
            position[1] -= 1
        
        if self.gameMap[position[0],position[1]] == '<':
            position[1] = self.w - 2
        if self.gameMap[position[0],position[1]] == '>':
            position[1] = 1
        return position

    def movePacman(self,action):
        """
        Move the agent and compute the reward
        """
        action = self.ACTION_NAMES[action]
        moves = self.getValidMovements(self.pacManPos[0],self.pacManPos[1])
        if self.powerPillCounter >= 0:
            self.powerPillCounter -= 1
        if action not in moves:
            return self.computeNewStateInformation() -10
        else:
            self.pacManPos = self.makeMove(action,self.pacManPos)
            return self.computeNewStateInformation()

    def moveGhosts(self):
        """
        If the agent is close:
            -If the agent has the power pill, flee
            -If the agent has no power pill, follow it
        Else:
            -Random move
        """
        for i,ghost in enumerate(self.ghostPoses):
            u = np.random.uniform(size=(1,)).item()
            
            if ( abs(ghost[0]-self.pacManPos[0]) + abs(ghost[1]-self.pacManPos[1]) <= 5 ) and u > self.ghost_random_move_prob[i]:
                if self.powerPillCounter < 0:
                    # print('Aggressive')
                    self.ghostDirs[i] = self.moveGhostAggressive(i)
                else:
                    # print('Defensive')
                    self.ghostDirs[i]= self.moveGhostDefensive(i)
            else:
                # print('Random')
                self.ghostDirs[i] = self.moveGhostRandom(i)
            

    def moveGhostAggressive(self,i_ghost):
        """
        Go into the direction closest to the agent
        """
        best_dist = float('inf')
        best_dir = -1
        if np.random.uniform(1) < self.CHASE_PROB:
            moves = self.getValidMovements(self.ghostPoses[i_ghost][0],self.ghostPoses[i_ghost][1])
            for dir_ in self.ACTION_NAMES:
                if dir_ not in moves or dir_ == self.oppositeDir(self.ghostDirs[i_ghost]):
                    continue
                dist = self.directionalDistance(self.pacManPos,self.ghostPoses[i_ghost],dir_)
                if dist <= best_dist:
                    best_dir = dir_
        # print('Best dir (Ghost %d):%s'%(i_ghost,best_dir))
        if best_dir != -1:
            self.ghostPoses[i_ghost] = self.makeMove(best_dir,self.ghostPoses[i_ghost])
        else:
            self.moveGhostRandom(i_ghost)
        return best_dir

    def oppositeDir(self,dir_):
        if dir_ == 'NORTH':
            return 'SOUTH'
        if dir_ == 'SOUTH':
            return 'NORTH'
        if dir_ == 'EAST':
            return 'WEST'
        if dir_ == 'WEST':
            return 'EAST'
        return -1

    def moveGhostDefensive(self,i_ghost):
        """
        Move away from agent, in the furthest direction possible
        """
        best_dist = float('-inf')
        best_dir = -1
        if np.random.uniform(1) < self.DEFENSIVE_SLIP:
            moves = self.getValidMovements(self.ghostPoses[i_ghost][0],self.ghostPoses[i_ghost][1])
            for dir_ in self.ACTION_NAMES:
                if dir_ not in moves or dir_ == self.oppositeDir(self.ghostDirs[i_ghost]):
                    continue
                dist = self.directionalDistance(self.pacManPos,self.ghostPoses[i_ghost],dir_)
                if dist >= best_dist:
                    best_dir = dir_
        if best_dir != -1:
            self.ghostPoses[i_ghost] = self.makeMove(best_dir,self.ghostPoses[i_ghost])
        else:
            self.moveGhostRandom(i_ghost)
        return best_dir

    def moveGhostRandom(self,i_ghost):
        """
        Randomly execute a move
        """
        moves = self.getValidMovements(self.ghostPoses[i_ghost][0],self.ghostPoses[i_ghost][1])
        move = np.random.choice(moves,size=1)[0]

        while move == self.oppositeDir(self.ghostDirs[i_ghost]):
            move = np.random.choice(moves,size=1)[0]
        # if i_ghost == 2 and self.ghostPoses[i_ghost][0] == 9 and self.ghostPoses[i_ghost][1] == 10:
        #     print(move)
        #     print(moves)
        self.ghostPoses[i_ghost] = self.makeMove(move,self.ghostPoses[i_ghost])
        i_move = np.where(move==moves)
        return i_move

    def resetGhost(self,i_ghost):
        """
        Reset the ith ghost to its position (it died)
        """
        self.ghostPoses[i_ghost] = self.INIT_GHOST_POSES[i_ghost]


    """
    Utils:
     -pretty_print
    """

    def pretty_print(self):
        for row in self.gameMap:
            acc = ''
            for col in row:
                acc += col
            print(acc)

    def pretty_plot(self):
        screen = 128 * np.ones(self.gameMap.shape).reshape(self.gameMap.shape[0],self.gameMap.shape[1],1).repeat(3,axis=2).astype(int)
        for y,row in enumerate(self.gameMap):
            for x,symbol in enumerate(row):
                if symbol == 'x':
                    screen[y,x,:] = [0,0,0]
                if symbol == '.':
                    screen[y,x,:] = [128,255,128]
                if symbol == 'o':
                    screen[y,x,:] = [0,255,255]
        screen[self.pacManPos[0],self.pacManPos[1],:] = [255,255,0]
        for i,(y,x) in enumerate(self.ghostPoses):
            screen[y,x,:] = self.GHOST_COLORS[i]
        if len(self.harmless_ghosts):
            for i in range(4):
                if i not in self.harmless_ghosts:
                    screen[0,0,:] = self.GHOST_COLORS[i]

        fig,ax = plt.subplots(1)
        plt.imshow(screen)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == '__main__':
    import sys,tty,termios
    class _Getch:
        def __call__(self):
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(3)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch
    def get():
        inkey = _Getch()
        while(1):
                k=inkey()
                if k!='':break
        if k=='\x1b[A':
                return 0
        elif k=='\x1b[B':
                return 3
        elif k=='\x1b[C':
                return 1
        elif k=='\x1b[D':
                return 2

    env = PocMan(observation_type='full_rgb',ghost_random_move_prob=[1.,1.,1.,1.])
    x = env.reset()
    done = False
    plt.imshow(x)
    plt.pause(0.05)
    while not done:
        action = get()
        x, r, done , _ = env.step(action)
        print(env.ACTION_NAMES[action]+' '+str(r))
        plt.imshow(x)
        plt.pause(0.05)
    plt.show()
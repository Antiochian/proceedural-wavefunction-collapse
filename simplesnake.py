# -*- coding: utf-8 -*- python3
"""
Created on Wed Feb 12 18:28:36 2020

@author: Antiochian

VERY SIMPLE WAVEFUNCTION ALGORITHM
All this does is produce snake-like loop patterns in black and white out of
a set of tileable patterns and a hardcoded ruleset

its a proof-of-concept

"""
import numpy as np
import math
import time
import pygame
import sys


class block:
    """ Perhaps this class should really be called 'Tile', as it is a 3x3 square
    on the grid, which can exist in a superposition of multiple possible states
    until resolved by the wavefunction_collapse method """
    
    def __init__(self, block_loc):
        self.collapsed = False #start block as unresolved/uncollapsed
        self.block_loc = block_loc #coordinates on grid
        self.block_opts, self.block_weights = initialise_options() #possible states and their amplitudes
        self.arr = self.superposition()
        
    def superposition(self):
        """ Returns a 3x3 pixel array representing a superposition of all current
        probabilities for this block. This function is called every frame for multiple
        blocks and singlehandedly slows the program down by an enormous amount, 
        but its worth it for the cool animation effect"""
        superpos_array = [[0,0,0],[0,0,0],[0,0,0]]
        #check normalised:
        n = sum(self.block_weights)
        if n != 1:
            #normalise here if required
            self.block_weights = [x/n for x in self.block_weights]
        o = self.block_opts
        w = self.block_weights
        for i in range(TILE_SIZE):
            for j in range(TILE_SIZE):
                for k in range(len(o)):
                    superpos_array[j][i] += 254*get_blocks(o[k])[j][i]*w[k] 
                    
        return superpos_array
                
        #propgate changes out
    def propogate(self):
        """Start updating all surrounding uncollapsed blocks with new information"""
        X = len(grid[0])
        Y = len(grid)
        for DIR in [[1,0], [-1,0], [0,1], [0,-1]]:
            target_x, target_y = self.block_loc[0]+DIR[0], self.block_loc[1]+DIR[1]
            if 0 <= target_x < X and 0 <= target_y < Y: #if inbounds:
                target_block = grid[target_y][target_x]
                if not target_block.collapsed: #only ping uncollapsed blocks
                    self.send_update(target_block,DIR)
        return 
        
    def send_update(self, target_block, DIR):
        """Update the wavefunction of a block located DIR (vector) away from you"""
        new_opts = []
        new_weights = []
        if len(self.block_opts) != 1:
            raise Exception ("Improperly collapsed block!")
        i = self.block_opts[0] #our state
        for k in range(len(target_block.block_opts)): #k is their state
            #print("Checking ",i,k,DIR)
            if check_allowed(i,target_block.block_opts[k],DIR):
                new_opts.append(target_block.block_opts[k])
                new_weights.append(target_block.block_weights[k])
        target_block.block_opts = new_opts
        n = sum(new_weights)
        target_block.block_weights = [x/n for x in new_weights]
        target_block.block_weights = new_weights
        target_block.arr = target_block.superposition()
        return
    
    def collapse_wavefunction(self):
        """Randomly choose one of the block's current possible states to collapse into,
        depending on the probability amplitudes"""
        #check normalised:
        n = sum(self.block_weights)
        if n != 1:
            #normalise here if required
            self.block_weights = [x/n for x in self.block_weights]
        #make choice
        choice = np.random.choice(self.block_opts, p = self.block_weights)
        #update self accordingly
        self.block_opts = [choice]
        self.block_weights = [1]
        self.collapsed = True
        self.propogate()
        self.arr = self.superposition()
        return
    
    def debug_block(self):
        #prints summary, useful for debugging, not usually in use
        print("Block ID: ",self.block_loc)
        print("Collapsed? ",self.collapsed)
        print("Options: ",self.block_opts)
        print("Weights: ", self.block_weights)
        return

def initialise_options():
    """blocks start off with a default superposition of maximum entropy"""
    default_options = list(range(NUMBER_OF_TILES))
    default_weights = [1/NUMBER_OF_TILES]*NUMBER_OF_TILES
    return default_options, default_weights

def get_blocks(index):
    #call with -1 to get full blocklist
    """ This contains all the allowed "tiles" that can make up the image,
    and is currently hardcoded for this demo proof of concept
    
    0 = WALL (black) ; 1 = PATH (white)"""
    #the reason this is a function instead of just a list is that originally
    #i had plans to support dynamic tilesets, for example if only a certain
    #number of each tile were available. in the end this didnt happen though
    all_blocks = [
    [[0,0,0],[1,1,1],[0,0,0]], #0 - (horizontal passage)
    [[0,1,0],[0,1,0],[0,1,0]], #1 | (vertical passage)
    
    [[0,0,0],[1,1,0],[0,1,0]], #2 >v various L-junctions
    [[0,1,0],[1,1,0],[0,0,0]], #3 >^
    [[0,0,0],[0,1,1],[0,1,0]], #4 ^>
    [[0,1,0],[0,1,1],[0,0,0]], #5 v>
    
    [[0,0,0],[0,0,0],[0,0,0]], #6 0 empty
    [[0,1,0],[1,1,1],[0,1,0]], #7 + cross
    
    [[0,1,0],[1,1,1],[0,0,0]], #8  _|_ various T-junctions
    [[0,0,0],[1,1,1],[0,1,0]], #9   T
    [[0,1,0],[1,1,0],[0,1,0]], #10 -|
    [[0,0,0],[1,1,1],[0,0,0]]] #11  |-
    
#    [[0,1,0],[0,1,0],[0,0,0]], #12 #unsued "dead end" pieces
#    [[0,0,0],[0,1,0],[0,1,0]], #13
#    [[0,0,0],[0,1,1],[0,0,0]], #14
#    [[0,0,0],[1,1,0],[0,0,0]] ]#15
    if index == -1:
        return all_blocks
    else:
        return all_blocks[index] 

def rule_list():
    """an un-used function that make all the rules at once and puts them in a 
       dictionary for a faster lookup. I ended up checking rules on-demand instead,
       although this probably would have been better - i will use a hash table
       in the next iteration for sure"""
    #check RIGHT and DOWN borders
    all_blocks = get_blocks(-1)
    allowed = {}
    for i in range(len(all_blocks)): #index
        for j in range(len(all_blocks)):
            #check RIGHT border
            allowed[(i,j)] = [False,False]
            if all_blocks[i][1][2] == all_blocks[j][1][0]:
                allowed[(i,j)][0] = True
            #check DOWN border
            if all_blocks[i][2][1] == all_blocks[j][0][1]:
                allowed[(i,j)][1] = True
    return allowed

def check_allowed(i,j,DIR):
    """These rules are hardcoded for this demo, simply requiring neighbouring tiles to match colours"""
    #DIR is a unit vector pointing from i to j (eg: DIR = [0,1] indicates that j is 1 to the right of i)
    #check only specific arrangement on-demand
    if DIR == [1,0]: #i j #RIGHTWARD
        if get_blocks(i)[1][2] == get_blocks(j)[1][0]:
            return True
    elif DIR == [-1,0]: #LEFTWARD
        if get_blocks(j)[1][2] == get_blocks(i)[1][0]: #reverse indices
            return True
    elif DIR == [0,-1]: #UPWARD
        if get_blocks(j)[2][1] == get_blocks(i)[0][1]:
                return True    
    elif DIR == [0,1]: #DOWNWARD
        if get_blocks(i)[2][1] == get_blocks(j)[0][1]:
                return True
    else:
        raise ValueError ("Invalid DIR vector!")
    return False

def make_grid(X,Y):
    """ Initialises the system assuming zero information"""          
    grid = []
    for j in range(Y):
        row = []
        for i in range(X):
            row.append( block((i,j)) )
        grid.append(row)
    return grid

def get_min_shannon_entropy(grid):
    """calculate the Shannon Entropy of each uncollapsed state and choose 
       one of the lowest entroy states at random to target for  the next 
       wavefunction collapse event"""
    curr_min = math.inf
    curr_best = []
    for i in range(len(grid[0])):
        for j in range(len(grid)):
            if not grid[j][i].collapsed:
                w = grid[j][i].block_weights
                shannon_entropy = sum([-math.log(el) for el in w] )
                if shannon_entropy < curr_min:
                    curr_min = shannon_entropy
                    curr_best = [(i,j)]
                elif shannon_entropy == curr_min:
                    curr_best.append((i,j))
    idx = np.random.choice(range(len(curr_best))) #choose randomly if theres a tie
    return curr_best[idx] #x,y

def check_done(grid):
    """check for uncollapsed states (inefficient to say the least)"""
    for row in grid:
        for el in row:
            if not el.collapsed:
                return False
    else:
        return True
            
def render_image(grid,window):
    """draws screen to pygame surface"""
    X = len(grid[0])
    Y = len(grid)
#top row:
    for j in range(Y):
        for sub_j in range(3): #3 rows 
            ROW = []
            for i in range(X):
                ROW += grid[j][i].arr[sub_j]
            
            for k in range(len(ROW)):
                COLOR = (ROW[k],ROW[k],ROW[k])
                Y_pos = (3*j + sub_j)*pixel_size*scale
                X_pos = k*(pixel_size)*scale
                width = height = pixel_size*scale
                pygame.draw.rect(window,COLOR,(X_pos,Y_pos,width,height))
            
#            print(ROW)
    return 

def animate():
    """game/animation loop"""
    while not check_done(grid):
        #ANIMATION LOOP:
        clock.tick(FPS)
        for event in pygame.event.get(): #detect events
            #QUIT DETECTION (Esc key or corner button)
            if event.type == pygame.QUIT or pygame.key.get_pressed()[27]:
                pygame.quit()
                sys.exit()     
            if pygame.key.get_pressed()[114]:
                return True #indicates animation was interrupted early
        #actual algorithm
        (x,y) = get_min_shannon_entropy(grid)
        grid[y][x].collapse_wavefunction()
        render_image(grid,window)
        left = (x-1)*tile_size*scale
        top = (y-1)*tile_size*scale
        width = height = 3*tile_size*scale
        pygame.display.update(    (left, top, width, height))
    return False #indicates animation is finished


"""Main driver code is here"""

#CONSTANT SETUP
NUMBER_OF_TILES = len(get_blocks(-1))
TILE_SIZE = len(get_blocks(0)[0]) 

WALL = (0,0,0)
PATH = (255,255,255)

X = 15
Y = 15
pixel_size = 3
tile_size = pixel_size*3 #each tile is 3x3 pixels
aspect_ratio = X//Y
scale = 600//max(X*tile_size,Y*tile_size) #ensure window is no bigger than 600px on any side
Nx,Ny = scale*X*tile_size,scale*Y*tile_size
FPS = 100

pygame.init()

window = pygame.display.set_mode( (Nx,Ny) )
clock = pygame.time.Clock()

#setup
global grid
grid = make_grid(X,Y)
play_animation = True
render_image(grid,window)
pygame.display.update()
while True:
    if play_animation:
        grid = make_grid(X,Y)
        render_image(grid,window)
        pygame.display.update()
        play_animation = animate()
    #ANIMATION LOOP:
    clock.tick(FPS)
    for event in pygame.event.get(): #detect events
        #QUIT DETECTION (Esc key or corner button)
        if event.type == pygame.QUIT or pygame.key.get_pressed()[27]:
            pygame.quit()
            sys.exit()
        if pygame.key.get_pressed()[114]:
            play_animation = True





########################################
"""UNUSED FUNCTIONS (For debugging and timing):"""


def render_text(grid):
    """ 'draws' current state to console window (for debugging)"""
    X = len(grid[0])
    Y = len(grid)
#top row:
    for j in range(Y):
        for sub_j in range(3): #3 rows 
            ROW = []
            for i in range(X):
                ROW += grid[j][i].arr[sub_j]
            print(ROW)
            
def command_line(X=3,Y=3,visible=True):
    """ version of main() that prints out to console, useful for debugging """
    #visible flag can be set to False for timing purposes
    if X == 0 or Y == 0:
        return
#    np.random.seed(1)
    global grid
    grid = make_grid(X,Y)
    while not check_done(grid):
        (x,y) = get_min_shannon_entropy(grid)
        #print("Collapsing ",grid[y][x].block_loc)
        grid[y][x].collapse_wavefunction()
    if visible:
        render_text(grid)
        #print("---"*5)
    return
    
def timings(nmax=10,nstart=0):
    """function to perform speedtests with increasing NxN grids
   The time complexity is O(n^3) - ever since i implemented the horribly inefficient
   partial-superposition visualisation, i completely gave up on speed - this is
   just a proof of concept and not worth optimising when im going to rewrite 
   the whole thing later anyway"""
    times = []
    for N in range(nstart,nmax):
        t0 = time.time()
        command_line(N,N,False)
        times.append(time.time()-t0)
        print("#",N," OK")
    print("\n  N |  time (seconds)")
    print("-----------------------")
    for n in range(nstart,nmax):
        print(" ",n," | ",times[n])
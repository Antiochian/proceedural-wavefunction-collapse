# -*- coding: utf-8 -*- python3
"""
Created on Thu Feb 13 14:40:00 2020

@author: Antiochian

intermediate snake

this version should be able to extrapolate rules from an input image, rather
than having them hard-coded

it still contains hardcoded 2x2 "tiles" though, however
"""
from copy import deepcopy
import numpy as np
from PIL import Image
import math
import time

global recursion_nums
recursion_nums = 0

class tile:
    def __init__(self,loc, init_amplitudes,markov,image_dims):
        self.loc = loc
        self.opts = init_amplitudes           #could pull this out into a function but since all tiles are built at the same time its not a good idea
        self.markov = markov            #ditto here, although i think the dictionary is references, so thats actually totally fine maybe?
        self.image_dims = image_dims    #stores XY dimensions of overall image (in TILES, not pixels)
        self.neighbours = self.get_neighbour_list() #list of all valid adjacent blocks (i.e. not edges)
        self.collapsed = False
        
    def get_neighbour_list(self):
        X_TILES, Y_TILES = self.image_dims
        number_of_neighbours = 0
        neighbour_list = []
        for Dx in range(-1,2):
            for Dy in range(-1,2):
                DIR = (Dx,Dy)
                if DIR != (0,0): #i think there is a better way to do this
                    target_x, target_y = self.loc[0] + DIR[0], self.loc[1] + DIR[1]
                    if (0 <= target_x < X_TILES and 0 <= target_y < Y_TILES):  #only proceed if target is in-bounds
                        number_of_neighbours += 1
                        neighbour_list.append( (target_x,target_y) )
        return neighbour_list
    
    def update(self,recursive=0):
        global recursion_nums
        if recursive:
            #recursing
            for target_loc in self.neighbours:
                recursion_nums += 1
                obj_dict[target_loc].update(recursive=recursive-1)
        elif self.collapsed:
            if recursive:
            #recursing
                for target_loc in self.neighbours:
                    recursion_nums += 1
                    obj_dict[target_loc].update(recursive=recursive-1)
            return
        #opts can never INCREASE, so we can take our current opts as a superset
        new_opts = self.opts.copy()  #NB: I think a shallow copy is sufficient, but its worth checking this later tbh
        for target_loc in self.neighbours:
            target = obj_dict[target_loc]
            #get relative DIR pointing FROM self TO target
            DIR = (target_loc[0] - self.loc[0], target_loc[1] - self.loc[1])
            """ 
            take an option in target (targ_opt)
            look at its probability
            multiply it by the overall targ_weight, thats its effect on self
            add to running total IF its compatible with current allowed options (normalise later)
            """
            if target.collapsed:
                targ_key = target.final_state
                for possible_outcome in new_opts:
                    if DIR in self.markov[targ_key]: #just ignore if if theres no data (hmm, doesnt seem right..)
                        if possible_outcome in self.markov[targ_key][DIR]:
                            markov_prob = self.markov[targ_key][DIR][possible_outcome]
                            if possible_outcome in self.opts: #This is a really unsafe assumption, that these states can be ignored, but oh well
                                new_opts[possible_outcome] += markov_prob*1
            elif False: #OLD SYSTEM
                raise Exception("Get out of here!")
                for targ_key in target.opts:
                    #all possible states of target
                    targ_weight =  target.opts[targ_key] #chance of being in that state
                    for possible_outcome in new_opts:
                        if possible_outcome in self.markov[targ_key][DIR]:
                            markov_prob = self.markov[targ_key][DIR][possible_outcome]
                            if possible_outcome in self.opts: #This is a really unsafe assumption, that these states can be ignored, but oh well
                                new_opts[possible_outcome] += markov_prob*targ_weight           
        N = sum(new_opts.values()) #normalise new amplitudes:
        for key in new_opts:
            new_opts[key] /= N
        #update
        self.opts = new_opts
        if len(self.opts.values()) == 1:
            self.collapse_wavefunction()
#        if recursive:
#            #recursing
#            for target_loc in self.neighbours:
#                recursion_nums += 1
#                obj_dict[target_loc].update(recursive=recursive-1)
        return
        
    def collapse_wavefunction(self):
        """ pick an option randomly according to weighted probabilities"""
        allowed_states = list(self.opts.keys())
        probability_amplitudes = list(self.opts.values())
        choice = np.random.choice(allowed_states,p=probability_amplitudes)
        self.opts = {choice : 1}
        self.collapsed = True
        self.final_state = choice
        return
        
def make_tile_dict(TILE_SIZE):
    """ This generates a big dictionary of every possible tile given a certain dimensional size (2x2, 3x3, etc) """
    tile_dict = {}
    for ID in range(2**(TILE_SIZE**2)): #BIT MASK FOR TILE PERMUTATIONS
        bi = '{0:0200b}'.format(ID) #binary representaiton of ID (for permutations)
        bi = bi[-(TILE_SIZE**2):]
        tile_arr = [[None]*TILE_SIZE for _ in range(TILE_SIZE)]
        for n in range(0,TILE_SIZE**2): #for each element
            j,i = divmod(n,TILE_SIZE) #get coords
            tile_arr[j][i] = int(bi[n])
        tile_dict[ID] = tile_arr
        """
        The subtlety here is that the ID is an integer (1 to n^2) for human readability, but 
        can also be recovered by flattening an array into a binary number and converting,
        so no costly search in dict operations are required (NICE) when generating markov chains
        i am a genius
        """
    return tile_dict

def png_to_matrix(TILE_SIZE,filename="test1.png"):
    im = Image.open(filename)
    im = im.convert('RGB')
    width, height = im.size
    X_TILES = width // TILE_SIZE #crop side and bottom to fit tilesize
    Y_TILES = height // TILE_SIZE
    tile_ID_matrix = [[None]*X_TILES for _ in range(Y_TILES)]
    pixel = im.load() #pixelAccess object
    #load as TILES (not pixels)
    blank_tile = [[None]*TILE_SIZE for _ in range(TILE_SIZE)] #blank tile array
    for x in range(X_TILES):
        for y in range(Y_TILES):
            current_tile = blank_tile.copy()
            for i in range(TILE_SIZE):
                for j in range(TILE_SIZE):
                    coords = (TILE_SIZE*x + i, TILE_SIZE*y + j)
                    """ Temporary bodge for black/white case only, color detection can come later """
                    if pixel[coords] == (0,0,0):
                        current_tile[j][i] = 0
                    else:
                        current_tile[j][i] = 1
            #flatten current_tile
            #tile_binary = [k for k in current_tile]
            tile_binary = "".join(map(str,[k for sublist in current_tile for k in sublist]))
            tile_ID_matrix[y][x] = int(tile_binary,2)
    return tile_ID_matrix

def partial_tile_dict(tile_ID_matrix,TILE_SIZE):
    tile_dict = {}
    curr_tile = [[None]*TILE_SIZE for _ in range(TILE_SIZE)]
    all_IDs = list(set([k for sublist in tile_ID_matrix for k in sublist]))
    for ID in all_IDs:
        if ID not in tile_dict:
            binary_ID = '{0:0200b}'.format(ID) #binary representaiton of ID (for permutations)
            binary_ID = binary_ID[-(TILE_SIZE**2):]
            for i in range(TILE_SIZE):
                for j in range(TILE_SIZE):
                    curr_tile[j][i] = int(binary_ID[TILE_SIZE*j + i])
            tile_dict[ID] = deepcopy(curr_tile)

    return tile_dict


def make_test_markov(tile_ID_matrix):
    #for now, border behaviour is PERIODIC (wrap surface edges around to meet)
    X_TILES = len(tile_ID_matrix[0])
    Y_TILES = len(tile_ID_matrix)
    
    markov = {}
    print("Markov Initialised. Continuing...")
    #ZERO OFFSET SYSTEM:
    for x in range(X_TILES):
        for y in range(Y_TILES): #select a square
            for Dx in range(-1,2):
                for Dy in range(-1,2):
                    DIR = (Dx,Dy)
                    if DIR != (0,0): #must be a better way to do this, but as a one-time function call its alriht
                        target_x, target_y = x + DIR[0], y + DIR[1]
                        if (0 <= target_x < X_TILES and 0 <= target_y < Y_TILES):  #only proceed if target is in-bounds                      
                            current_ID = tile_ID_matrix[y][x]
                            target_ID = tile_ID_matrix[target_y][target_x]
                            
                            if current_ID not in markov:
                                markov[current_ID] = {}
                            if DIR not in markov[current_ID]:
                                markov[current_ID][DIR] = {}
                            if target_ID in markov[current_ID][DIR]: #UPDATE MARKOV
                                markov[current_ID][DIR][target_ID] += 1 #left
                            else:
                                markov[current_ID][DIR][target_ID] = 1
            
    return markov

def populate_obj_dict(tile_ID_matrix,markov, OUTPUT_SIZE):
    X_TILES = OUTPUT_SIZE[0]
    Y_TILES = OUTPUT_SIZE[1]
    
    default_amp = get_defaults(tile_ID_matrix)
    for i in range(X_TILES):
        for j in range(Y_TILES):
            obj_dict[(i,j)] = tile((i,j),default_amp,markov, (X_TILES, Y_TILES))
    return


def get_defaults(tile_ID_matrix):
    """ Uses bulk appearance data to set up initial (general) superposition of states """
    #create freq dict
    freq_dict = {}
    flattened = [k for sublist in tile_ID_matrix for k in sublist]
    total_sum = 0
    for el in flattened:
        if el in freq_dict:
            freq_dict[el] += 1
            total_sum +=1 #this SHOULD end up being the area of the matrix but lets just make sure
        else:
            freq_dict[el] = 1
            total_sum +=1
    for key in freq_dict.keys(): #NORMALISE
        freq_dict[key] /= total_sum
    return freq_dict #here serves as "default amplitdues" data

def get_min_shannon_entropy():
    """calculate the Shannon Entropy of each uncollapsed state and choose 
       one of the lowest entroy states at random to target for  the next 
       wavefunction collapse event"""
    curr_min = math.inf
    curr_best = []
    for key in obj_dict:
        if not obj_dict[key].collapsed:
            w = obj_dict[key].opts.values()
            w = [x for x in w if x > 0] #filter out 0-weight options (shouldnt exist)
            shannon_entropy = sum([-math.log(el) for el in w] )
            if shannon_entropy < curr_min:
                curr_min = shannon_entropy
                curr_best = key
#            elif shannon_entropy == curr_min:
#                curr_best.append(key)
#    idx = np.random.choice(range(len(curr_best))) #choose randomly if theres a tie
    return curr_best #x,y

def check_done():
    """check for uncollapsed states (inefficient to say the least)"""
    for key in obj_dict:
        if not obj_dict[key].collapsed:
            return False
    else:
        return True

def print_state(OUTPUT_SIZE):
    """ very crude function that prints current state to console"""
    Nx,Ny = OUTPUT_SIZE
    state_arr = [[None]*Nx for _ in range(Ny)]
    for key in obj_dict:
        if obj_dict[key].collapsed:
            state_arr[key[1]][key[0]] = str(obj_dict[key].final_state)
        else:
            state_arr[key[1]][key[0]] = "X"
    print(state_arr)
    return

def output_png(TILEDICT,OUTPUT_SIZE,TILE_SIZE,filename="output.png"):
    """ saves state to a png file """
    Nx,Ny = OUTPUT_SIZE
    
    output_arr = [[None]*Nx*TILE_SIZE for _ in range(Ny*TILE_SIZE)] 
    for i in range(Nx):
        for j in range(Ny):
            key = (i,j)
            if obj_dict[key].collapsed:
                tile_arr = TILEDICT[obj_dict[key].final_state]
                for di in range(TILE_SIZE):
                    for dj in range(TILE_SIZE):
                        x = TILE_SIZE*i + di
                        y = TILE_SIZE*j + dj
                        val = tile_arr[dj][di]
                        if val == 1:
                            output_arr[y][x] = (255,255,255)
                        else:
                            output_arr[y][x] = (0,0,0)
            else:
                """default_tile goes here (if desired)"""
                for di in range(TILE_SIZE):
                    for dj in range(TILE_SIZE):
                        x = TILE_SIZE*i + di
                        y = TILE_SIZE*j + dj
                        output_arr[y][x] = (127,127,127)
    #flatten array (for image conversion)
    output_arr = [k for sublist in output_arr for k in sublist]
    pixel_size = (TILE_SIZE*OUTPUT_SIZE[0],TILE_SIZE*OUTPUT_SIZE[1])
    new_image = Image.new('RGB',pixel_size)
    new_image.putdata(output_arr)
    new_image.save(filename)     
    return
        
#def setup(Nx,Ny,filename = 'test4.png'):
def main(TILE_SIZE=5, MAX_RECURSION_DEPTH=3,PIXEL_SIZE=(70,70),output_filename="output.png", input_filename="horiz_test.png"):  
    MAX_BIN_LENGTH = 50 #hardcoded inside make_tile_dict funtion
    global obj_dict 
    obj_dict = {}
    #TILEDICT = make_tile_dict(TILE_SIZE)
    print("Making ID matrix...")
    IDMATRIX = png_to_matrix(TILE_SIZE,filename = input_filename)
    print("Making tile permutation dictionary...")
    TILEDICT = partial_tile_dict(IDMATRIX,TILE_SIZE)
    print("Making Markov chain...")
    MARKOV = make_test_markov(IDMATRIX)
    OUTPUT_SIZE = (PIXEL_SIZE[0]//TILE_SIZE,PIXEL_SIZE[1]//TILE_SIZE)
    print("Initialising objects...")
    populate_obj_dict(IDMATRIX,MARKOV, OUTPUT_SIZE)
    number_of_states = OUTPUT_SIZE[0]*OUTPUT_SIZE[1]
    steps = 0
    while not check_done():
        if steps%(number_of_states//10) == 0:
                percent = steps*100//number_of_states
                print("\rWorking...",percent,"%",end="")
        next_target = get_min_shannon_entropy()
        obj_dict[next_target].collapse_wavefunction()
        obj_dict[next_target].update(recursive=MAX_RECURSION_DEPTH)
        steps += 1
    print("\nTilesize: ",TILE_SIZE," MRD: ",MAX_RECURSION_DEPTH)
    print('\nNumber of Recursions:',recursion_nums)
    output_png(TILEDICT,OUTPUT_SIZE,TILE_SIZE,output_filename)

def batch_test():
    input_filename="dot_test.png"
    trial = 1
    filepath= ".\\batch\\"
    for ts in range(5,25,5):
            mrd = 70//(1.4142*ts) #square root mean free path
            trialname = filepath+"trial"+str(trial)+"-TS"+str(ts)+"MRD"+str(mrd)+".png"
            main(TILE_SIZE= ts, MAX_RECURSION_DEPTH= mrd, input_filename = input_filename, output_filename = trialname)
            print("-"*20,"\n\t",trial, "OK\n","-"*20)
            trial += 1

def timings(nmax=10,nstart=2):
    """function to perform speedtests with increasing NxN grids
   The time complexity is O(n^3) - ever since i implemented the horribly inefficient
   partial-superposition visualisation, i completely gave up on speed - this is
   just a proof of concept and not worth optimising when im going to rewrite 
   the whole thing later anyway"""
    times = []
    TILE_SIZE = 3
    for N in range(nstart*TILE_SIZE,nmax*TILE_SIZE,TILE_SIZE):
        t0 = time.time()
        mrd = 70//(1.4142*TILE_SIZE)
        main(TILE_SIZE,mrd,PIXEL_SIZE=(N,N))
        times.append(time.time()-t0)
        print("#",N," OK")
    print("\n  N |  time (seconds)")
    print("-----------------------")
    print(times)
    for n in range(nstart,nmax):
        print(" ",n," | ",times[n-nstart])


"""
note to self
STRUCTURE OF MARKOV DICTIONARY

markov[current_tile] = [L:subdict R:subdict U:subdict D:subdict]
this results in a double-redunancy of memory usage, but reduces risk of errors 
compraed to an [R: D:] dict like we had previously

Each subdict can also be built seperately and then easily collated into the 
masterdict afterwards

I think this is the plan I will go with for now

"""    
from selfies import split_selfies
import numpy as np


branch_tokens=[f"[Branch{i}]" for i in range(1,4)]
branch_tokens+=[f"[=Branch{i}]" for i in range(1,3)]
branch_tokens+=[f"[#Branch{i}]" for i in range(1,3)]
ring_tokens=set([f"[Ring{i}]" for i in range(1,4)])

num_dict={
        "[C]" : 0,
        "[Ring1]" : 1,
        "[Ring2]" : 2,
        "[Branch1]" : 3,
        "[=Branch1]" : 4,
        "[#Branch1]" : 5,
        "[Branch2]" : 6,
        "[=Branch2]" : 7,
        "[#Branch2]" : 8,
        "[O]" : 9,
        "[N]" : 10,
        "[=N]" : 11,
        "[=C]" : 12,
        "[#C]" : 13,
        "[S]" : 14,
        "[P]" : 15
    }

def selfies_split(selfies):
    return [block+"]" for block in selfies.split("]")][:-1]

def get_numeric_value(token):
    if token not in num_dict.keys():
        print("Not a valid number token: " + token)
        value = None
    else:
        value = num_dict[token]
    return value


def computeHexNumber(number_tokens):
    sum = 0
    digits = len(number_tokens)
    for i in range(digits):
        sum += get_numeric_value(number_tokens[i]) * (16 ** (digits - i - 1))
    sum = sum + 1 #idk why the + 1 is needed but it is even though it is not in the docs
    return sum


#Adding all possible edges from nodes in list1 to nodes in list2
def fullyConnect(list1, list2, adj_matrix, value, bidirectional=False):
    for i in list1:
        for j in list2:
            adj_matrix[i, j] = value
            if bidirectional:
                adj_matrix[j, i] = value


def processAtoms(prev_atom_idx, cur_idx, num_tokens_to_process, tokens, adj_matrix, atom_list):
    start_idx = cur_idx
    while cur_idx <= start_idx + num_tokens_to_process:
        cur_token = tokens[cur_idx]
        if cur_token in branch_tokens:
            cur_idx = processBranch(prev_atom_idx, cur_idx, tokens, adj_matrix, atom_list)
        elif cur_token in ring_tokens:
            cur_idx = processRing(prev_atom_idx, cur_idx, tokens, adj_matrix, atom_list)
        else: #cur_token is an atom

            atom_list.append(cur_idx)

            #Adding molecular bond edges
            adj_matrix[cur_idx, prev_atom_idx] = 1
            adj_matrix[prev_atom_idx, cur_idx] = 1
            
            prev_atom_idx = cur_idx
            cur_idx += 1


def processBranch(prev_atom_idx, start_idx, tokens, adj_matrix, atom_list):
    #print("enter branch")

    branch_token = tokens[start_idx]

    #find number tokens and next atom
    if branch_token[-2] == "1":
        number_tokens = [tokens[start_idx+1]]
        next_atom_idx = start_idx + 2
        number_token_idxs = [start_idx+1]
    elif branch_token[-2] == "2":
        number_tokens = tokens[start_idx+1:start_idx+3]
        next_atom_idx = start_idx + 3
        number_token_idxs = [start_idx+1, start_idx+2]
    elif branch_token[-2] == "3":
        number_tokens = tokens[start_idx+1:start_idx+4]
        next_atom_idx = start_idx + 4
        number_token_idxs = [start_idx+1, start_idx+2, start_idx+3]
    else:
        print("Invalid branch token: " + branch_token)
    
    
    atom_list.append(next_atom_idx)

    #Adding molecular edges
    adj_matrix[prev_atom_idx, next_atom_idx] = 1
    adj_matrix[next_atom_idx, prev_atom_idx] = 1

    num = computeHexNumber(number_tokens)
    branch_tokens_idxs = [next_atom_idx + j for j in range(num)]
    num_tokens_to_process = num - 2
    
    
    #print("branch token: " + str(start_idx) + " connected to number tokens " + str(number_token_idxs))
    #print("nuimber tokens " + str(number_token_idxs) + "connected to themselves")
    #print("number tokens, " + str(number_token_idxs) + "  connected to tokens in branch " + str(branch_tokens_idxs))
    getGrammarEdges1([start_idx], number_token_idxs, branch_tokens_idxs, adj_matrix)

    #process rest of atoms in branch
    processAtoms(next_atom_idx, next_atom_idx+1, num_tokens_to_process, tokens, adj_matrix, atom_list)
    
    token_after_branch_idx = next_atom_idx + num

    #print("exit branch")
    return token_after_branch_idx


def processRing(prev_atom_idx, start_idx, tokens, adj_matrix, atom_list):
    #print("enter ring")
    ring_token = tokens[start_idx]
    if ring_token[-2] == "1":
        number_tokens = [tokens[start_idx+1]]
        next_idx = start_idx + 2
        number_token_idxs = [start_idx+1]
    elif ring_token[-2] == "2":
        number_tokens = tokens[start_idx+1:start_idx+3]
        next_idx = start_idx + 3
        number_token_idxs = [start_idx+1, start_idx+2]
    elif ring_token[-2] == "3":
        number_tokens = tokens[start_idx+1:start_idx+4]
        next_idx = start_idx + 4
        number_token_idxs = [start_idx+1, start_idx+2, start_idx+3]
    else:
        print("Invalid branch token: " + ring_token)  
    
    num = computeHexNumber(number_tokens)

    if len(atom_list) >= num+1:
        start_ring_atom_idx = atom_list[-num-1]
    else:
        print("Ring of size " + str(num+1) + " too big for atom list of size " + str(len(atom_list)))
        print(atom_list)
    
    ring_atom_idxs = [prev_atom_idx, start_ring_atom_idx] #list of two atoms closing ring

    #Adding molecular edges
    adj_matrix[prev_atom_idx, start_ring_atom_idx] = 1
    adj_matrix[start_ring_atom_idx, prev_atom_idx] = 1

    #print("ring token: " + str(start_idx) + " connected to number tokens " + str(number_token_idxs))
    #print("nuimber tokens " + str(number_token_idxs) + "connected to themselves")
    #print("number tokens, " + str(number_token_idxs) + "  connected to tokens joining ring" + str(ring_atom_idxs))
    getGrammarEdges1([start_idx], number_token_idxs, ring_atom_idxs, adj_matrix)
    
    #print("exit ring")

    return next_idx



def getGrammarEdges1(branch_or_ring_idx, number_token_idxs, other_token_id, adj_matrix):
    value = 1
    fullyConnect(branch_or_ring_idx, number_token_idxs, adj_matrix, value, bidirectional=True) #Connect branch/ring to number tokens
    fullyConnect(number_token_idxs, number_token_idxs, adj_matrix, value, bidirectional=True) #Connect digits in hex number to each other
    fullyConnect(number_token_idxs, other_token_id, adj_matrix, value, bidirectional=True) #Connect number tokens to tokens in branch or to tokens closing ring


def getAdjMatrixFromSelfie(selfie, length, c=0.3):

    #preprocess
    tokens = list(selfies_split(selfie))
    start_token = tokens[0]

    #initializing
    adj_matrix = np.full((len(tokens), len(tokens)), c)
    atom_list = []

    if start_token in branch_tokens or start_token in ring_tokens:
        print("Error: seflie must start with atom token")
    else:
        atom_list.append(0)
        processAtoms(0, 1, len(tokens)-2, tokens, adj_matrix, atom_list)

    #Add ones along diagonal
    for i in range(adj_matrix.shape[0]):
        adj_matrix[i, i] = 1

    full_matrix = np.full((length, length), c)
    full_matrix[1:1+len(tokens), 1:1+len(tokens)] = adj_matrix # added start and padding tokens

    return atom_list, full_matrix


#atom_list, full_matrix = getAdjMatrixFromSelfie("[C][C][=C][C][=C][Branch1][Branch1][C][=C][Ring1][=Branch1][C]", 20, c=0.0)
#print(atom_list)
#print(full_matrix)
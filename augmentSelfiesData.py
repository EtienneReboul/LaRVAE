import pandas as pd
import numpy as np
import pickle as pkl


sf_tok_list =['[C]', '[S]', '[=Branch1]', '[=O]', '[=C]', '[NH1]', '[Branch2]', '[=N]', 
              '[O]', '[Ring1]', '[#Branch2]', '[Branch1]', '[Cl]', '[#Branch1]', '[N]',
              '[Ring2]', '[=Branch2]', '[#C]', '[F]', '[#N]', '[P]', '[Br]', '[=S]']
num_dict={
        "[0]" : 0,
        "[1]" : 1,
        "[2]" : 2,
        "[3]" : 3,
        "[4]" : 4,
        "[5]" : 5,
        "[6]" : 6,
        "[7]" : 7,
        "[8]" : 8,
        "[9]" : 9,
        "[10]" : 10,
        "[11]" : 11,
        "[12]" : 12,
        "[13]" : 13,
        "[14]" : 14,
        "[15]" : 15
    }

branch_ring_dict = {
        "[Ring1]" : 1,
        "[Ring2]" : 2,
        "[Branch1]" : 1,
        "[=Branch1]" : 1,
        "[#Branch1]" : 1,
        "[Branch2]" : 2,
        "[=Branch2]" : 2,
        "[#Branch2]" : 2,
}

overloading_dict = {
    0 : "[C]",
    1 : "[Ring1]",
    2 : "[Ring2]",
    3 : "[Branch1]",
    4 : "[=Branch1]",
    5 : "[#Branch1]",
    6 : "[Branch2]",
    7 : "[=Branch2]",
    8 : "[#Branch2]",
    9 : "[O]",
    10 : "[N]",
    11 : "[=N]",
    12 : "[=C]",
    13 : "[#C]",
    14 : "[S]",
    15 : "[P]"
}
def get_numeric_value(token):
    if token not in num_dict.keys():
        print("Not a valid number token: " + token)
        raise RuntimeError
        value = None
    else:
        value = num_dict[token]
    return value

def computeHexNumber(number_tokens):
    sum = 0
    digits = len(number_tokens)
    for i in range(digits):
        sum += get_numeric_value(number_tokens[i]) * (16 ** (digits - i - 1))
    return sum

def tokenizer(mol):
    tokens=[block+"]" for block in mol.split("]")][:-1]
    return tokens


        
def convert(selfie):
    new_tokens = []
    tokens = tokenizer(selfie)
    i = 0
    while i < len(tokens):
        first_num_tok = num_dict.get(tokens[i], 0)
        if first_num_tok != 0:
            num_number_toks = branch_ring_dict.get(tokens[i-1], 0)
            if num_number_toks == 0:
                print("problem with"  + ''.join(tokens))
                break
            else:
                num_toks = tokens[i:i+num_number_toks]
                new_num_tok = "[" + str(computeHexNumber(num_toks)) +"]"
                new_tokens.append(new_num_tok)
                i = i+num_number_toks
        else:
            new_tokens.append(tokens[i])
            i+=1
    return ''.join(new_tokens)


def unHex(hexed):
    selfie_tokens = []
    tokens = tokenizer(hexed)
    for token in tokens:
        num_toks = []
        if token not in sf_tok_list:
            hex_num = hex(int(token[1:-1]))[2:] #delete "0x" from beginning
            for digit in hex_num:
                print(digit)
                num_toks.append(overloading_dict[int(digit, 16)])
            selfie_tokens += num_toks
        else: selfie_tokens.append(token)
    
    return ''.join(selfie_tokens)

def unOverload(overloaded):
    selfie_tokens = []
    tokens = tokenizer(overloaded)
    for token in tokens:
        if token in num_dict.keys():
            token = overloading_dict[num_dict[token]]
        selfie_tokens.append(token)
    return ''.join(selfie_tokens)



s = unHex("[C][C][=C][C][=C][Branch2][20][S][C][=C][N][=C][C][=C][Ring1][4][C][=C][Branch1][4][C][Branch1][0][N][=O][S][Ring1][6][C][=C][Ring2][17]")
print(s)
#dataset = pd.read_csv("data/moses_train_no_overload.txt").to_numpy()
#new_dataset = [convert(selfie[0]) for selfie in dataset]
#df = pd.DataFrame(new_dataset, columns=["selfies"])
#df.to_csv("data/moses_train_no_hex.txt", index=False)


#dataset = pd.read_csv("data/moses_test_no_overload.txt").to_numpy()
#new_dataset = [convert(selfie[0]) for selfie in dataset]
#df = pd.DataFrame(new_dataset, columns=["selfies"])
#df.to_csv("data/moses_test_no_hex.txt", index=False)


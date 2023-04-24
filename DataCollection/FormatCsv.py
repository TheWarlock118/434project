import os
from csv import writer
root_dir = 'DataSet'

# Read the Pokedex text file
Pokedex = {}
Generation_Dict = {}
failed = []

# Generation 1
with open('Pokedexes/Generation1.txt') as f:
    gen1Lines = f.readlines()
    gen1Lines = [s.rstrip('\n') for s in gen1Lines]

# Generation 2
with open('Pokedexes/Generation2.txt') as f:
    gen2Lines = f.readlines()
    gen2Lines = [s.rstrip('\n') for s in gen2Lines]

# Generation 3
with open('Pokedexes/Generation3.txt') as f:
    gen3Lines = f.readlines()
    gen3Lines = [s.rstrip('\n') for s in gen3Lines]

# Generation 4
with open('Pokedexes/Generation4.txt') as f:
    gen4Lines = f.readlines()
    gen4Lines = [s.rstrip('\n') for s in gen4Lines]

# Generation 5
with open('Pokedexes/Generation5.txt') as f:
    gen5Lines = f.readlines()
    gen5Lines = [s.rstrip('\n') for s in gen5Lines]

# Generation 6
with open('Pokedexes/Generation6.txt') as f:
    gen6Lines = f.readlines()
    gen6Lines = [s.rstrip('\n') for s in gen6Lines]

# Generation 7
with open('Pokedexes/Generation7.txt') as f:
    gen7Lines = f.readlines()
    gen7Lines = [s.rstrip('\n') for s in gen7Lines]

# Generation 8
with open('Pokedexes/Generation8.txt') as f:
    gen8Lines = f.readlines()
    gen8Lines = [s.rstrip('\n') for s in gen8Lines]

# Generation 9
with open('Pokedexes/Generation9.txt') as f:
    gen9Lines = f.readlines()
    gen9Lines = [s.rstrip('\n') for s in gen9Lines]

def get_generation(dexNumber):
    if (dexNumber in gen1Lines):
        return 1
    if (dexNumber in gen2Lines):
        return 2
    if (dexNumber in gen3Lines):
        return 3
    if (dexNumber in gen4Lines):
        return 4
    if (dexNumber in gen5Lines):
        return 5
    if (dexNumber in gen6Lines):
        return 6
    if (dexNumber in gen7Lines):
        return 7
    if (dexNumber in gen8Lines):
        return 8
    if (dexNumber in gen9Lines):
        return 9

with open('Pokedexes/Pokedex.txt') as f:
    pokedexLines = f.readlines()
    for line in pokedexLines:
        words = line.split('\t')    
        dexNumber = words[0]
        name = words[1]
        if(name[-1] == '\n'):
            name = name[:-1]
        Pokedex[name] = dexNumber
        Generation_Dict[dexNumber] = get_generation(dexNumber)        

with open('PokemonData.csv', mode='w') as csv:
    w = writer(csv, lineterminator = '\n')

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:        
            try:      
                path = os.path.join(subdir, file)
                path = path.replace('\\', '/')        
                row = [path, Generation_Dict[Pokedex[subdir[8:]]]]                
                w.writerow(row)
            except Exception as e:
                failed.append(subdir[8:])
                print(e)


    csv.close()
    print("Failed:")
    print(failed)


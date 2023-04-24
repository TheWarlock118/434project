import requests
import os

get_official = False
get_5th_gen = True
get_7th_gen_small = True
get_8th_gen_small = True
get_9th_gen_small = True
get_7th_gen_large = True
get_8th_gen_large = True
get_9th_gen_large = True

# Read the Pokedex text file
with open('Pokedexes/Pokedex.txt') as f:
    pokedexLines = f.readlines()



base_url_official_art = "https://assets.pokemon.com/assets/cms2/img/pokedex/full/" # + dexNumber + ".png"
base_url_5th_gen = "https://img.pokemondb.net/sprites/black-white/normal/" # + name.lower() + ".png"
base_url_7th_gen_small = "https://www.serebii.net/pokedex-sm/icon/" # + dexNumber + ".png" 1-809
base_url_8th_gen_small = "https://www.serebii.net/pokedex-swsh/icon/" # + dexNumber + ".png" 810-905
base_url_9th_gen_small = "https://www.serebii.net/pokedex-sv/icon/new/" # + dexNumber + ".png" 906-1010

base_url_7th_gen_large = "https://www.serebii.net/sunmoon/pokemon/" # + dexNumber + ".png" 1-809
base_url_8th_gen_large = "https://www.serebii.net/swordshield/pokemon/" # + dexNumber + ".png" 810-905
base_url_9th_gen_large = "https://www.serebii.net/scarletviolet/pokemon/new/" # + dexNumber + ".png" 906-1010

Pokedex = {}
Generation_Dict = {}

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


for line in pokedexLines:
    words = line.split('\t')
    print(words)
    dexNumber = words[0]
    name = words[1]
    name = name.replace('\n','')

    Pokedex[name] = dexNumber

    path = "DataSetGenerational/" + str(get_generation(dexNumber)) + "/"
    if(not os.path.exists(path)):
        os.makedirs(path)

    if(get_official):
        img_data = requests.get(base_url_official_art + dexNumber + ".png").content
        with open(path + name + "_official.png", 'wb') as handler:
            handler.write(img_data)

    if(get_5th_gen and int(dexNumber) < 650):
        img_data = requests.get(base_url_5th_gen + name.lower() + ".png").content
        with open(path + name + "_5thGenSprite.png", 'wb') as handler:
            handler.write(img_data)
    
    if(get_7th_gen_small and int(dexNumber) < 810):
        img_data = requests.get(base_url_7th_gen_small + dexNumber + ".png").content
        with open(path + name + "_7thGenSmall.png", 'wb') as handler:
            handler.write(img_data)

    if(get_8th_gen_small and int(dexNumber) > 809 and int(dexNumber) < 906):
        img_data = requests.get(base_url_8th_gen_small + dexNumber + ".png").content
        with open(path + name + "_8thGenSmall.png", 'wb') as handler:
            handler.write(img_data)

    if(get_9th_gen_small and int(dexNumber) > 905):
        img_data = requests.get(base_url_9th_gen_small + dexNumber + ".png").content
        with open(path + name + "_9thGenSmall.png", 'wb') as handler:
            handler.write(img_data)

    if(get_7th_gen_large and int(dexNumber) < 810):
        img_data = requests.get(base_url_7th_gen_large + dexNumber + ".png").content
        with open(path + name + "_7thGenlarge.png", 'wb') as handler:
            handler.write(img_data)

    if(get_8th_gen_large and int(dexNumber) > 809 and int(dexNumber) < 906):
        img_data = requests.get(base_url_8th_gen_large + dexNumber + ".png").content
        with open(path + name + "_8thGenlarge.png", 'wb') as handler:
            handler.write(img_data)

    if(get_9th_gen_large and int(dexNumber) > 905):
        img_data = requests.get(base_url_9th_gen_large + dexNumber + ".png").content
        with open(path + name + "_9thGenlarge.png", 'wb') as handler:
            handler.write(img_data)
    

    

import requests
import os

get_official = False
get_5th_gen = False
get_7th_gen_small = False
get_8th_gen_small = False
get_9th_gen_small = False
get_7th_gen_large = False
get_8th_gen_large = False
get_9th_gen_large = False

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

for line in pokedexLines:
    words = line.split('\t')
    print(words)
    dexNumber = words[0]
    name = words[1]
    if(name[-1] == '\n'):
        name = name[:-1]

    path = "DataSet/" + name + "/"
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
    

    

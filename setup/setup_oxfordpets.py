mkdir -p ./data/OxfordPets
cd data/OxfordPets
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz

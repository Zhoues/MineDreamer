# Create the directory structure if it doesn't exist
mkdir -p data/weights/vpt
mkdir -p data/weights/mineclip
mkdir -p data/weights/steve1
mkdir -p data/weights/cvae
mkdir -p data/memory
mkdir -p data/prompt_mapping


# vpt arch file
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model -P data/weights/vpt

# vpt weights file
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-foundation-2x.weights -P data/weights/vpt

# MineCLIP weights
gdown https://drive.google.com/uc?id=1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW -O data/weights/mineclip/attn.pth

# STEVE-1 weights
gdown https://drive.google.com/uc?id=1E3fd_-H1rRZqMkUKHfiMhx-ppLLehQPI -O data/weights/steve1/steve1.weights

# STEVE-1 Prior weights
gdown https://drive.google.com/uc?id=1OdX5wiybK8jALVfP5_dEo0CWm9BQbDES -O data/weights/steve1/steve1_prior.pt

# MineDreamer Prompt Generator weights
gdown https://drive.google.com/uc?id=1fiZbEVndG8Nn-E0CHkV6GPJFxH2j-r17 -O data/weights/cvae.zip
unzip data/weights/cvae.zip -d data/weights
rm data/weights/cvae.zip

# MineDreamer Multi-Modal Memory
gdown https://drive.google.com/uc?id=1MxppdHwCXh3fyKOi4Tjp5j8lFTSXMHVY -O data/memory.zip
unzip data/memory.zip -d data
rm data/memory.zip

# MineDreamer Multi-Modal Memory
gdown https://drive.google.com/uc?id=1qevCKLtv6GMAHefu66k3Nqt3sN9hmOEf -O data/prompt_mapping/prompt_to_key.json
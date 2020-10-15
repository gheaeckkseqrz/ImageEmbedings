# Image	Embeddings

## Setup

Download and extract libtorch	at the root of the repo.
```
wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
```

## Build

The project uses a classic cmake workflow.
```
mkdir build
cd build
cmake ..
make -j
```

## Run

Prepare	your dataset. The expected structure is	one fodler per identity	contatining images (jpg or png).

Run with ```ImageEmbeddings PATH_TO_DATASET_FOLDER```

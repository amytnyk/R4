[![Ubuntu Tests](https://github.com/amytnyk/r4/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/amytnyk/r4/actions/workflows/ubuntu.yml)
[![Arch Tests](https://github.com/amytnyk/r4/actions/workflows/arch.yml/badge.svg)](https://github.com/amytnyk/r4/actions/workflows/arch.yml)

# 4D Raytracing

### Installing dependencies

Ubuntu
```shell
sudo apt-get update
sudo apt-get install build-essential cmake gcc g++ freeglut3-dev libglfw3-dev libglew-dev libfmt-dev
```

Arch
```shell
pacman -Syu
pacman -S base-devel cmake gcc freeglut glfw-x11 glew fmt
```

### Build

Manual
```shell
./build.sh
```

Automatic
```shell
mkdir -p build
cd build
cmake ..
cmake --build .
```

### Usage

Rendering 3D image:
```shell
mkdir output
./bin/ray4 -b24 -a1:1:1 -s_:_:_ -r512:512:512 -iinputs.r4/dots4.r4 -ooutput/dots4.r4img
```

Interactive 3D image slice viewer (use left and right arrow keys to navigate, press *escape* to exit):
```shell
./bin/r4display output/dots4.r4img
```

Program loads all file into a memory. If you want to load on-fly use --lazy option:
```shell
./bin/r4display output/dots4.r4img --lazy
```

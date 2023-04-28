[![Tests](https://github.com/amytnyk/r4/actions/workflows/build.yml/badge.svg)](https://github.com/amytnyk/r4/actions/workflows/build.yml)

# 4D Raytracing

### Compilation

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

Interactive 3D image slice viewer:
```shell
./bin/r4display output/dots4.r4img
```

Program loads all file into a memory. If you want to load on-fly use --lazy option:
```shell
./bin/r4display output/dots4.r4img --lazy
```

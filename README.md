# Using MLPACK/armadillo with dr_wav.h

This is an example that uses a C++ library (MLPACK/armadillo) to solve a linear system and another library (dr_wav.h) to create a .wav file. These files are given:

    CMakeLists.txt . . . . CMake build system configuration
    dr_wav.h . . . . . . . Header-only audio library
    main.cpp . . . . . . . Example source code

To build this system, first install MLPACK and Armadillo:

    brew install mlpack armadillo

If you do not have Homebrew yet, install XCode using the Apple App store and then go here: https://brew.sh

Once these components are installed, make a build folder, run cmake, then build and run the example:

    cd path/to/this/repo
    mkdir build
    cd build
    cmake ..
    cmake --build .
    ./main

For information on how to use MLPACK/Armadillo, consult the documentation:

    <https://arma.sourceforge.net/docs.html#solve>

See also: <https://www.mlpack.org>




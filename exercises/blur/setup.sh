#!/bin/bash

#build tools
make -C tools/ clean
make -C tools/

#build exes
make -C hybrid/ clean
make -C hybrid/ CCFLAGS="-DNDEBUG -O3 -march=native"
mv hybrid/blur_hybrid.x .
mv hybrid/blur_hybrid_wcomm.x .

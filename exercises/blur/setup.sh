#!/bin/bash

#build tools
make -C tools/ clean
make -C tools/

#build exes
make -C hybrid/ clean
make -C hybrid/ CCFLAGS="-O3 -march=native -DNDEBUG -DKAHAN_OFF"
#-DBLOCKING_ON 
#-DNDEBUG 

mv hybrid/blur_hybrid.x .
mv hybrid/blur_hybrid_wcomm.x .

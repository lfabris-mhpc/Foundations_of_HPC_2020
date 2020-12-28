#!/bin/bash

#build tools
make -C tools/ clean
make -C tools/

comment="
#generate gradient images
tools/make_gradient.x 16 16 65535
mv gradient.pgm images/gradient_small.pgm
tools/make_gradient.x 512 512 65535
mv gradient.pgm images/gradient.pgm
tools/make_gradient.x 2048 2048 65535
mv gradient.pgm images/gradient_big.pgm

rm images/gradient_big_*.pgm
rm images/check_me_*.pgm

for img in images/*.pgm
do
	tools/img_normalize.x ${img}
	echo
done

i=1
j=1
cp -T images/gradient_big.pgm images/gradient_big_1.pgm
cp -T images/check_me.pgm images/check_me_1.pgm
for e in {1..7}
do
	if ((i > j))
	then
		((j *= 2))
	else
		((i *= 2))
	fi
	((ij = i* j))
	tools/img_tile.x images/gradient_big.pgm images/gradient_big_${ij}.pgm ${i} ${j}
	tools/img_tile.x images/check_me.pgm images/check_me_${ij}.pgm ${i} ${j}
done
"

#build exes
make -C hybrid/ clean
make -C hybrid/ CCFLAGS=-DNDEBUG
mv hybrid/blur_hybrid.x .
mv hybrid/blur_hybrid_wcomm.x .

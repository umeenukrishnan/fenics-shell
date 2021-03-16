#!/bin/bash
docker run --rm -ti -v "$(pwd)":/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:2019.1.0.r2 'export PATH=/home/fenics/shared/utils/pylit:$PATH; /bin/bash -i'

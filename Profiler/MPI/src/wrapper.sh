#!/bin/bash

#if [ $(($OMPI_COMM_WORLD_RANK%3)) == 0 ] ; then
#    cp ../counter.txt /tmp/counter${OMPI_COMM_WORLD_RANK}.txt
#    rocprof -i /tmp/counter${OMPI_COMM_WORLD_RANK}.txt ../cluster ../testData.fasta
#    rm -rf  /tmp/counter${OMPI_COMM_WORLD_RANK}.txt
#else
#    ../cluster ../testData.fasta
#fi

cp ../counter.txt /tmp/counter${OMPI_COMM_WORLD_RANK}.txt
rocprof -i /tmp/counter${OMPI_COMM_WORLD_RANK}.txt ../cluster ../testData.fasta
rm -rf  /tmp/counter${OMPI_COMM_WORLD_RANK}.txt
#!/bin/bash
export LD_PRELOAD=./pmpi.so
mpirun -n 2 ./a.out

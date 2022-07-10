mpi_datatype = {r'0x2b5319ddb840' : ['MPI_CHAR',            1],\
                r'0x2b5319ddb640' : ['MPI_SIGNED_CHAR',     1],\
                r'0x2b5319ddb440' : ['MPI_UNSIGNED_CHAR',   1],\
                r'0x2b5319ddb240' : ['MPI_BYTE',            1],\
                r'0x2b5319dd9a40' : ['MPI_WCHAR',           2],\
                r'0x2b5319ddb040' : ['MPI_SHORT',           2],\
                r'0x2b5319ddae40' : ['MPI_UNSIGNED_SHORT',  2],\
                r'0x6261c0'       : ['MPI_INT',             4],\
                r'0x2b5319ddaa40' : ['MPI_UNSIGNED',        4],\
                r'0x2b5319dda840' : ['MPI_LONG',            4],\
                r'0x2b5319dda640' : ['MPI_UNSIGNED_LONG',   4],\
                r'0x2b5319dda040' : ['MPI_FLOAT',           4],\
                r'0x2b5319dd9e40' : ['MPI_DOUBLE',          8],\
                r'0x2b5319dd9c40' : ['MPI_LONG_DOUBLE',    16],\
                r'0x2b5319dda440' : ['MPI_LONG_LONG_INT',   8],\
                r'0x2b5319dda240' : ['MPI_UNSIGNED_LONG_LONG',8]\
               } 

mpi_func = {}
mpi_func['0'] = 'MPI_Init'
mpi_func['1'] = 'MPI_Init_thread'
mpi_func['2'] = 'MPI_Finalize'
mpi_func['3'] = 'MPI_Ibsend'
mpi_func['4'] = 'MPI_Irsend'
mpi_func['5'] = 'MPI_Isend'
mpi_func['6'] = 'MPI_Issend'
mpi_func['7'] = 'MPI_Bsend'
mpi_func['8'] = 'MPI_Rsend'
mpi_func['9'] = 'MPI_Send'
mpi_func['10'] = 'MPI_Ssend'
mpi_func['11'] = 'MPI_Sendrecv'
mpi_func['12'] = 'MPI_Sendrecv_replace'
mpi_func['13'] = 'MPI_Mrecv'
mpi_func['14'] = 'MPI_Recv'
mpi_func['15'] = 'MPI_Irecv'
mpi_func['16'] = 'MPI_Imecv'
mpi_func['17'] = 'MPI_Bcast'
mpi_func['18'] = 'MPI_Ibcast'
mpi_func['19'] = 'MPI_Scatter'
mpi_func['20'] = 'MPI_Scatterv'
mpi_func['21'] = 'MPI_Iscatter'
mpi_func['22'] = 'MPI_Iscatterv'
mpi_func['23'] = 'MPI_Gather'
mpi_func['24'] = 'MPI_Gatherv'
mpi_func['25'] = 'MPI_Igather'
mpi_func['26'] = 'MPI_Igatherv'
mpi_func['27'] = 'MPI_Allgather'
mpi_func['28'] = 'MPI_Allgatherv'
mpi_func['29'] = 'MPI_Iallgather'
mpi_func['30'] = 'MPI_Iallgatherv'
mpi_func['31'] = 'MPI_Neighbor_allgather'
mpi_func['32'] = 'MPI_Neighbor_allgatherv'
mpi_func['33'] = 'MPI_Ineighbor_allgather'
mpi_func['34'] = 'MPI_Ineighbor_allgatherv'


编号 函数名 参数
0  MPI_Init                 编号
1  MPI_Init_thread          编号
2  MPI_Finalize             编号
3  MPI_Ibsend               编号 数据量 数据类型 目的地址
4  MPI_Irsend               编号 数据量 数据类型 目的地址
5  MPI_Isend                编号 数据量 数据类型 目的地址
6  MPI_Issend               编号 数据量 数据类型 目的地址
7  MPI_Bsend                编号 数据量 数据类型 目的地址 耗时
8  MPI_Rsend                编号 数据量 数据类型 目的地址 耗时
9  MPI_Send                 编号 数据量 数据类型 目的地址 耗时
10 MPI_Ssend                编号 数据量 数据类型 目的地址 耗时
11 MPI_Sendrecv             编号 发送量 发送类型 目的地址 接收量 接收类型 源地址 耗时
12 MPI_Sendrecv_replace     编号 数据量 数据类型 目的地址 源地址 耗时
13 MPI_Mrecv                编号 数据量 数据类型 耗时
14 MPI_Recv                 编号 数据量 数据类型 源地址 耗时
15 MPI_Irecv                编号 数据量 数据类型 源地址
16 MPI_Imecv                编号 数据量 数据类型
17 MPI_Bcast                编号 数据量 数据类型 根节点 耗时
18 MPI_Ibcast               编号 数据量 数据类型 根节点 耗时
19 MPI_Scatter              编号 数据量 数据类型 根节点 耗时
20 MPI_Scatterv             编号 数据量 数据类型 根节点 耗时
21 MPI_Iscatter             编号 数据量 数据类型 根节点
22 MPI_Iscatterv            编号 数据量 数据类型 根节点
23 MPI_Gather               编号 数据量 数据类型 根节点 耗时
24 MPI_Gatherv              编号 数据量 数据类型 根节点 耗时
25 MPI_Igather              编号 数据量 数据类型 根节点
26 MPI_Igatherv             编号 数据量 数据类型 根节点
27 MPI_Allgather            编号 数据量 数据类型 耗时
28 MPI_Allgatherv           编号 数据量 数据类型 耗时
29 MPI_Iallgather           编号 数据量 数据类型
30 MPI_Iallgatherv          编号 数据量 数据类型
31 MPI_Ineighbor_allgather  编号 发送量 发送类型 接收量 接收类型
32 MPI_Ineighbor_allgatherv 编号 发送量 发送类型 接收量 接收类型
33 MPI_Neighbor_allgather   编号 发送量 发送类型 接收量 接收类型 耗时
34 MPI_Neighbor_allgatherv  编号 发送量 发送类型 接收量 接收类型 耗时
MPI通讯函数分为阻塞与非阻塞，阻塞函数有执行时间，非阻塞函数没有执行时间。
阻塞函数：
7  MPI_Bsend
8  MPI_Rsend
9  MPI_Send
10 MPI_Ssend
11 MPI_Sendrecv
12 MPI_Sendrecv_replace
13 MPI_Mrecv
14 MPI_Recv
17 MPI_Bcast
19 MPI_Scatter
20 MPI_Scatterv
23 MPI_Gather
24 MPI_Gather
27 MPI_Allgather
28 MPI_Allgatherv
33 MPI_Neighbor_allgather
34 MPI_Neighbor_allgather
非阻塞函数：
3  MPI_Ibsend
4  MPI_Irsend
5  MPI_Isend
6  MPI_Issend
15 MPI_Irecv
16 MPI_Imecv
18 MPI_Ibcast
21 MPI_Iscatter
22 MPI_Iscatterv
25 MPI_Igather
26 MPI_Igatherv
29 MPI_Iallgather
30 MPI_Iallgatherv
31 MPI_Ineighbor_allgather
32 MPI_Ineighbor_allgatherv
MPI数据类型 代码
MPI_CHAR               -380835776
MPI_SIGNED_CHAR        -380836288
MPI_UNSIGNED_CHAR      -380836800
MPI_BYTE               -380837312
MPI_WCHAR              -380843456
MPI_SHORT              -380837824
MPI_UNSIGNED_SHORT     -380838336
MPI_INT                6423552
MPI_UNSIGNED           -380839360
MPI_LONG               -380839872
MPI_UNSIGNED_LONG      -380840384
MPI_FLOAT              -380841920
MPI_DOUBLE             -380842432
MPI_LONG_DOUBLE        -380842944
MPI_LONG_LONG_INT      -380840896
MPI_UNSIGNED_LONG_LONG -380841408

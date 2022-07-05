# SIAT-Profiler

## 功能介绍

* SIAT-Profiler是一个用于收集在国产DCU加速卡上运行的程序和MPI多节点程序的性能数据，并对所采集的性能数据进行分析，通过这种方式来获取程序在DCU上运行时的潜在性能瓶颈及程序在使用MPI进行进程间通信时所存在的潜在性能瓶颈。
* 主要的功能分成两部分：DCU和MPI。

### DCU端

DCU端主要通过AMD [ROC-profiler](https://github.com/ROCm-Developer-Tools/rocprofiler)工具来收集程序运行时DCU中的硬件计数器信息，并对收集到的硬件计数器信息进行分析、可视化，从而得到程序在DCU运行时潜在的性能瓶颈。主要的分析流程如下所示：

![image-20220705014202579](https://raw.githubusercontent.com/zhuangbility111/typora_img/main/img/image-20220705014202579.png)

#### 1. 识别热点函数

通过ROC-profiler得到的硬件计数器信息，本工具可以识别到程序在DCU上执行的所有核函数中，哪些核函数是热点函数。例如如下图是本工具通过分析得到的分子动力学软件Gromacs在DCU上运行的热点核函数图：

![image-20220705014718921](https://raw.githubusercontent.com/zhuangbility111/typora_img/main/img/image-20220705014718921.png)

#### 2. 代表性数据分析

因为得到的DCU硬件计数器的数据量较大，很难直接对所有硬件计数器数据进行分析，本工具提供了一个数据相似性分析的功能，通过比较同一个节点上不同DCU所产生的硬件计数器数据以及比较不同节点上的不同DCU所产生的硬件计数器数据的相似度，确定能否使用其中一个节点上的一个DCU所产生的硬件计数器数据作为该程序在DCU上运行时的代表性硬件计数器数据。本工具通过这种方式，可以大幅度减少后续数据处理及数据分析的开销。

![image-20220705015259988](https://raw.githubusercontent.com/zhuangbility111/typora_img/main/img/image-20220705015259988.png)

* 节点内皮尔逊系数

![](https://raw.githubusercontent.com/zhuangbility111/typora_img/main/img/image-20220705015841725.png)

* 节点间皮尔逊系数

![image-20220705015940286](https://raw.githubusercontent.com/zhuangbility111/typora_img/main/img/image-20220705015940286.png)

#### 3. 性能雷达图分析

雷达图分析采用五个维度来刻画热点核函数的性能特征，分别是：**计算能力、显存读、显存写、显存带宽占用率、缓存命中率**。借助性能雷达图分析，本工具可以得到程序在DCU上执行的核函数的不同性能特征，从而识别出潜在的性能瓶颈，例如，某核函数的计算能力较弱，说明该核函数未能充分利用DCU的计算能力，需要考虑在该核函数中的计算部分进行优化，提高核函数利用DCU计算的能力。

![image-20220705015744653](https://raw.githubusercontent.com/zhuangbility111/typora_img/main/img/image-20220705015744653.png)

### MPI端

MPI端主要采用动态库劫持`LD_PRELOAD`的方式，利用`PMPI.so`动态库劫持对MPI函数的调用，从而收集到程序中MPI函数调用的信息，进而分析出程序使用MPI在跨节点通信上的通讯模式，以及可能存在的通信性能瓶颈，并根据性能瓶颈给出一定的优化建议。本工具的MPI性能分析流程如下图所示：

![image-20220705020407740](https://raw.githubusercontent.com/zhuangbility111/typora_img/main/img/image-20220705020407740.png)

本工具所支持的MPI函数有：

**初始化类函数**

* MPI_Init、MPI_Init_thread、MPI_Finalize

**点对点发送、接收类函数**

* MPI_Ibsend、MPI_Irsend、MPI_Isend、MPI_Bsend、MPI_Rsend、MPI_Send、MPI_Ssend、MPI_Sendrecv、 MPI_Sendrecv_replace、MPI_Mrecv、MPI_Recv、MPI_Irecv、MPI_Imecv

**集合通信类函数**

* MPI_Bcast、 MPI_Ibcast、MPI_Gatherv、MPI_Iallgatherv、MPI_Scatter、MPI_Igather、MPI_Ineighbor_allgather、MPI_Scatterv、MPI_Igatherv、MPI_Ineighbor_allgatherv、MPI_Iscatter、MPI_Allgather、MPI_Neighbor_allgather、MPI_Iscatterv、MPI_Allgatherv、MPI_Neighbor_allgatherv、MPI_Gather、MPI_Iallgather

本工具所支持的MPI通讯特征分析有：

1. **MPI发送消息大小及消息数量统计图：**衡量发送的MPI消息尺寸和数量的特征
2. **进程调用函数数量统计图：**统计各进程调用函数类型及其调用次数，衡量MPI消息传递特性(比如单点通讯为主还是集合通讯为主)
3. **进程函数调用时间统计图：**统计阻塞函数消耗时间，可以衡量MPI的热点函数
4. **各进程间数据传输分布图：**用于统计各进程间传输的数据量，可以衡量MPI数据传输在网络当中的均匀特性

## 使用方式

### 1. 采集性能数据

#### 1.1 单节点程序

如果程序运行在单节点上，意味着程序没有使用MPI，只需要使用`rocprof`命令来采集程序运行时在DCU上的硬件计数器。具体`rocprof`命令的简单使用示例如下：

```shell
rocprof -i ${counter_path}/counter.txt -d ./result  <运行待测程序的命令>
```

其中参数解释如下：

* -i : 包含所需要采集硬件计数器的文本文件，将`${counter_path}`替换成`counter.txt`所在的目录，`counter.txt`在目录`./Profiler/DCU/src/` 下
* -d: 指定采集的硬件计数器数据所存放的目录，可以自行替换

`rocprof`的更多参数使用及解释可以参照https://github.com/ROCm-Developer-Tools/rocprofiler

#### 1.2 多节点程序

如果程序运行在多节点上，则需要采集DCU和MPI上的性能数据。具体步骤如下：

1. 编写一个名为`wrapper.sh`的shell脚本，内容如下：

```shell
cp ${counter_path}/counter.txt /tmp/counter${OMPI_COMM_WORLD_RANK}.txt
rocprof -i /tmp/counter${OMPI_COMM_WORLD_RANK}.txt -d ./result <运行待测程序的命令>
rm -rf  /tmp/counter${OMPI_COMM_WORLD_RANK}.txt
```

其中

* 将`${counter_path}`替换成`counter.txt`所在的目录，`counter.txt`已经放在了目录`./Profiler/DCU/src/` 下
* 将`<运行待测程序的命令>`替换成自己执行程序时所输入的命令及参数
* -d: 指定采集的硬件计数器数据所存放的目录，可以自行替换

2. 使用`mpicxx`编译文件`pmpi.cpp`，从而得到本工具需要的动态库文件`pmpi.so`：
```shell
mpicxx pmpi.cpp -fPIC -shared -o pmpi.so
```

3. 使用`LD_PRELOAD`劫持动态库，并使用`mpirun`来执行`warpper.sh`脚本：

```shell
export LD_PRELOAD=${pmpi_so_path}/pmpi.so
mpirun <mpi 运行参数> ./wrapper.sh
```

​	其中

* 将`${pmpi_so_path}`替换成上一步编译得到的动态库文件`pmpi.so`所在的目录。

* 将`<mpi 运行参数>`替换成mpi执行时的运行参数，比如`-n 4`指定MPI进程数为4。

### 2. DCU端性能数据分析

1. 进入到`./Profiler/DCU/src`目录下
2. 使用如下命令对DCU性能数据的相似度进行分析

```shell
python DCU_similarity_analysis.py -N ${node_num} -n ${dcu_num_per_node} -i ${input_dcu_result_path} -t ${similarity_threshold}
```

参数解释说明如下：

* -N: 程序所使用的节点数量
* -n: 每个节点所使用的DCU数量
* -i: 保存采集到的DCU性能数据的目录路径${input_mpi_result_path}
* -t: 相似度阈值。如果两个数据的相似度超过这个阈值，则可以认为这两个数据高度相似，取其中一个数据作为输入即可。
* -h: 查看参数帮助信息

3. 根据相似度分析结果的提示，进行进一步的DCU性能特征分析。如果各DCU产生的性能数据相似度较高，则任取一性能数据文件作为输入，使用如下命令对DCU性能特征进行分析；如果各DCU产生的性能数据相似度较低，则应该取不相似的性能数据文件作为输入，使用如下命令对DCU性能特征进行分析。

```shell
python DCU_performance_analysis.py  -i ${counter_file} -o ${output_path}
```

参数解释说明如下：

* -i: 输入的性能数据文件名及路径

* -o: 保存输出分析结果图像的路径${output_path}
* -h: 查看参数帮助信息

4. 使用`scp`命令或者其他文件传输方式，将${output_path}中所保存的输出分析结果图像传送至支持图形化界面的系统上，然后直接打开图像即可。输出图像的文件名及所包含内容如下所示：

```shell
2022-07-01-23-24-16_counter10_hotbar.png: 热点核函数图
2022-07-01-23-24-16_dynamicProgramming(char**, int*, int, int, int*)_radarmap.png: 具体核函数对应的雷达性能特征图
```

### 3. MPI端性能数据分析

1. 进入到`./Profiler/MPI/src`目录下
2. 使用如下命令对MPI性能数据进行分析

```shell
python result_analysis.py  -i ${input_mpi_result_path} -o ${output_path}
```

参数解释说明如下：

* -i: 保存采集到的MPI性能数据的目录路径${input_mpi_result_path}

* -o: 保存输出分析结果图像的路径${output_path}
* -h: 查看参数帮助信息

3. 使用`scp`命令或者其他文件传输方式，将${output_path}中所保存的输出分析结果图像传送至支持图形化界面的系统上，然后直接打开图像即可。输出图像的文件名及所包含内容如下所示：

```shel
2022_07_02_22_40_40_count-message.png: 信息大小及数量统计图
2022_07_02_22_40_41_elapsed time of MPI function.png: MPI函数耗时统计图
2022_07_02_22_40_47_calling times of MPI function.png: MPI函数调用次数统计图
2022_07_02_22_40_54_communication_pattern.png: MPI进程间传输模式图
```


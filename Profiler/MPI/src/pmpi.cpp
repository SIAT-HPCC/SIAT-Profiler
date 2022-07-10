// pmpi.cpp 劫持mpi动态链接库，输出profile信息。
// 只支持Open MPI4.0，文档参考：https://www-lb.open-mpi.org/doc/v4.0/
// Open MPI4.0.4版，共440个API函数，这里只劫持其中关于初始化/结束/通讯的常用函数共35个。列表如下：
// 编号 函数名 参数
// 0  MPI_Init                 编号
// 1  MPI_Init_thread          编号
// 2  MPI_Finalize             编号
// 3  MPI_Ibsend               编号 数据量 数据类型 目的地址
// 4  MPI_Irsend               编号 数据量 数据类型 目的地址
// 5  MPI_Isend                编号 数据量 数据类型 目的地址
// 6  MPI_Issend               编号 数据量 数据类型 目的地址
// 7  MPI_Bsend                编号 数据量 数据类型 目的地址 耗时
// 8  MPI_Rsend                编号 数据量 数据类型 目的地址 耗时
// 9  MPI_Send                 编号 数据量 数据类型 目的地址 耗时
// 10 MPI_Ssend                编号 数据量 数据类型 目的地址 耗时
// 11 MPI_Sendrecv             编号 发送量 发送类型 目的地址 接收量 接收类型 源地址 耗时
// 12 MPI_Sendrecv_replace     编号 数据量 数据类型 目的地址 源地址 耗时
// 13 MPI_Mrecv                编号 数据量 数据类型 耗时
// 14 MPI_Recv                 编号 数据量 数据类型 源地址 耗时
// 15 MPI_Irecv                编号 数据量 数据类型 源地址
// 16 MPI_Imecv                编号 数据量 数据类型
// 17 MPI_Bcast                编号 数据量 数据类型 根节点 耗时
// 18 MPI_Ibcast               编号 数据量 数据类型 根节点 耗时
// 19 MPI_Scatter              编号 数据量 数据类型 根节点 耗时
// 20 MPI_Scatterv             编号 数据量 数据类型 根节点 耗时
// 21 MPI_Iscatter             编号 数据量 数据类型 根节点
// 22 MPI_Iscatterv            编号 数据量 数据类型 根节点
// 23 MPI_Gather               编号 数据量 数据类型 根节点 耗时
// 24 MPI_Gatherv              编号 数据量 数据类型 根节点 耗时
// 25 MPI_Igather              编号 数据量 数据类型 根节点
// 26 MPI_Igatherv             编号 数据量 数据类型 根节点
// 27 MPI_Allgather            编号 数据量 数据类型 耗时
// 28 MPI_Allgatherv           编号 数据量 数据类型 耗时
// 29 MPI_Iallgather           编号 数据量 数据类型
// 30 MPI_Iallgatherv          编号 数据量 数据类型
// 31 MPI_Neighbor_allgather   编号 发送量 发送类型 接收量 接收类型 耗时
// 32 MPI_Neighbor_allgatherv  编号 发送量 发送类型 接收量 接收类型 耗时
// 33 MPI_Ineighbor_allgather  编号 发送量 发送类型 接收量 接收类型
// 34 MPI_Ineighbor_allgatherv 编号 发送量 发送类型 接收量 接收类型
// MPI通讯函数分为阻塞与非阻塞，阻塞函数有执行时间，非阻塞函数没有执行时间。
// 阻塞函数：
// 7  MPI_Bsend
// 8  MPI_Rsend
// 9  MPI_Send
// 10 MPI_Ssend
// 11 MPI_Sendrecv
// 12 MPI_Sendrecv_replace
// 13 MPI_Mrecv
// 14 MPI_Recv
// 17 MPI_Bcast
// 19 MPI_Scatter
// 20 MPI_Scatterv
// 23 MPI_Gather
// 24 MPI_Gather
// 27 MPI_Allgather
// 28 MPI_Allgatherv
// 31 MPI_Neighbor_allgather
// 32 MPI_Neighbor_allgather
// 非阻塞函数：
// 3  MPI_Ibsend
// 4  MPI_Irsend
// 5  MPI_Isend
// 6  MPI_Issend
// 15 MPI_Irecv
// 16 MPI_Imecv
// 18 MPI_Ibcast
// 21 MPI_Iscatter
// 22 MPI_Iscatterv
// 25 MPI_Igather
// 26 MPI_Igatherv
// 29 MPI_Iallgather
// 30 MPI_Iallgatherv
// 33 MPI_Ineighbor_allgather
// 34 MPI_Ineighbor_allgatherv
// 编译方法：
// mpicxx pmpi.cpp -fPIC -shared -o pmpi.so
// 使用方法是写一个脚本job.sh，在脚本中添加LD_PRELOAD环境变量
// cat job.sh
// #!/bin/bash
// export LD_PRELOAD=./pmpilib.so
// a.out

#include <fstream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#include <dirent.h>
#include <vector>
#include <sys/stat.h>

using namespace std;

int mpiRank, mpiErr;
struct timeval timeStart, timeEnd;  // 时间戳
ofstream logFile;  // 日志文件
string moveFileCmd;  // 从tmp拷贝日志回本地目录的命令
int noRoot=-1, noSize=0;
float noTime=0.0;

void sayReadme() {  // 打印readme
    ofstream readmeFile;
    readmeFile.open("pmpiReadme");
    readmeFile<<"编号 函数名 参数"<<endl;
    readmeFile<<"0  MPI_Init                 编号"<<endl;
    readmeFile<<"1  MPI_Init_thread          编号"<<endl;
    readmeFile<<"2  MPI_Finalize             编号"<<endl;
    readmeFile<<"3  MPI_Ibsend               编号 数据量 数据类型 目的地址"<<endl;
    readmeFile<<"4  MPI_Irsend               编号 数据量 数据类型 目的地址"<<endl;
    readmeFile<<"5  MPI_Isend                编号 数据量 数据类型 目的地址"<<endl;
    readmeFile<<"6  MPI_Issend               编号 数据量 数据类型 目的地址"<<endl;
    readmeFile<<"7  MPI_Bsend                编号 数据量 数据类型 目的地址 耗时"<<endl;
    readmeFile<<"8  MPI_Rsend                编号 数据量 数据类型 目的地址 耗时"<<endl;
    readmeFile<<"9  MPI_Send                 编号 数据量 数据类型 目的地址 耗时"<<endl;
    readmeFile<<"10 MPI_Ssend                编号 数据量 数据类型 目的地址 耗时"<<endl;
    readmeFile<<"11 MPI_Sendrecv             编号 发送量 发送类型 目的地址 接收量 接收类型 源地址 耗时"<<endl;
    readmeFile<<"12 MPI_Sendrecv_replace     编号 数据量 数据类型 目的地址 源地址 耗时"<<endl;
    readmeFile<<"13 MPI_Mrecv                编号 数据量 数据类型 耗时"<<endl;
    readmeFile<<"14 MPI_Recv                 编号 数据量 数据类型 源地址 耗时"<<endl;
    readmeFile<<"15 MPI_Irecv                编号 数据量 数据类型 源地址"<<endl;
    readmeFile<<"16 MPI_Imecv                编号 数据量 数据类型"<<endl;
    readmeFile<<"17 MPI_Bcast                编号 数据量 数据类型 根节点 耗时"<<endl;
    readmeFile<<"18 MPI_Ibcast               编号 数据量 数据类型 根节点 耗时"<<endl;
    readmeFile<<"19 MPI_Scatter              编号 数据量 数据类型 根节点 耗时"<<endl;
    readmeFile<<"20 MPI_Scatterv             编号 数据量 数据类型 根节点 耗时"<<endl;
    readmeFile<<"21 MPI_Iscatter             编号 数据量 数据类型 根节点"<<endl;
    readmeFile<<"22 MPI_Iscatterv            编号 数据量 数据类型 根节点"<<endl;
    readmeFile<<"23 MPI_Gather               编号 数据量 数据类型 根节点 耗时"<<endl;
    readmeFile<<"24 MPI_Gatherv              编号 数据量 数据类型 根节点 耗时"<<endl;
    readmeFile<<"25 MPI_Igather              编号 数据量 数据类型 根节点"<<endl;
    readmeFile<<"26 MPI_Igatherv             编号 数据量 数据类型 根节点"<<endl;
    readmeFile<<"27 MPI_Allgather            编号 数据量 数据类型 耗时"<<endl;
    readmeFile<<"28 MPI_Allgatherv           编号 数据量 数据类型 耗时"<<endl;
    readmeFile<<"29 MPI_Iallgather           编号 数据量 数据类型"<<endl;
    readmeFile<<"30 MPI_Iallgatherv          编号 数据量 数据类型"<<endl;
    readmeFile<<"31 MPI_Neighbor_allgather   编号 发送量 发送类型 接收量 接收类型"<<endl;
    readmeFile<<"32 MPI_Neighbor_allgatherv  编号 发送量 发送类型 接收量 接收类型"<<endl;
    readmeFile<<"33 MPI_Ineighbor_allgather  编号 发送量 发送类型 接收量 接收类型 耗时"<<endl;
    readmeFile<<"34 MPI_Ineighbor_allgatherv 编号 发送量 发送类型 接收量 接收类型 耗时"<<endl;
    readmeFile<<"MPI通讯函数分为阻塞与非阻塞，阻塞函数有执行时间，非阻塞函数没有执行时间。"<<endl;
    readmeFile<<"阻塞函数："<<endl;
    readmeFile<<"7  MPI_Bsend"<<endl;
    readmeFile<<"8  MPI_Rsend"<<endl;
    readmeFile<<"9  MPI_Send"<<endl;
    readmeFile<<"10 MPI_Ssend"<<endl;
    readmeFile<<"11 MPI_Sendrecv"<<endl;
    readmeFile<<"12 MPI_Sendrecv_replace"<<endl;
    readmeFile<<"13 MPI_Mrecv"<<endl;
    readmeFile<<"14 MPI_Recv"<<endl;
    readmeFile<<"17 MPI_Bcast"<<endl;
    readmeFile<<"19 MPI_Scatter"<<endl;
    readmeFile<<"20 MPI_Scatterv"<<endl;
    readmeFile<<"23 MPI_Gather"<<endl;
    readmeFile<<"24 MPI_Gather"<<endl;
    readmeFile<<"27 MPI_Allgather"<<endl;
    readmeFile<<"28 MPI_Allgatherv"<<endl;
    readmeFile<<"31 MPI_Neighbor_allgather"<<endl;
    readmeFile<<"32 MPI_Neighbor_allgather"<<endl;
    readmeFile<<"非阻塞函数："<<endl;
    readmeFile<<"3  MPI_Ibsend"<<endl;
    readmeFile<<"4  MPI_Irsend"<<endl;
    readmeFile<<"5  MPI_Isend"<<endl;
    readmeFile<<"6  MPI_Issend"<<endl;
    readmeFile<<"15 MPI_Irecv"<<endl;
    readmeFile<<"16 MPI_Imecv"<<endl;
    readmeFile<<"18 MPI_Ibcast"<<endl;
    readmeFile<<"21 MPI_Iscatter"<<endl;
    readmeFile<<"22 MPI_Iscatterv"<<endl;
    readmeFile<<"25 MPI_Igather"<<endl;
    readmeFile<<"26 MPI_Igatherv"<<endl;
    readmeFile<<"29 MPI_Iallgather"<<endl;
    readmeFile<<"30 MPI_Iallgatherv"<<endl;
    readmeFile<<"33 MPI_Ineighbor_allgather"<<endl;
    readmeFile<<"34 MPI_Ineighbor_allgatherv"<<endl;
    readmeFile<<"MPI数据类型 代码"<<endl;
    MPI_Datatype type;
    //MPI_CHAR
    type = MPI_CHAR;
    readmeFile<<"MPI_CHAR               "<<type<<endl;
    //MPI_SIGNED_CHAR
    type = MPI_SIGNED_CHAR;
    readmeFile<<"MPI_SIGNED_CHAR        "<<type<<endl;
    //MPI_UNSIGNED_CHAR
    type = MPI_UNSIGNED_CHAR;
    readmeFile<<"MPI_UNSIGNED_CHAR      "<<type<<endl;
    //MPI_BYTE
    type = MPI_BYTE;
    readmeFile<<"MPI_BYTE               "<<type<<endl;
    //MPI_WCHAR
    type = MPI_WCHAR;
    readmeFile<<"MPI_WCHAR              "<<type<<endl;
    //MPI_SHORT
    type = MPI_SHORT;
    readmeFile<<"MPI_SHORT              "<<type<<endl;
    //MPI_UNSIGNED_SHORT
    type = MPI_UNSIGNED_SHORT;
    readmeFile<<"MPI_UNSIGNED_SHORT     "<<type<<endl;
    //MPI_INT
    type = MPI_INT;
    readmeFile<<"MPI_INT                "<<type<<endl;
    //MPI_UNSIGNED
    type = MPI_UNSIGNED;
    readmeFile<<"MPI_UNSIGNED           "<<type<<endl;
    //MPI_LONG
    type = MPI_LONG;
    readmeFile<<"MPI_LONG               "<<type<<endl;
    //MPI_UNSIGNED_LONG
    type = MPI_UNSIGNED_LONG;
    readmeFile<<"MPI_UNSIGNED_LONG      "<<type<<endl;
    //MPI_FLOAT
    type = MPI_FLOAT;
    readmeFile<<"MPI_FLOAT              "<<type<<endl;
    //MPI_DOUBLE
    type = MPI_DOUBLE;
    readmeFile<<"MPI_DOUBLE             "<<type<<endl;
    //MPI_LONG_DOUBLE
    type = MPI_LONG_DOUBLE;
    readmeFile<<"MPI_LONG_DOUBLE        "<<type<<endl;
    //MPI_LONG_LONG_INT
    type = MPI_LONG_LONG_INT;
    readmeFile<<"MPI_LONG_LONG_INT      "<<type<<endl;
    //MPI_UNSIGNED_LONG_LONG
    type = MPI_UNSIGNED_LONG_LONG;
    readmeFile<<"MPI_UNSIGNED_LONG_LONG "<<type<<endl;
}

vector<int> translate_all(MPI_Comm comm){
    int size;
    MPI_Group local_group, comm_group;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_group(comm, &local_group);
    MPI_Comm_group(MPI_COMM_WORLD, &comm_group);

    int *tmp = nullptr, *idx = nullptr;
    
    tmp = (int*)malloc(sizeof(int) * size);
    idx = (int*)malloc(sizeof(int) * size);

    if(tmp == nullptr || idx == nullptr){
        printf("tmp of idx malloc failed\n");
        exit(-1);
    }   

    for(int i = 0; i < size; i++) tmp[i] = i;

    MPI_Group_translate_ranks(local_group, size, tmp, comm_group, idx); // translate rank
    // after translate, idx[local_rank] = currrent process's rank in MPI_COMM_WORLD  

    vector<int> ret(idx, idx + size);
    free(idx);
    free(tmp);

    return ret;
}

int translate_one(MPI_Comm comm, int rank){
    MPI_Group local_group, comm_group;

    MPI_Comm_group(comm, &local_group);
    MPI_Comm_group(MPI_COMM_WORLD, &comm_group);

    int ret;
    MPI_Group_translate_ranks(local_group, 1, &rank, comm_group, &ret); 
    return ret;
}

void init() {  // 打开日志文件
    char hostName[256];
    gethostname(hostName, sizeof(hostName));
    PMPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    string fileName = hostName;
    string folderName = "./mpi_result";
    const char* folderName_cstr = folderName.c_str();
    if (opendir(folderName_cstr) == nullptr)
        mkdir(folderName_cstr, 0777);
    fileName += "_"+to_string(mpiRank);
    fileName = folderName+"/"+fileName;
    logFile.open(fileName);
    if(mpiRank == 0) {  // 主进程打印一些说明
        sayReadme();
    }
    logFile<<"thread_no:"<<mpiRank<<endl;
    moveFileCmd = "mv "+fileName+" .";
}

void finalize() {  // 关闭日志文件
    logFile.close();
    // system(moveFileCmd.c_str());
}

int MPI_Init(int *argc, char ***argv) {
    mpiErr = PMPI_Init(argc, argv);
    init();
    logFile<<"0"<<endl;
    return mpiErr;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    mpiErr = PMPI_Init_thread(argc, argv, required, provided);
    init();
    logFile<<"1"<<endl;
    return mpiErr;
}

int MPI_Finalize() {
    mpiErr = PMPI_Finalize();
    logFile<<"2"<<endl;
    finalize();
    return mpiErr;
}

void print_for_send_and_recv(int func_id, int count, MPI_Datatype datatype, int dest, float time) {
    logFile<<func_id<<","<<count<<","<<datatype<<","<<dest<<","<<time<<endl;
}

float get_elapsed_time() {
    return 1000000.0*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec;
}

int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    int globalDest = translate_one(comm, dest);
    mpiErr = PMPI_Ibsend(buf, count, datatype, dest, tag, comm, request);
    // logFile<<"3,"<<count<<","<<datatype<<","<<globalDest<<","<<0<<endl;
    print_for_send_and_recv(3, count, datatype, globalDest, noTime);
    return mpiErr;
}

int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    int globalDest = translate_one(comm, dest);
    mpiErr = PMPI_Irsend(buf, count, datatype, dest, tag, comm, request);
    // logFile<<"4,"<<count<<","<<datatype<<","<<globalDest<<","<<0<<endl;
    print_for_send_and_recv(4, count, datatype, globalDest, noTime);
    return mpiErr;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    int globalDest = translate_one(comm, dest);
    mpiErr = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    // logFile<<"5,"<<count<<","<<datatype<<","<<globalDest<<","<<0<<endl;
    print_for_send_and_recv(5, count, datatype, globalDest, noTime);
    return mpiErr;
}

int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    int globalDest = translate_one(comm, dest);
    mpiErr = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
    // logFile<<"6,"<<count<<","<<datatype<<","<<globalDest<<","<<0<<endl;
    print_for_send_and_recv(6, count, datatype, globalDest, noTime);
    return mpiErr;
}

int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    int globalDest = translate_one(comm, dest);
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Bsend(buf, count, datatype, dest, tag, comm);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"7,"<<count<<","<<datatype<<","<<globalDest<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(7, count, datatype, globalDest, get_elapsed_time());
    return mpiErr;
}

int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    int globalDest = translate_one(comm, dest);
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Rsend(buf, count, datatype, dest, tag, comm);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"8,"<<count<<","<<datatype<<","<<globalDest<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(8, count, datatype, globalDest, get_elapsed_time());
    return mpiErr;
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    // int local_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    int globalDest = translate_one(comm, dest);
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Send(buf, count, datatype, dest, tag, comm);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"9,"<<count<<","<<datatype<<","<<globalDest<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(9, count, datatype, globalDest, get_elapsed_time());
    return mpiErr;
}

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    int globalDest = translate_one(comm, dest);
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Ssend(buf, count, datatype, dest, tag, comm);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"10,"<<count<<","<<datatype<<","<<globalDest<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(10, count, datatype, globalDest, get_elapsed_time());
    return mpiErr;
}

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, \
int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, \
int source, int recvtag, MPI_Comm comm, MPI_Status *status){
    int globalDest = translate_one(comm, dest);
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"11,"<<sendcount<<","<<sendtype<<","<<globalDest<<","<<recvcount<<","<<recvtype<<","<<source<<","
    // logFile<<"11,"<<sendcount<<","<<sendtype<<","<<globalDest<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(11, sendcount, sendtype, globalDest, get_elapsed_time());
    return mpiErr;
}

int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, \
    int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status) {
    int globalDest = translate_one(comm, dest);
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"12,"<<count<<","<<datatype<<","<<dest<<","<<source<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    // logFile<<"12,"<<count<<","<<datatype<<","<<globalDest<<","<<count<<","<<datatype<<","<<source<<","
    // logFile<<"12,"<<count<<","<<datatype<<","<<globalDest<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(12, count, datatype, globalDest, get_elapsed_time());
    return mpiErr;
}

int MPI_Mrecv(void *buf, int count, MPI_Datatype type, MPI_Message *message, MPI_Status *status) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Mrecv(buf, count, type, message, status);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"13,"<<count<<","<<type<<","<<-1<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(13, count, type, noRoot, get_elapsed_time());
    return mpiErr;
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
    gettimeofday(&timeEnd, NULL);
    // logFile<<"14,"<<count<<","<<datatype<<","<<source<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    print_for_send_and_recv(14, count, datatype, noRoot, get_elapsed_time());
    return mpiErr;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request) {
    mpiErr = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    // logFile<<"15,"<<count<<","<<datatype<<","<<source<<","<<0<<endl;
    print_for_send_and_recv(15, count, datatype, noRoot, noTime);
    return mpiErr;
}

int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Request *request) {
    mpiErr = PMPI_Imrecv(buf, count, datatype, message, request);
    // logFile<<"16,"<<count<<","<<datatype<<","<<-1<<","<<0<<endl;
    print_for_send_and_recv(16, count, datatype, noRoot, noTime);
    return mpiErr;
}

void print_for_bcast_scatter_and_gather(int func_id, MPI_Comm comm, int root, float time,
                                        int root_transfer_count,    MPI_Datatype root_transfer_type,
                                        int notroot_transfer_count, MPI_Datatype notroot_transfer_type,
                                        const int* root_transfer_counts) {
    int local_rank;
    MPI_Comm_rank(comm, &local_rank);

    if(local_rank == root){ // root process 
        int size, global_rank;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
        vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

        logFile<<func_id<<","
                <<root_transfer_count<<","<<root_transfer_type<<","
                <<idx[local_rank]<<","<<time<<","<<size;
        for (int i = 0; i < size; i++) {
            logFile<<","<<idx[i];
            // if (root_transfer_counts != nullptr)
            //     logFile<<","<<root_transfer_counts[i];
        }
        logFile<<endl;
    } else { // not root process
        int globalRoot = translate_one(comm, root);
        logFile<<func_id<<","
                <<notroot_transfer_count<<","<<notroot_transfer_type<<","
                <<globalRoot<<","<<time<<","<<noSize<<endl;
    } 
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    // int local_rank;
    // MPI_Comm_rank(comm, &local_rank);
    //printf("pmpi.cpp: MPI_Bcast: (comm:%10p) size = %d, rank %3d(%10p) = rank %3d(%10p)\n", comm, size, local_rank, comm, comm_rank, MPI_COMM_WORLD);

    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Bcast(buffer, count, datatype, root, comm);
    gettimeofday(&timeEnd, NULL);
    print_for_bcast_scatter_and_gather(17, comm, root, get_elapsed_time(), count, datatype, count, datatype, nullptr);
    
    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"17,"<<count<<","<<datatype<<","<<idx[local_rank]<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i];
    //     logFile<<endl;
    // } else {
    //     int globalRoot = translate_one(comm, root);
    //     logFile<<"17,"<<count<<","<<datatype<<","<<globalRoot<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<0<<endl;
    // } 
    // logFile<<"17,"<<count<<","<<datatype<<","<<root<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;

    return mpiErr;
}

int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request *request) {
    // int local_rank;
    // MPI_Comm_rank(comm, &local_rank);
    mpiErr = PMPI_Ibcast(buffer, count, datatype, root, comm, request);
    print_for_bcast_scatter_and_gather(18, comm, root, noTime, count, datatype, count, datatype, nullptr);

    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"18,"<<count<<","<<datatype<<","<<idx[local_rank]<<","
    //             <<0<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i];
    //     logFile<<endl;
    // } else {
    //     int globalRoot = translate_one(comm, root);
    //     logFile<<"18,"<<count<<","<<datatype<<","<<globalRoot<<","
    //             <<0<<","
    //             <<0<<endl;
    // }
    // logFile<<"18,"<<count<<","<<datatype<<","<<root<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, \
void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    gettimeofday(&timeEnd, NULL);

    print_for_bcast_scatter_and_gather(19, comm, root, get_elapsed_time(), sendcount, sendtype, recvcount, recvtype, nullptr);

    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"19,"<<sendcount<<","<<sendtype<<","<<idx[local_rank]<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i];
    //     logFile<<endl;
    // } else {
    //     int globalRoot = translate_one(comm, root);
    //     logFile<<"19,"<<recvcount<<","<<recvtype<<","<<globalRoot<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<0<<endl;
    // } 
    // logFile<<"19,"<<sendcount<<","<<sendtype<<","<<root<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, \
void *recvbuf, int recvcount,MPI_Datatype recvtype, int root, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
    gettimeofday(&timeEnd, NULL);

    print_for_bcast_scatter_and_gather(20, comm, root, get_elapsed_time(), -1, sendtype, recvcount, recvtype, sendcounts);

    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"20,"<<-1<<","<<sendtype<<","<<idx[local_rank]<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i]<<","<<sendcounts[i];
    //     logFile<<endl;
    // } else {
    //     int globalRoot = translate_one(comm, root);
    //     logFile<<"20,"<<recvcount<<","<<recvtype<<","<<globalRoot<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<0<<endl;
    // }
    // logFile<<"20,"<<sendcounts<<","<<sendtype<<","<<root<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, \
int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);

    print_for_bcast_scatter_and_gather(21, comm, root, noTime, sendcount, sendtype, recvcount, recvtype, nullptr);
    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"21,"<<sendcount<<","<<sendtype<<","<<root<<","
    //             <<0<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i];
    //     logFile<<endl;
    // } else {
    //     int globalRoot = translate_one(comm, root);
    //     logFile<<"21,"<<recvcount<<","<<recvtype<<","<<globalRoot<<","
    //             <<0<<","
    //             <<0;
    // }
    // logFile<<"21,"<<sendcount<<","<<sendtype<<","<<root<<endl;
    return mpiErr;
}

int MPI_Iscatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, \
void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request);

    print_for_bcast_scatter_and_gather(22, comm, root, noTime, -1, sendtype, recvcount, recvtype, sendcounts);
    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"22,"<<-1<<","<<sendtype<<","<<root<<","
    //             <<0<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i]<<","<<sendcounts[i];
    //     logFile<<endl;
    // } else {
    //     int globalRoot = translate_one(comm, root);
    //     logFile<<"22,"<<recvcount<<","<<recvtype<<","<<globalRoot<<","
    //             <<0<<","
    //             <<0;
    // }

    // logFile<<"22,"<<sendcounts<<","<<sendtype<<","<<root<<endl;
    return mpiErr;
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, \
MPI_Datatype recvtype, int root, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    gettimeofday(&timeEnd, NULL);

    print_for_bcast_scatter_and_gather(23, comm, root, get_elapsed_time(), recvcount, recvtype, sendcount, sendtype, nullptr);

    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"23,"<<recvtype<<","<<recvtype<<","<<root<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i];
    //     logFile<<endl;
    // } 
    // logFile<<"23,"<<sendcount<<","<<sendtype<<","<<root<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, \
const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
    gettimeofday(&timeEnd, NULL);

    print_for_bcast_scatter_and_gather(24, comm, root, get_elapsed_time(), -1, recvtype, sendcount, sendtype, recvcounts);

    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"24,"<<-1<<","<<recvtype<<","<<root<<","
    //             <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i]<<","<<recvcounts[i];
    //     logFile<<endl;
    // } 
    // logFile<<"24,"<<sendcount<<","<<sendtype<<","<<root<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, \
MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);

    print_for_bcast_scatter_and_gather(25, comm, root, noTime, recvcount, recvtype, sendcount, sendtype, nullptr);
    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"25,"<<recvcount<<","<<recvtype<<","<<root<<","
    //             <<0<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i];
    //     logFile<<endl;
    // } 
    // logFile<<"25,"<<sendcount<<","<<sendtype<<","<<root<<endl;
    return mpiErr;
}

int MPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], \
const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request);

    print_for_bcast_scatter_and_gather(26, comm, root, noTime, -1, recvtype, sendcount, sendtype, recvcounts);
    // if(local_rank == root){
    //     int size, global_rank;
    //     MPI_Comm_size(comm, &size);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    //     vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    //     logFile<<"26,"<<-1<<","<<recvtype<<","<<root<<","
    //             <<0<<","
    //             <<size;
    //     for (int i = 0; i < size; i++) 
    //         logFile<<","<<idx[i]<<","<<recvcounts[i];
    //     logFile<<endl;
    // }
    // logFile<<"26,"<<sendcount<<","<<sendtype<<","<<root<<endl;
    return mpiErr;
}

void print_for_allgather(int func_id, MPI_Comm comm, float time,
                            int sendcount, MPI_Datatype sendtype) {
    int local_rank;
    MPI_Comm_rank(comm, &local_rank);

    int size, global_rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    logFile<<func_id<<","
            <<sendcount<<","<<sendtype<<","<<idx[local_rank]<<","
            <<time<<","<<size;
    for (int i = 0; i < size; i++) 
        logFile<<","<<idx[i];
    logFile<<endl;

}

int MPI_Allgather(const void *sendbuf, int  sendcount, MPI_Datatype sendtype, \
void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    gettimeofday(&timeEnd, NULL);

    print_for_allgather(27, comm, get_elapsed_time(), sendcount, sendtype);

    // int size, global_rank;
    // MPI_Comm_size(comm, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    // vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    // logFile<<"27,"<<sendcount<<","<<sendtype<<","<<-1<<","
    //         <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //         <<size;
    // for (int i = 0; i < size; i++) 
    //     logFile<<","<<idx[i];
    // logFile<<endl;
    // logFile<<"27,"<<sendcount<<","<<sendtype<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], \
const int displs[], MPI_Datatype recvtype, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    gettimeofday(&timeEnd, NULL);

    print_for_allgather(28, comm, get_elapsed_time(), sendcount, sendtype);
    // int size, global_rank;
    // MPI_Comm_size(comm, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    // vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    // logFile<<"28,"<<sendcount<<","<<recvtype<<","<<-1<<","
    //         <<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<","
    //         <<size;
    // for (int i = 0; i < size; i++) 
    //     logFile<<","<<idx[i];
    // logFile<<endl;

    // logFile<<"28,"<<sendcount<<","<<sendtype<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Iallgather(const void *sendbuf, int  sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, \
MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);

    print_for_allgather(29, comm, noTime, sendcount, sendtype);
    // int size, global_rank;
    // MPI_Comm_size(comm, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    // vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    // logFile<<"29,"<<sendcount<<","<<sendtype<<","<<-1<<","
    //         <<0<<","
    //         <<size;
    // for (int i = 0; i < size; i++) 
    //     logFile<<","<<idx[i];
    // logFile<<endl;

    // logFile<<"29,"<<sendcount<<","<<sendtype<<endl;
    return mpiErr;
}

int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], \
const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);

    print_for_allgather(30, comm, noTime, sendcount, sendtype);
    // int size, global_rank;
    // MPI_Comm_size(comm, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    // vector<int> idx = translate_all(comm); // idx[local_rank] = currrent process's rank in MPI_COMM_WORLD     

    // logFile<<"30,"<<sendcount<<","<<sendtype<<","<<-1<<","
    //         <<0<<","
    //         <<size;
    // for (int i = 0; i < size; i++) 
    //     logFile<<","<<idx[i];
    // logFile<<endl;
    // logFile<<"30,"<<sendcount<<","<<sendtype<<endl;
    return mpiErr;
}

int MPI_Neighbor_allgather(const void *sendbuf, int  sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, \
MPI_Datatype recvtype, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Neighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    gettimeofday(&timeEnd, NULL);
    logFile<<"31,"<<sendcount<<","<<sendtype<<","<<recvcount<<","<<recvtype<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Neighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], \
const int displs[], MPI_Datatype recvtype, MPI_Comm comm) {
    gettimeofday(&timeStart, NULL);
    mpiErr = PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    gettimeofday(&timeEnd, NULL);
    logFile<<"32,"<<sendcount<<","<<sendtype<<","<<recvcounts<<","<<recvtype<<","<<1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec<<endl;
    return mpiErr;
}

int MPI_Ineighbor_allgather(const void *sendbuf, int  sendcount, MPI_Datatype sendtype, void *recvbuf, \
int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    logFile<<"33,"<<sendcount<<","<<sendtype<<","<<recvcount<<","<<recvtype<<endl;
    return mpiErr;
}

int MPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], \
const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request) {
    mpiErr = PMPI_Ineighbor_allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);
    logFile<<"34,"<<sendcount<<","<<sendtype<<","<<recvcounts<<","<<recvtype<<endl;
    return mpiErr;
}


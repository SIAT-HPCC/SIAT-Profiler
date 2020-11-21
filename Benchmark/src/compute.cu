#include <vector>
#include <iostream>
#include <map>
#include <math.h>

using namespace std;

char **device_reads;  // 显存中保存序列的数组
int *device_readLength;  // 显存中保存序列长度的数组
int *device_indexTableOne, *device_indexTableTwo;  // 两个index table
int host_sameWords=0, *device_sameWords;  // 相同的word数

// copyData 把序列拷贝到显存当中，序列放入device_reads，序列长度放入device_readLength
void copyData(vector<string> &reads, char **device_reads, int *device_readLength) {
    // 定义数据结构
    char **host_reads;  // 与device_reads对应的主机数据结构
    host_reads = (char**)malloc(reads.size()*sizeof(char*));
    int *host_readLength;  // 与device_readLength对应的主机数据结构；
    host_readLength = (int*)malloc(reads.size()*sizeof(int));
    // 拷贝数据
    for(int i=0; i<reads.size(); i++) {
        char *device_readTemp;  // 临时存放序列数据
        cudaMalloc((void**)&device_readTemp, reads[i].length()*sizeof(char));
        cudaMemcpy(device_readTemp, reads[i].data(), reads[i].length()*sizeof(char), cudaMemcpyHostToDevice);
        host_reads[i] = device_readTemp;
        host_readLength[i] = reads[i].length();
    }
    cudaMemcpy(device_reads, host_reads, reads.size()*sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(device_readLength, host_readLength, reads.size()*sizeof(int), cudaMemcpyHostToDevice);
    //收尾
    free(host_reads);
    free(host_readLength);
}

// kernel_convertBaseToNumber 根据device_convertor把碱基序列转换为char的数字，转换后的数字存储在device_read中。
__global__ void kernel_convertBaseToNumber(char **device_reads, int *device_readLength, char *device_convertor, int readCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    for(int i=0; i<readCount; i++) {
        if(index < device_readLength[i]) {
            device_reads[i][index] = device_convertor[device_reads[i][index]-'A'];
        }
    }
}

// convertBaseToNumber 根据device_convertor把碱基序列转换为char的数字，转换后的数字存储在device_read中
void convertBaseToNumber(vector<string> &reads, char **device_reads, int *device_readLength) {
    // 基因序列转换器          A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
    char baseConvertor[26] = {0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4};
    char *device_convertor;
    cudaMalloc((void**)&device_convertor, 26*sizeof(char));
    cudaMemcpy(device_convertor, baseConvertor, 26*sizeof(char), cudaMemcpyHostToDevice);
    // 转换数据
    kernel_convertBaseToNumber<<<(reads[0].length()+1023)/1024, 1024>>>(device_reads, device_readLength, device_convertor, reads.size());
    // 收尾
    cudaFree(device_convertor);
}

__global__ void test(char **device_reads) {
    printf("%d\n", (int)device_reads[3][3]);
}

// prepareData 分配显存和拷贝数据到显存
void prepareData(vector<string> &reads, int wordLength, int mpiRank) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(mpiRank%deviceCount);
    cudaMalloc((void**)&device_reads, reads.size()*sizeof(char*));  // 分配reads显存
    cudaMalloc((void**)&device_readLength, reads.size()*sizeof(int*));  // 分配readsLength显存
    copyData(reads, device_reads, device_readLength);  // 拷贝数据到显存
    convertBaseToNumber(reads, device_reads, device_readLength);  // 把碱基转换为数字
    cudaMalloc((void**)&device_indexTableOne, (1<<wordLength*2)*sizeof(int));  // 分配indexTable的空间
    cudaMalloc((void**)&device_indexTableTwo, (1<<wordLength*2)*sizeof(int));
    cudaMalloc((void**)&device_sameWords, sizeof(int));  // 分配sameWords空间
}

// freeData 释放显存
void freeData() {
    cudaFree(device_reads);
    cudaFree(device_readLength);
    cudaFree(device_indexTableOne);
    cudaFree(device_indexTableTwo);
    cudaFree(device_sameWords);
}

// kernel_generateIndexTable 根据输入序列生成indextable
__global__ void kernel_generateIndexTable(char **device_reads, int *device_readLength, int readIndex, int *device_indexTable, int wordLength) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < device_readLength[readIndex]-wordLength+1) {  // 结尾的几个元素没法算
        int indexTableIndex = 0;  // index table的索引
        int indexTableValue = 1;  // index table的索引位置要增加的值
        for(int j=0; j<wordLength; j++) {
            if(device_reads[readIndex][index+j] != 4) {
                indexTableIndex += device_reads[readIndex][index+j]<<j*2;
            } else {  // 如果是4，说明这是个未知碱基，直接退出
                indexTableValue = 0;
                break;
            }
        }
        atomicAdd(&device_indexTable[indexTableIndex], indexTableValue);
    }
}

// kernel_sameWordCounter 计数相同的wrod的个数
__global__ void kernel_sameWordCounter(int *device_indexTableOne, int *device_indexTableTwo, int wordLength, int *device_sameWords) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ int tempResult[1024];
    for(int i=threadIdx.x; i<1024; i+=blockDim.x) {
        tempResult[i] = 0;
    }
    if(index < 1<<wordLength*2) {
        tempResult[threadIdx.x] = min((int)device_indexTableOne[index], (int)device_indexTableTwo[index]);
        // 规约
        __syncthreads();  // 先等结果都算完了再规约
        for(int i=blockDim.x/2; i>0; i/=2) {
            if(threadIdx.x < i) {
                tempResult[threadIdx.x] += tempResult[threadIdx.x+i];
            }
            __syncthreads();  // 每轮都要同步
        }
        if(threadIdx.x == 0) {
            atomicAdd(device_sameWords, tempResult[0]);
        }
    }
}

// dynamicProgramming 输入两个序列，计算最长公共子串的长度，存入device_longestCommonSubsequence
__global__ void dynamicProgramming(char **device_reads, int *device_readLength, int readOne, int readTwo, int *device_longestCommonSubsequence) {
    extern __shared__ char tempData[];
    int index = threadIdx.x;
    while(index<device_readLength[readOne]*5+device_readLength[readTwo]*6+6) {
        tempData[index] = 0;
        index += blockDim.x;
    }
    short *tempPrevious = (short*) &tempData[0];
    short *tempNow      = (short*) &tempData[device_readLength[readTwo]*2+2];
    short *tempNext     = (short*) &tempData[device_readLength[readTwo]*4+4];
    char  *baseTable[5] = { (char*)  &tempData[device_readLength[readTwo]*6+6],                                     // A
                            (char*)  &tempData[device_readLength[readTwo]*6+6+device_readLength[readOne]*1],   // T
                            (char*)  &tempData[device_readLength[readTwo]*6+6+device_readLength[readOne]*2],   // G
                            (char*)  &tempData[device_readLength[readTwo]*6+6+device_readLength[readOne]*3],   // C
                            (char*)  &tempData[device_readLength[readTwo]*6+6+device_readLength[readOne]*4]};  // N
                        //   A  T  G  C  N
    char convertor[5][5] ={ {1, 0, 0, 0, 0},   // A
                            {0, 1, 0, 0, 0},   // T
                            {0, 0, 1, 0, 0},   // G
                            {0, 0, 0, 1, 0},   // C
                            {0, 0, 0, 0, 0}};  // N
    index = threadIdx.x;
    while(index<device_readLength[readOne]) {
        baseTable[0][index] = convertor[0][device_reads[readOne][index]];
        baseTable[1][index] = convertor[1][device_reads[readOne][index]];
        baseTable[2][index] = convertor[2][device_reads[readOne][index]];
        baseTable[3][index] = convertor[3][device_reads[readOne][index]];
        baseTable[4][index] = convertor[4][device_reads[readOne][index]];
        index += blockDim.x;
    }
    for(int i=1; i<device_readLength[readOne]+device_readLength[readTwo]; i++) {
        index = threadIdx.x + 1;
        while(index<device_readLength[readTwo]+1) {  // 计算tempNext
            if(index-1<i && i<device_readLength[readOne]+index) {
                tempNext[index] = tempPrevious[index-1]+baseTable[device_reads[readTwo][index-1]][i-index];
                tempNext[index] = (short)max((int)tempNext[index], (int)tempPrevious[index-1]);
                tempNext[index] = (short)max((int)tempNext[index], (int)tempNow[index]);
                tempNext[index] = (short)max((int)tempNext[index], (int)tempNow[index-1]);
            }
            index += blockDim.x;
        }
        __syncthreads();  // 保证tempNext都计算完成
        index = threadIdx.x + 1;
        while(index<device_readLength[readTwo]+1) {  // 滑动tempNext，tempNow，tempPrevious
            tempPrevious[index] = tempNow[index];
            tempNow[index] = tempNext[index];
            index += blockDim.x;
        }
        __syncthreads();  // 保证滑动都完成
    }
    __syncthreads();  // 保证结果计算完成
    if(threadIdx.x==0) {
        *device_longestCommonSubsequence = tempNow[device_readLength[readTwo]];
    }
}

// clustering 聚类序列，结果保存在cluster中，相似度保存在similarity中
int clustering(vector<string> &reads, int wordLength, double threadhold, int readOne, int readTwo) {
    // 开始聚类
    cudaMemset(device_indexTableOne, 0, (1<<wordLength*2)*sizeof(int));  // indexTable清零
    kernel_generateIndexTable<<<(readOne+1023)/1024, 1024>>>(device_reads, device_readLength, readOne, device_indexTableOne, wordLength);
    cudaMemset(device_indexTableTwo, 0, (1<<wordLength*2)*sizeof(int));
    kernel_generateIndexTable<<<(readTwo+1023)/1024, 1024>>>(device_reads, device_readLength, readTwo, device_indexTableTwo, wordLength);
    cudaMemset(device_sameWords, 0, sizeof(int));  // samewords清零
    kernel_sameWordCounter<<<((1<<wordLength*2)+1023)/1024, 1024>>>(device_indexTableOne, device_indexTableTwo, wordLength, device_sameWords);
    cudaMemcpy(&host_sameWords, device_sameWords, sizeof(int), cudaMemcpyDeviceToHost);  // 得到samewords
    int shorterLength = min(reads[readOne].length(), reads[readTwo].length());
    if(host_sameWords<(shorterLength+1-wordLength-shorterLength*(1-threadhold)*wordLength)) {  // 明显小于阈值，返回0
        return 0;
    } else {  // 大于阈值就用动态规划重算
        int host_longestCommonSubsequence, *device_longestCommonSubsequence;
        cudaMalloc((void**)&device_longestCommonSubsequence, sizeof(int));
        // 动态规划计算相似度
        dynamicProgramming<<<1, 1024, reads[readOne].length()*5+reads[readTwo].length()*6+6>>>(device_reads, device_readLength, readOne, readTwo, device_longestCommonSubsequence);
        cudaMemcpy(&host_longestCommonSubsequence, device_longestCommonSubsequence, sizeof(int), cudaMemcpyDeviceToHost);
        if(host_longestCommonSubsequence<shorterLength*threadhold) {  // 小于阈值，返回0
            cudaFree(device_longestCommonSubsequence);
            return 0;
        } else {
            cudaFree(device_longestCommonSubsequence);
            return 1;
        }
    }
    return 0;
}
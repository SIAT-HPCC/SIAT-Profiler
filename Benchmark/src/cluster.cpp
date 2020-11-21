// readFile.cpp
// 多节点的聚类版本
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <map>
#include <sys/time.h>

#include "mpi.h"
#include "compute.h"

using namespace std;

// checkInput 判断输入的参数是否正确
void checkInput(int argc, char **argv) {
    if(argc != 2) {
        cout<<"输入参数错误，使用方法："<<endl;
        cout<<"cluster fileName"<<endl;
        exit(0);
    }
    ifstream file(argv[1]);
    if(!file) {
        cout<<"文件不存在，使用方法："<<endl;
        cout<<"cluster fileName"<<endl;
        exit(0);
    }
    file.close();
}

// sortReads 根据read的长度从长到短，给reads和names排序
void sortReads(vector<string> &reads, vector<string> &names, int left, int right) {
    if(left > right) {
        return;
    }
    int i = left;
    int j = right;
    string readTemp = reads[i];
    string nameTemp = names[i];
    while(i < j) {
        while((reads[j].length()<=readTemp.length())and(i<j)) {
            j--;
        }
        reads[i] = reads[j];
        names[i] = names[j];
        while((reads[i].length()>=readTemp.length())and(i<j)) {
            i++;
        }
        reads[j] = reads[i];
        names[j] = names[i];
    }
    reads[i] = readTemp;
    names[i] = nameTemp;
    sortReads(reads, names, left, i-1);
    sortReads(reads, names, j+1, right);
}

// readFile 读取fileName文件，read存到reads，read的名存到names
void readFile(string fileName, vector<string> &reads, vector<string> &names) {
    reads.clear();
    names.clear();
    ifstream file(fileName);
    if(file) {
        string line;
		while (getline(file, line)) { // getline不读换行符
            if(line[0] == '>') {
                names.push_back(line);
            } else {
                reads.push_back(line);
            }
		}
	}
    file.close();
    sortReads(reads, names, 0, reads.size()-1);  // 从长到短给reads和names排序
}

int main(int argc, char **argv) {
    int mpiSize, mpiRank;  // 总mpi进程数，本进程编号
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    if(mpiSize < 2) {  // 程序需要至少两个节点，节点0统筹，其他节点计算。
        cout<<"程序需要进程数最少为2"<<endl;
        exit(0);
    }
    // 定义数据结构
    checkInput(argc, argv);  // 检查输入
    int wordLength = 8;  // k-mer的长度
    double threadhold = 0.95;  // 聚类相似度阈值
    vector<string> reads;  // 保存序列的数组
    vector<string> names;  // 序列名字的数组
    readFile((string)argv[1], reads, names);  // 读文件
    prepareData(reads, wordLength, mpiRank);  // 准备设备端数据

    int readsRemainder[reads.size()+1];  // 剩余的待聚类序列，[0]是剩余read数，[1]是最新的代表read，之后是要聚类的read序号
    readsRemainder[0] = reads.size()-1;
    for(int i=0; i<reads.size(); i++) {  // 初始化readsRemainder，第一个类是0
        readsRemainder[i+1] = i;
    }
    int readsCluster[reads.size()];  // 总体聚类结果，每个元素表示一条read，值为类的代表read序号，-1表示read未聚类
    for(int i=0; i<reads.size(); i++) {
        readsCluster[i] = -1;
    }
    // 开始计算
    int clusteringResults[(reads.size()/(mpiSize-1)+1)*mpiSize];  // 每个进程聚类的结果汇总，多分配点空间，防止溢出
    int tempResults[reads.size()/(mpiSize-1)+1];  // 每个进程聚类的结果
    MPI_Barrier(MPI_COMM_WORLD);  // 进程同步
    if(mpiRank == 0) {  // 进程0处理readsCluster和readsRemainder
        cout<<"读取数据完成，开始计算"<<endl;
        while(1) {
            MPI_Bcast(readsRemainder, reads.size()+1, MPI_INT, 0, MPI_COMM_WORLD);  // 广播readsRemainder
            if(readsRemainder[0] == 0) {  // 如果聚类完成则退出
                readsCluster[readsRemainder[1]] = readsRemainder[1];  // 写入代表read
                break;
            }
            MPI_Gather(tempResults, readsRemainder[0]/(mpiSize-1)+1, MPI_INT,
            clusteringResults, readsRemainder[0]/(mpiSize-1)+1, MPI_INT, 0, MPI_COMM_WORLD);  // 收集进程聚类结果
            readsCluster[readsRemainder[1]] = readsRemainder[1];  // 写入代表read
            for(int i=0; i<readsRemainder[0]; i++) {  // 根据进程聚类结果更新总体聚类结果
                readsCluster[readsRemainder[2+i]] = clusteringResults[readsRemainder[0]/(mpiSize-1)+1+i];
            }
            while(readsCluster[readsRemainder[1]]!=-1 and readsRemainder[1]<reads.size()-1) {  // 找到下一个代表read
                readsRemainder[1]++;
            }
            int readsCount = 0;  // 剩余需要聚类的read数
            for(int i=readsRemainder[1]+1; i<reads.size(); i++) {  // 根据总体聚类结果更新readsRemainder
                if(readsCluster[i] == -1) {
                    readsCount++;
                    readsRemainder[readsCount+1] = i;
                }
            }
            readsRemainder[0] = readsCount;
            cout<<"当前代表序列/总序列数"<<readsRemainder[1]<<"/"<<readsRemainder[0]<<endl;
        }
        freeData();
        int sum = 0;
        for(int i=0; i<reads.size(); i++) {
            if(readsCluster[i] == i) {
                sum++;
            }
        }
        cout<<"聚为"<<sum<<"类"<<endl;
        MPI_Finalize();
    } else {  // 其他进程计算
        while(1) {
            MPI_Bcast(readsRemainder, reads.size()+1, MPI_INT, 0, MPI_COMM_WORLD);  // 广播readsRemainder
            if(readsRemainder[0] == 0) {  // 如果聚类完成则退出
                break;
            }
            int start = (readsRemainder[0]/(mpiSize-1)+1)*(mpiRank-1)+2;
            for(int i=0; i<readsRemainder[0]/(mpiSize-1)+1; i++) {  // 逐一计算序列相似度
                if(start>readsRemainder[0]+1) {  // 超出范围
                    break;
                }
                int temp = clustering(reads, wordLength, threadhold, readsRemainder[1], readsRemainder[start]);
                if(temp) {
                    tempResults[i] = readsRemainder[1];
                } else {
                    tempResults[i] = -1;
                }
                start++;
            }
            MPI_Gather(tempResults, readsRemainder[0]/(mpiSize-1)+1, MPI_INT,
            clusteringResults, readsRemainder[0]/(mpiSize-1)+1, MPI_INT, 0, MPI_COMM_WORLD);  // 收集进程聚类结果
        }
        freeData();
        MPI_Finalize();
    }
    return 0;
}

/*
cudaError_t err = cudaSuccess;
if(err != cudaSuccess) {
    cout<<cudaGetErrorString(err)<<endl;
}
*/

#include <vector>

using namespace std;

// prepareData 分配显存和拷贝数据到显存
void prepareData(vector<string> &reads, int woreLength, int mpiRank);
// clustering 聚类序列，相似度小于阈值就返回0，大于阈值返回1
int clustering(vector<string> &reads, int wordLength, double threadhold, int readOne, int readTwo);
// freeData 释放显存
void freeData();

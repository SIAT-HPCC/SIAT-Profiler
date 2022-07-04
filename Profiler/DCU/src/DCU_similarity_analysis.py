from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os
from DCU_performance_analysis import Data_generator
import argparse

class Data_similarity_analyser:
    def read_and_normalize_data(self, dir, generator):
        data = generator.read_performance_data(dir)

        # 对数据分组，并且求均值
        data_groupby_kernel_mean = generator.groupby_and_mean_data(data)
        data_groupby_kernel_mean = data_groupby_kernel_mean.drop(labels = ['Index', 'KernelName'], axis=1)
        data_groupby_kernel_mean_norm = generator.normalize_data(data_groupby_kernel_mean)

        return data_groupby_kernel_mean_norm

    def analyse_different_dcu_in_one_node(self, node_id, dcu_data_list, dcu_number_per_node, similarity_threshold):
        similarity_array = np.ones([dcu_number_per_node, dcu_number_per_node])
        for i in range(0, dcu_number_per_node):
            for j in range(i + 1, dcu_number_per_node):
                # 行数或列数不相同，相似度直接设置成0
                if dcu_data_list[i].shape[0] != dcu_data_list[j].shape[0] or dcu_data_list[i].shape[1] != dcu_data_list[j].shape[1]:
                    print("On node " + str(node_id) + ", between DCU " + str(i) + " and DCU " + str(j) + ", the number of kernel function is different.")
                    similarity_array[i][j] = 0.0
                    continue

                # similarity_array[i][j] = 1.0
                # 取所有皮尔逊系数中的最小值作为当前两个DCU之间的相似度
                for row in range(dcu_data_list[i].shape[0]):
                    similarity_array[i][j] = min(pearsonr(dcu_data_list[i].iloc[row], dcu_data_list[j].iloc[row])[0], similarity_array[i][j])
                similarity_array[i][j] = round(similarity_array[i][j], 4)
                if similarity_array[i][j] < similarity_threshold:
                    print("On node " + str(node_id) + ", the similarity coefficient of hardware counters between DCU " + str(i) + " and DCU " + str(j)  \
                                    + " is " + str(similarity_array[i][j]) \
                                    + ", less than the similarity threshold(" + str(similarity_threshold) + ").")
                else:
                    print("On node " + str(node_id) + ", the similarity coefficient of hardware counters between DCU " + str(i) + " and DCU " + str(j)  \
                                    + " is " + str(similarity_array[i][j]) \
                                    + ", greater than the similarity threshold(" + str(similarity_threshold) + ").")

    def analyse_different_dcu_in_different_node(self, node_id, pre_dcu_data_list, dcu_data_list, dcu_number_per_node, similarity_threshold):
        # 如果是第一个节点，则无法进行节点间的相似度分析，直接返回
        if node_id == 0:
            return
        
        for dcu_id in range(dcu_number_per_node):
            if pre_dcu_data_list[dcu_id].shape[0] != dcu_data_list[dcu_id].shape[0]:
                print("Between node " + str(node_id - 1) + " and node " + str(node_id) + ", on DCU " + str(dcu_id) + ", the number of kernel function is different.")
                continue

            min_similarity = 1.0
            row_num = dcu_data_list[dcu_id].shape[0]
            for row in range(row_num):
                min_similarity = min(pearsonr(pre_dcu_data_list[dcu_id].iloc[row], dcu_data_list[dcu_id].iloc[row])[0], min_similarity)
            min_similarity = round(min_similarity, 4)
            if min_similarity < similarity_threshold:
                print("Between node " + str(node_id - 1) + " and node " + str(node_id) + ", the similarity coefficient of hardware counters on DCU " + str(dcu_id) \
                                + " is " + str(min_similarity) \
                                + ", less than the similarity threshold(" + str(similarity_threshold) + ").")
            else:
                print("Between node " + str(node_id - 1) + " and node " + str(node_id) + ", the similarity coefficient of hardware counters on DCU " + str(dcu_id) \
                                + " is " + str(min_similarity) \
                                + ", greater than the similarity threshold(" + str(similarity_threshold) + ").")
                

    def analyse_similarity(self, dir, node_number, dcu_number_per_node, similarity_threshold):
        generator = Data_generator()
        # 遍历所有节点
        file_no = 0
        pre_dcu_data_list = []
        for node_id in range(node_number):
            if node_id != 0:
                pre_dcu_data_list = dcu_data_list
            dcu_data_list = []
            # 读取同一个节点上不同DCU的硬件计数器结果（遍历同一个节点上的不同DCU）
            for dcu_id in range(dcu_number_per_node):
                # data = self.read_and_normalize_data(dir + '/counter' + str(file_no) + '.csv', generator)
                data = self.read_and_normalize_data(os.path.join(dir, 'counter' + str(file_no) + '.csv'), generator)
                # 去除nan的列
                data = data.dropna(axis=1,how='any')
                dcu_data_list.append(data)

            file_no += dcu_number_per_node

            self.analyse_different_dcu_in_one_node(node_id, dcu_data_list, dcu_number_per_node, similarity_threshold)
            self.analyse_different_dcu_in_different_node(node_id, pre_dcu_data_list, dcu_data_list, dcu_number_per_node, similarity_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hardware configuration information')
    parser.add_argument("-N", "--node_num", dest='node_num', type=int, help='The total number of compute node.')
    parser.add_argument("-n", "--dcu_num_per_node", dest='dcu_num_per_node', type=int, help='The number of DCU per compute node.')
    parser.add_argument("-i", "--input_path", dest='input_path', type=str, help='The filepath of folder/directory includes performance data.')
    parser.add_argument("-t", "--similarity_threshold", dest='similarity_threshold', type=float, default=0.9, \
                        help='The similarity threshold, similarity value between two DCUs smaller than it represents the performance result generated by these DCUs are not similar.')

    args = parser.parse_args()
    analyser = Data_similarity_analyser()
    analyser.analyse_similarity(args.input_path, args.node_num, args.dcu_num_per_node, args.similarity_threshold)

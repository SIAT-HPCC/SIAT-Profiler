
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pow

from mpi_info import mpi_datatype
from mpi_info import mpi_func

message_dict = {}

def count_thread(DIR):
    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


# def count_message(file, message_dict, thread_num):
#     for line in file:
# #         print(line)
#         line=line.strip('\n')
#         temp_list = line.split(',')
#         if len(temp_list) <= 1:
#             continue
#         func_no = int(temp_list[0])
#         if temp_list[2] not in mpi_datatype:
#             temp_list[2] = r'0x6261c0'
#         size = 0
#         if func_no >= 3 and func_no <= 16:
#             size = int(temp_list[1]) / 2 * mpi_datatype[temp_list[2]][1] / 1024
#         elif func_no >= 17 and func_no <= 26:
# #             print(temp_list)
#             size = int(temp_list[1]) * mpi_datatype[temp_list[2]][1] / 1024
#         elif func_no >= 27 and func_no <= 34:
#             size = int(temp_list[1]) * mpi_datatype[temp_list[2]][1] * (thread_num-1) / 1024
#         if size in message_dict:
#             message_dict[size] += 1
#         else:
#             message_dict[size] = 1
# #     print(message_dict)

# def sum_message(my_dict):
#     sum = 0
#     for i in my_dict: 
#         sum = sum + my_dict[i] 
#     return sum

def draw_bar(filename, message_dict):
    axis_y = list()
    max_size = int(max(message_dict.keys())) + 1
    i = 1
    while(True):
        axis_y.append(pow(2, i))
        if axis_y[-1] >= max_size:
            break
        i += 1
    
    temp_dict = {}
    for key in message_dict.keys():
        temp_key = int(key)
        for i in axis_y:
            if i > temp_key:
                if i in temp_dict:
                    temp_dict[i] += message_dict[key]
                else:
                    temp_dict[i] = message_dict[key]
                break
    # print(temp_dict)
    
    axis_y.insert(0, 0)
    
    x = []
    y = []
    
    for i in range(0, len(axis_y)-1):
        x.append(str(axis_y[i]) + "-" + str(axis_y[i+1]))
        if axis_y[i+1] in temp_dict:
            y.append(temp_dict[axis_y[i+1]])
        else:
            y.append(0)
    
    # print(x)
    # print(y)
    plt.barh(x, y, height=0.5)
    plt.xlabel('message num')
    plt.ylabel('message size(KBytes)')
#     fig = plt.figure(figsize=(10,10), dpi=200)
    # plt.xticks(rotation=60)
    plt.title("MPI message size and MPI message number statistics")
    plt.savefig(filename, dpi=200)
    # plt.show()
    print("draw MPI message size and MPI message number statistics successfully.")

import os
import time
import re
from mpi_info import mpi_datatype

def count_thread(DIR):
    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

def is_send_func(func_id):
    if func_id >= 3 and func_id <= 12:
        return True
    return False

def is_bcast_scatter_or_gather_func(func_id):
    if func_id >= 17 and func_id <= 26:
        return True
    return False

def is_allgather_func(func_id):
    if func_id >= 27 and func_id <= 30:
        return True
    return False

def is_neighbor_allgather_func(func_id):
    if func_id >= 31 and func_id <= 34:
        return True
    return False

def check_data_type(data_type):
    if data_type not in mpi_datatype:
        data_type = r'0x6261c0'
    return data_type

def get_message_size(temp_list):
    data_count = int(temp_list[1])
    data_type = check_data_type(temp_list[2])
    sizeof_data_type = mpi_datatype[data_type][1]
    return data_count * sizeof_data_type / 1024

def count_thread_sendrecv_and_messagesize(line, line_no, thread_no, temp_list, thread_send_list, message_dict):
    if len(temp_list) <= 1:
        return

    func_id = int(temp_list[0])

    cur_proc_id = thread_no
    message_size = 0
        
    # 计算每个进程的通信数据量
    if is_send_func(func_id):
        message_size = get_message_size(temp_list)
        dest_proc_id = int(temp_list[3])
        thread_send_list[cur_proc_id][dest_proc_id] += message_size
        if message_size in message_dict:
            message_dict[message_size] += 1
        else:
            message_dict[message_size] = 1

    elif is_bcast_scatter_or_gather_func(func_id):
        is_child_proc = True if int(temp_list[5]) == 0 else False
        if is_child_proc:
            message_size = get_message_size(temp_list)
            root_proc_id = int(temp_list[3])
            if func_id <= 22: # bcast or scatter func
                thread_send_list[root_proc_id][cur_proc_id] += message_size
            else: # gather func
                thread_send_list[cur_proc_id][root_proc_id] += message_size
            if message_size in message_dict:
                message_dict[message_size] += 1
            else:
                message_dict[message_size] = 1

    elif is_allgather_func(func_id):
        message_size = get_message_size(temp_list)
        proc_size = int(temp_list[5])

        for i in range(proc_size):
            proc_id = int(temp_list(6+i))
            if proc_id != cur_proc_id:
                thread_send_list[cur_proc_id][proc_id] += message_size
        if message_size in message_dict.keys():
            message_dict[message_size] += proc_size - 1
        else:
            message_dict[message_size] = proc_size - 1

    elif is_neighbor_allgather_func(func_id):
        print("not support!")
    #     if temp_list[4] not in mpi_datatype:
    #         temp_list[4] = r'0x6261c0'
    #     for i in range(0, thread_num):
    #         thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]

# def count_message(temp_list, message_dict, thread_num):            
#     if len(temp_list) <= 1:
#         return

#     func_id = int(temp_list[0])
#     if temp_list[2] not in mpi_datatype:
#         temp_list[2] = r'0x6261c0'
    
#     size = 0
#     if func_id >= 3 and func_id <= 16:
#         size = int(temp_list[1]) / 2 * mpi_datatype[temp_list[2]][1] / 1024
#     elif func_id >= 17 and func_id <= 26:
#         size = int(temp_list[1]) * mpi_datatype[temp_list[2]][1] / 1024
#     elif func_id >= 27 and func_id <= 34:
#         size = int(temp_list[1]) * mpi_datatype[temp_list[2]][1] * (thread_num-1) / 1024
    
#     if size in message_dict:
#         message_dict[size] += 1
#     else:
#         message_dict[size] = 1

def count_thread_info(line, line_no, thread_no, temp_list, func_time_dict, func_call_dict):
    if len(temp_list) <= 1:
        return
    func_id = int(temp_list[0])
    func_id_key = str(temp_list[0])

    # obtain the elapsed time of each mpi funcion call
    if is_send_func(func_id) or is_bcast_scatter_or_gather_func(func_id) or is_allgather_func(func_id):
        time = float(temp_list[4]) / 1000000.0 # seconds
        if func_id_key in func_time_dict.keys():
            func_time_dict[func_id_key] += time
        else:
            func_time_dict[func_id_key] = time

    # obtain the transfering data vol of each mpi function call （to-do）

    # obtain the calling number of each mpi funcion call
    if func_id_key in func_call_dict.keys():
        func_call_dict[func_id_key] += 1
    else:
        func_call_dict[func_id_key] = 1


    # if temp_list[2] not in mpi_datatype:
    #     temp_list[2] = r'0x6261c0'
    
    # 计算每个进程的函数调用时间
#     time = -1
#     if (func_id >= 7 and func_id <= 14) \
#         or (func_id >= 17 and func_id <= 20)\
#         or (func_id >= 23 and func_id <= 24)\
#         or (func_id >= 27 and func_id <= 28)\
#         or (func_id >= 33 and func_id <= 34):
#         time = int(temp_list[-1]) / 1e6
#     if time != -1:
# #             func_name = mpi_func[str(func_id)]
#         if str(func_id) in func_time_dict:
#             func_time_dict[str(func_id)] += time
#         else:
#             func_time_dict[str(func_id)] = time
    
    
#         if func_id >= 3 and func_id <= 10:
#             thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
        
    # 计算每个进程的通信数据量
#     if func_id >= 3:
#         if func_id == 11:
#             if temp_list[5] not in mpi_datatype:
#                 temp_list[5] = r'0x6261c0'
#             data_vol = (int(temp_list[1]*mpi_datatype[temp_list[2]][1]) + int(temp_list[4])*mpi_datatype[temp_list[5]][1])
# #                 thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
#         elif func_id == 12:
#             data_vol = 2 * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
# #                 thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
#         elif func_id >= 19 and func_id <= 22:
#             if thread_no == int(temp_list[3]):
#                 data_vol = thread_num * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
# #                     for i in range(0, thread_num):
# #                         thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
#             else:
#                 data_vol = int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
#         elif func_id >= 23 and func_id <= 26:
#             if thread_no == int(temp_list[3]):
#                 data_vol = thread_num * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
# #                     for i in range(0, thread_num):
# #                         thread_send_list[i][thread_no] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
#             else:
#                 data_vol = int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
#         elif func_id >= 27 and func_id <= 30:
#             data_vol = 2 * thread_num * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
# #                 for i in range(0, thread_num):
# #                     thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
#         elif func_id >= 31 and func_id <= 34:
#             if temp_list[4] not in mpi_datatype:
#                 temp_list[4] = r'0x6261c0'
#             data_vol = thread_num * (int(temp_list[1]) * mpi_datatype[temp_list[2]][1] + int(temp_list[3])*mpi_datatype[temp_list[4]][1])
# #                 for i in range(0, thread_num):
# #                     thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
#         else:
#             data_vol = int(temp_list[1]) * mpi_datatype[temp_list[2]][1]

#         data_vol /= 1024
#         if str(func_id) in func_data_vol_dict:
#             func_data_vol_dict[str(func_id)] += data_vol
#         else:
#             func_data_vol_dict[str(func_id)] = data_vol
    # 计算每个进程的函数调用次数
    # if str(func_id) in func_call_dict:
    #     func_call_dict[str(func_id)] += 1
    # else:
    #     func_call_dict[str(func_id)] = 1

def sum_message(my_dict):
    sum = 0
    for i in my_dict.keys(): 
        sum = sum + my_dict[i] 
    return sum

def get_runtime_info(file_path, max_thread_num):
    thread_num = count_thread(file_path)

    message_dict = {}  
    thread_time_dict = {}
    # thread_data_vol_dict = {}
    thread_call_dict = {}
    thread_send_list = [[0 for i in range(thread_num)] for j in range(thread_num)]

    cur_thread_num = 0
    for file in os.listdir(file_path):

        # cur_thread_num += 1
        # if cur_thread_num > max_thread_num:
        #     thread_num = max_thread_num
        #     break

        with open(os.path.join(file_path, file), 'r', encoding = 'utf-8') as f:
            
            func_time_dict = {}
            # func_data_vol_dict = {}
            func_call_dict = {}
            
            line_no = 0
            threada_no = 0
            for line in f:
                line_no += 1

                line = line.strip('\n')
                if line_no == 1:
                    # thread_no = int(re.split('[::]', line)[1])
                    thread_no = int(line.split(':')[1])
                    print("reading output file from thread " + str(thread_no))
                    continue 

                temp_list = line.split(',')

                #******************* count_thread_send_recv and count_message *************************** 
                
                count_thread_sendrecv_and_messagesize(line, line_no, thread_no, temp_list, thread_send_list, message_dict)

                #******************* count_thread_info ************************************

                count_thread_info(line, line_no, thread_no, temp_list, func_time_dict, func_call_dict)

            thread_time_dict[thread_no] = func_time_dict
            # thread_data_vol_dict[thread_no] = func_data_vol_dict
            thread_call_dict[thread_no] = func_call_dict

    message_num = sum_message(message_dict)

    # return (thread_num, message_num, message_dict, thread_time_dict, thread_data_vol_dict, thread_call_dict, thread_send_list)
    return (thread_num, message_num, message_dict, thread_time_dict, thread_call_dict, thread_send_list)


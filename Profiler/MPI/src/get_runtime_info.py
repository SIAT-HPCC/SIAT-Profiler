import os
import time
import re
from mpi_info import mpi_datatype

def count_thread(DIR):
    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

def count_thread_send_recv(line, line_no, thread_no, temp_list, thread_send_list, thread_num):
    if len(temp_list) <= 1:
        return

    func_no = int(temp_list[0])
    if temp_list[2] not in mpi_datatype:
        temp_list[2] = r'0x6261c0'
        
    if func_no >= 3 and func_no <= 10:
        thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
        
    # 计算每个进程的通信数据量
    elif func_no == 11:
        if temp_list[5] not in mpi_datatype:
            temp_list[5] = r'0x6261c0'
        thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
    elif func_no == 12:
        thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
    elif func_no >= 19 and func_no <= 22:
        if thread_no == int(temp_list[3]):
            for i in range(0, thread_num):
                thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
    elif func_no >= 23 and func_no <= 26:
        if thread_no == int(temp_list[3]):
            for i in range(0, thread_num):
                thread_send_list[i][thread_no] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
    elif func_no >= 27 and func_no <= 30:
        for i in range(0, thread_num):
            thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
    elif func_no >= 31 and func_no <= 34:
        if temp_list[4] not in mpi_datatype:
            temp_list[4] = r'0x6261c0'
        for i in range(0, thread_num):
            thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]

def count_message(temp_list, message_dict, thread_num):            
    if len(temp_list) <= 1:
        return

    func_no = int(temp_list[0])
    if temp_list[2] not in mpi_datatype:
        temp_list[2] = r'0x6261c0'
    
    size = 0
    if func_no >= 3 and func_no <= 16:
        size = int(temp_list[1]) / 2 * mpi_datatype[temp_list[2]][1] / 1024
    elif func_no >= 17 and func_no <= 26:
        size = int(temp_list[1]) * mpi_datatype[temp_list[2]][1] / 1024
    elif func_no >= 27 and func_no <= 34:
        size = int(temp_list[1]) * mpi_datatype[temp_list[2]][1] * (thread_num-1) / 1024
    
    if size in message_dict:
        message_dict[size] += 1
    else:
        message_dict[size] = 1

def count_thread_info(line, line_no, thread_no, temp_list, thread_num, thread_time_dict, thread_data_vol_dict, thread_call_dict, func_time_dict, func_data_vol_dict, func_call_dict):
    if len(temp_list) <= 1:
        return
    func_no = int(temp_list[0])
    if temp_list[2] not in mpi_datatype:
        temp_list[2] = r'0x6261c0'
    
    # 计算每个进程的函数调用时间
    time = -1
    if (func_no >= 7 and func_no <= 14) \
        or (func_no >= 17 and func_no <= 20)\
        or (func_no >= 23 and func_no <= 24)\
        or (func_no >= 27 and func_no <= 28)\
        or (func_no >= 33 and func_no <= 34):
        time = int(temp_list[-1]) / 1e6
    if time != -1:
#             func_name = mpi_func[str(func_no)]
        if str(func_no) in func_time_dict:
            func_time_dict[str(func_no)] += time
        else:
            func_time_dict[str(func_no)] = time
    
    
#         if func_no >= 3 and func_no <= 10:
#             thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
        
    # 计算每个进程的通信数据量
    if func_no >= 3:
        if func_no == 11:
            if temp_list[5] not in mpi_datatype:
                temp_list[5] = r'0x6261c0'
            data_vol = (int(temp_list[1]*mpi_datatype[temp_list[2]][1]) + int(temp_list[4])*mpi_datatype[temp_list[5]][1])
#                 thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
        elif func_no == 12:
            data_vol = 2 * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
#                 thread_send_list[thread_no][int(temp_list[3])] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
        elif func_no >= 19 and func_no <= 22:
            if thread_no == int(temp_list[3]):
                data_vol = thread_num * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
#                     for i in range(0, thread_num):
#                         thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
            else:
                data_vol = int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
        elif func_no >= 23 and func_no <= 26:
            if thread_no == int(temp_list[3]):
                data_vol = thread_num * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
#                     for i in range(0, thread_num):
#                         thread_send_list[i][thread_no] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
            else:
                data_vol = int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
        elif func_no >= 27 and func_no <= 30:
            data_vol = 2 * thread_num * int(temp_list[1]) * mpi_datatype[temp_list[2]][1]
#                 for i in range(0, thread_num):
#                     thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
        elif func_no >= 31 and func_no <= 34:
            if temp_list[4] not in mpi_datatype:
                temp_list[4] = r'0x6261c0'
            data_vol = thread_num * (int(temp_list[1]) * mpi_datatype[temp_list[2]][1] + int(temp_list[3])*mpi_datatype[temp_list[4]][1])
#                 for i in range(0, thread_num):
#                     thread_send_list[thread_no][i] += int(temp_list[1])*mpi_datatype[temp_list[2]][1]
        else:
            data_vol = int(temp_list[1]) * mpi_datatype[temp_list[2]][1]

        data_vol /= 1024
        if str(func_no) in func_data_vol_dict:
            func_data_vol_dict[str(func_no)] += data_vol
        else:
            func_data_vol_dict[str(func_no)] = data_vol
    # 计算每个进程的函数调用次数
    if str(func_no) in func_call_dict:
        func_call_dict[str(func_no)] += 1
    else:
        func_call_dict[str(func_no)] = 1

def sum_message(my_dict):
    sum = 0
    for i in my_dict: 
        sum = sum + my_dict[i] 
    return sum

def get_runtime_info(lammps_mpi_result_path, max_thread_num):
    thread_num = count_thread(lammps_mpi_result_path)

    message_dict = {}  
    thread_time_dict = {}
    thread_data_vol_dict = {}
    thread_call_dict = {}
    thread_send_list = [[0 for i in range(thread_num)] for j in range(thread_num)]

    cur_thread_num = 0
    for file in os.listdir(lammps_mpi_result_path):

        cur_thread_num += 1
        if cur_thread_num > max_thread_num:
            thread_num = max_thread_num
            break

        with open(os.path.join(lammps_mpi_result_path, file), 'r', encoding = 'utf-8') as f:
            
            func_time_dict = {}
            func_data_vol_dict = {}
            func_call_dict = {}
            
            line_no = 0
            threada_no = 0
            for line in f:
                line_no += 1

                line = line.strip('\n')
                if line_no == 1:
                    # thread_no = int(re.split('[::]', line)[1])
                    thread_no = int(line.split(':')[1])
                    continue 

                temp_list = line.split(',')

                #******************* count_thread_send_recv ************************************ 
                
                count_thread_send_recv(line, line_no, thread_no, temp_list, thread_send_list, thread_num)

                #******************* count_message ************************************
                
                count_message(temp_list, message_dict, thread_num)

                #******************* count_thread_info ************************************

                count_thread_info(line, line_no, thread_no, temp_list, thread_num, thread_time_dict, thread_data_vol_dict, thread_call_dict, func_time_dict, func_data_vol_dict, func_call_dict)

            thread_time_dict[thread_no] = func_time_dict
            thread_data_vol_dict[thread_no] = func_data_vol_dict
            thread_call_dict[thread_no] = func_call_dict

    message_num = sum_message(message_dict)

    return (thread_num, message_num, message_dict, thread_time_dict, thread_data_vol_dict, thread_call_dict, thread_send_list)


import communication_load as cl
import communication_pattern as cp
import get_runtime_info as gri
import os
import time
import pandas as pd
import argparse

'''
#/********************** draw bar ********************/ 
def draw_communication_load(input_dir, output_dir):
    message_dict = {}
    #input_dir = "../data/lammps_mpi_result/"
    thread_num = cl.count_thread(input_dir)
    # print(thread_num)
    max_thread_num = 16
    files = os.listdir(input_dir)
    cur_thread_num = 0
    for file in files:
        cur_thread_num += 1
        if cur_thread_num > max_thread_num:
            thread_num = max_thread_num
            break
        # filename = input_dir + "/" + file
        filename = os.path.join(input_dir, file)
        with open(filename, 'r', encoding = 'utf-8') as f:
            cl.count_message(f, message_dict, thread_num)

    message_num = cl.sum_message(message_dict)        
    # print(message_num)
    cl.draw_bar(os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "count-message.png"), message_dict)  
#/********************** draw bar ********************/   

#************************ communication pattern *****************
def draw_communication_pattern(input_dir, output_dir):
    #input_dir = "../data/cdhit_mpi_result/"
    thread_num = cp.count_thread(input_dir)
    real_thread_num = thread_num
    max_thread_num = 16
    # print(thread_num)
    thread_send_list = [[0 for i in range(thread_num)] for j in range(thread_num)]
    thread_time_dict = {}
    thread_data_vol_dict = {}
    thread_call_dict = {}
    # thread_send_list = [[0 for i in range(32)] for j in range(32)]
    files = os.listdir(input_dir)
    cur_thread_num = 0
    for file in files:
        cur_thread_num += 1
        if cur_thread_num > max_thread_num:
            thread_num = max_thread_num
            break

        # filename = input_dir + "/" + file
        filename = os.path.join(input_dir, file)
        with open(filename, 'r', encoding = 'utf-8') as f:
            cp.count_thread_info(thread_time_dict, thread_data_vol_dict, thread_call_dict, f, thread_num)
            
    filenum = 0
    for file in files:
        # filename = input_dir + "/" + file
        filename = os.path.join(input_dir, file)
    #     print(filename)
        with open(filename, 'r', encoding = 'utf-8') as f:
            cp.count_thread_send_recv(f, thread_send_list, real_thread_num)


    # print(thread_time_dict)
    # print(thread_send_list)

    app_name = "cdhit"
    # print(real_thread_num)

    cp.draw_bar3d_thread_func_time(thread_time_dict, thread_num, os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "elapsed time of MPI function.png"))

    # draw_bar3d_thread_func_data_vol(thread_data_vol_dict, thread_num, app_name + "Statistic of data transfer volume of MPI function")

    cp.draw_bar3d_thread_func_call(thread_call_dict, thread_num, os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "calling times of MPI function.png"))

    # draw_bar3d_thread_send(thread_send_list, thr4ead_num)

    pt = pd.DataFrame(thread_send_list)
    cp.draw_heat_map_thread_send(pt, real_thread_num, os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "communication_pattern.png")) 
#************************ communication pattern *****************
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input parameters for input dir, output dir ')
    parser.add_argument("-i", "--input_dir", dest='dir_include_data', type=str, help='The filepath of folder/directory includes performance data.')
    parser.add_argument("-o", "--output_dir", dest='output_dir', type=str, help='The output directory to save the profiling result.')
    args = parser.parse_args()
    input_dir = args.dir_include_data
    output_dir = args.output_dir
    max_thread_num = 6400
    thread_num, message_num, message_dict, thread_time_dict, thread_data_vol_dict, \
                                                thread_call_dict, thread_send_list = gri.get_runtime_info(input_dir, max_thread_num) 

    cl.draw_bar(os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "count-message.png"), message_dict)
    cp.draw_bar3d_thread_func_time(thread_time_dict, thread_num, os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "elapsed time of MPI function.png"))
    cp.draw_bar3d_thread_func_call(thread_call_dict, thread_num, os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "calling times of MPI function.png"))
    pt = pd.DataFrame(thread_send_list)
    cp.draw_heat_map_thread_send(pt, thread_num, os.path.join(output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime())) + "communication_pattern.png")) 

    # draw_communication_load(args.dir_include_data, args.output_dir)
    # draw_communication_pattern(args.dir_include_data, args.output_dir)
    

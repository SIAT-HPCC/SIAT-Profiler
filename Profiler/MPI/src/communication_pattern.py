import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms
import re
import seaborn as sns
import pandas as pd

from mpi_info import mpi_datatype
from mpi_info import mpi_func

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

thread_time_dict = {}
thread_data_vol_dict = {}
thread_call_dict = {}
# thread_send_recv_list = []

def count_thread(DIR):
    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

def count_thread_info(thread_time_dict, thread_data_vol_dict, thread_call_dict, file, thread_num):
    func_time_dict = {}
    func_data_vol_dict = {}
    func_call_dict = {}
    line_no = 0
    thread_no = 0
    for line in file:
        line_no += 1
        line=line.strip('\n')
        # print(line)
        if line_no == 1:
            # thread_no = int(re.split('[::]', line)[1])
            thread_no = int(line.split(':')[1])
            continue
        temp_list = line.split(',')
        # print(line)
        if len(temp_list) <= 1:
            continue
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
        
        
#         if func_no >= 3 and func_no <= 10:
#             send_recv_dict['send'] += int(temp_list[1])
#         elif func_no >= 13 and func_no <= 16:
#             send_recv_dict['recv'] += int(temp_liste[1])
#         elif func_noc == 11:
#             send_recv_dict['send'] += int(temp_list[1])
#             send_recv_dict['recv'] += int(temp_liste[4])
        
    
    thread_time_dict[thread_no] = func_time_dict
    thread_data_vol_dict[thread_no] = func_data_vol_dict
    thread_call_dict[thread_no] = func_call_dict
#     thread_send_recv_list.append(send_recv_dict)
#     print(send_recv_dictend_recv_dic)

def count_thread_send_recv(file, thread_send_list, thread_num):
    line_no = 0
    thread_no = 0
    for line in file:
#         print(line_no)
        line_no += 1
        line=line.strip('\n')
        if line_no == 1:
            # thread_no = int(re.split('[::]', line)[1])
            thread_no = int(line.split(':')[1])
            continue
        temp_list = line.split(',')
#         print(line)
        if len(temp_list) <= 1:
            continue
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

def draw_bar3d_thread_func_data_vol(thread_data_vol_dict, thread_num, filename):

    x = []
    _x = range(0, thread_num)
    y = []
    _y = range(0, len(mpi_func))
    z = []


    for func_no in mpi_func.keys():
        for thread in range(0, thread_num):
            if thread in thread_data_vol_dict and func_no in thread_data_vol_dict[thread]:
                x.append(thread)
                y.append(int(func_no))
                z.append(thread_data_vol_dict[thread][func_no])
    
    if len(z) == 0:
        print("all function time is zero!")
        return
    
    fig = plt.figure(figsize=(15,15), dpi=200)
    ax = fig.gca(projection='3d')
    
    # 设置坐标轴的范围
    plt.ylim(0, len(mpi_func))
    plt.xlim(0, thread_num)
    
    
    # 拉伸各方向的比例
    x_scale=0.5
    y_scale=2
    z_scale=0.3
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([x_scale, y_scale, z_scale, 1]))

    ax.bar3d(x, y, np.zeros_like(z), dx=0.3, dy=0.3, dz=z)
    plt.yticks(_y, [x for x in mpi_func.values()], ha='left', rotation=-15)
    plt.xticks(_x, [str(x) for x in range(0, thread_num)])
    plt.title("MPI各进程调用函数的数据通信量")
    ax.set_xlabel('Rank')
    ax.set_zlabel('Data Vol(KB)', labelpad=20)
    plt.savefig(filename,dpi=600)
    plt.show()
    

def draw_bar3d_thread_func_call(thread_call_dict, thread_num, filename):

    x = []
    _x = range(0, thread_num)
    y = []
    _y = range(0, len(mpi_func))
    z = []


    for func_no in mpi_func.keys():
        for thread in range(0, thread_num):
            if thread in thread_call_dict and func_no in thread_call_dict[thread]:
                x.append(thread)
                y.append(int(func_no))
                z.append(thread_call_dict[thread][func_no])
    
    if len(z) == 0:
        print("all function time is zero!")
        return
    
    fig = plt.figure(figsize=(15,15), dpi=200)
    ax = fig.gca(projection='3d')
    
    # 设置坐标轴的范围
    plt.ylim(0, len(mpi_func))
    plt.xlim(0, thread_num)
    
    # 拉伸各方向的比例
    x_scale=0.5
    y_scale=2
    z_scale=0.3
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([x_scale, y_scale, z_scale, 1]))

    ax.bar3d(x, y, np.zeros_like(z), dx=0.3, dy=0.3, dz=z)
    plt.yticks(_y, [x for x in mpi_func.values()], ha='left', rotation=-15)
    plt.xticks(_x, [str(x) for x in range(0, thread_num)])
    ax.set_xlabel('Rank')
    ax.set_zlabel('Call Times', labelpad=20)    
    plt.title("The number of times the MPI process calls MPI functions")
    plt.savefig(filename,dpi=600)
    plt.show()
    
def draw_bar3d_thread_func_time(thread_time_dict, thread_num, filename):
#     _x = []
#     _y = []
#     z = []
#     print(thread_time_dict)
#     for i in range(0, thread_num):
#         _x.append(i)
#     for i in range(1, 36):
#         _y.append(i)
#     _xx, _yy = np.meshgrid(_x, _y)
#     x, y = _xx.ravel(), _yy.ravel()

    x = []
    _x = range(0, thread_num)
    y = []
    _y = range(0, len(mpi_func))
    z = []

#     for func_no in mpi_func.keys():
#         for thread in range(0, thread_num):
#             if thread in thread_time_dict and func_no in thread_time_dict[thread]:
#                 z.append(thread_time_dict[thread][func_no])
#             else:
#                 z.append(0)
    for func_no in mpi_func.keys():
        for thread in range(0, thread_num):
            if thread in thread_time_dict and func_no in thread_time_dict[thread]:
                x.append(thread)
                y.append(int(func_no))
                z.append(thread_time_dict[thread][func_no])
    
    if len(z) == 0:
        print("all function time is zero!")
        return
    
    fig = plt.figure(figsize=(15,15), dpi=200)
    ax = fig.gca(projection='3d')
    
    # 设置坐标轴的范围
    plt.ylim(0, len(mpi_func))
    plt.xlim(0, thread_num)
    
    
    # 拉伸各方向的比例
    x_scale=0.5
    y_scale=2
    z_scale=0.3
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([x_scale, y_scale, z_scale, 1]))
    
# 移动x轴刻度的代码，不管用    
#     ax.yaxis._axinfo['label']['space_factor'] = 2.8
    
#     ax.tick_params(axis='both', width=10, labelsize=10, pad=5)
#     import matplotlib.transforms as mtrans
#     # ...
#     trans = mtrans.Affine2D().translate(20, 0)
#     for t in ax.get_yticklabels():
#         print(t)
#         t.set_transform(t.get_transform()+trans)
#         # Create offset transform by 5 points in x direction
#     dx = 0/72.; dy = 50/72. 
#     offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
#     # apply offset transform to all x ticklabels.
#     for label in ax.yaxis.get_majorticklabels():
#         label.set_transform(label.get_transform() + offset)
    
#     labels = ax.set_yticklabels([x for x in mpi_func.values()])
#     for i, label in enumerate(labels):
#         label.set_x(label.get_position()[1] - (i % 2) * 0.075)
#     print(_y)

    ax.bar3d(x, y, np.zeros_like(z), dx=0.3, dy=0.3, dz=z)
    plt.yticks(_y, [x for x in mpi_func.values()], ha='left', rotation=-15)
    plt.xticks(_x, [str(x) for x in range(0, thread_num)])
    ax.set_xlabel('Rank')
    ax.set_zlabel('Time(seconds)', labelpad=20)
    plt.title("The elapsed time of MPI functions called by MPI process")
    plt.savefig(filename,dpi=600)
    plt.show()
    
def draw_bar3d_thread_send(thread_send_list, thread_num):
#     _x = [x for x in range(0, thread_num)]
#     _y = [x for x in range(0, thread_num)]
#     z = []
    
    _x = range(0, thread_num)
    _y = range(0, thread_num)
    x = []
    y = []
    z = []

#     for t in thread_send_list:
#         for i in t:
#             z.append(i)
#     _xx, _yy = np.meshgrid(_x, _y)
#     x, y = _xx.ravel(), _yy.ravel()

    for i in range(0, thread_num):
        for j in range(0, thread_num):
            if thread_send_list[i][j] > 0:
                x.append(j)
                y.append(i)
                # print(type(thread_send_list[i][j]))
                z.append(thread_send_list[i][j])

    if len(z) == 0:
        print("all send data is zero!")
        return

    fig = plt.figure(figsize=(10,12), dpi=200)
    ax = fig.gca(projection='3d')
    
    
    # 设置坐标轴的范围
    plt.ylim(0, thread_num)
    plt.xlim(0, thread_num)
    
    ax.bar3d(x, y, np.zeros_like(z), dx=1, dy=1, dz=z, alpha=0.7)
    plt.xticks(_x, [str(x) for x in range(0, thread_num)])
    plt.yticks(_y, [str(x) for x in range(0, thread_num)], ha='left', rotation=-15)
    plt.xlabel('Send Rank', labelpad=10)
    plt.ylabel('Recv Rank', labelpad=10)
#     plt.savefig('filename1.png',dpi=600)
    plt.show()
    
def draw_heat_map_thread_send(pt, thread_num, output_path):
    fig, ax = plt.subplots(figsize = (10,8))
    # print(thread_send_list)
    
    res = pt
    # print(type(res))
    # 如果MPI进程数大于32，则限制只输出32个MPI进程的结果
    # if thread_num > 32:
    #     res = pt.iloc[0:32, 0:32]
    # cmap用matplotlib colormap
    print(res)
    max_vol = max(res)
    min_vol = min(res)
    # sns.heatmap(res, linewidths = 0.05, ax = ax, cmap='rainbow') 
    sns.heatmap(res, ax = ax, cmap='rainbow') 
    # rainbow为 matplotlib 的colormap名称
    plt.title('The amount of data transferred between MPI processes')
    ax.set_xlabel('Recv Rank')
    ax.set_ylabel('Send Rank')
    # ax.set_ylim([32, 0])
    ax.invert_yaxis()
    plt.savefig(output_path, dpi=200)
    # plt.show()

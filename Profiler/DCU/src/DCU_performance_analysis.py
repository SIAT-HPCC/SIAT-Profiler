# 完整版本的脚本
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

# plt 中文不乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 性能数据分析类
class Performance_data_analysis():
    def read_performance_data(self, dir):
        data = pd.DataFrame()
        chunksize = 1e6    #这个数字设置多少有待考察
        for chunk in pd.read_csv(dir, chunksize=chunksize):
            data = data.append(chunk)
        # 删去中间几列无意义的属性
        data.drop(labels=data.columns[2:14], axis=1, inplace=True)
        print("read performance data successfully!")
        return data

    # 归一化
    def normalize_data(self, data):
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data_norm

    # 绘制雷达图
    def draw_radar_map(self, data, pic_name):
        print("绘制雷达图中......")
        for i in range(0, len(data)):
            # 构造数据
            features = ['compute ability', 'memory read', 'memory write', 'L2cache hit rate', 'bandwidth usage']
            values = list()
            for f in features:
                values.append(data.iloc[i][f])
            
            feature_labels = ['计算能力', '显存读', '显存写', 'L2缓存命中率', '显存带宽利用率']
                    
            N = len(feature_labels)
            # 设置雷达图的角度，用于平分切开一个圆面
            angles=np.linspace(0, 2*np.pi, N, endpoint=False)

            # 为了使雷达图一圈封闭起来，需要下面的步骤
            values=np.concatenate((values,[values[0]]))
            angles=np.concatenate((angles,[angles[0]]))
            
            # 使用ggplot的绘图风格
            plt.style.use('ggplot')
            # 绘图
            fig=plt.figure()
            # 这里一定要设置为极坐标格式
            ax = fig.add_subplot(111, polar=True)
            # 绘制折线图
            ax.plot(angles, values, 'o--', linewidth=2)
            # 填充颜色
            ax.fill(angles, values, alpha=0.25)
            # 添加每个特征的标签
            ax.set_thetagrids(angles * 180/np.pi, feature_labels)
            # 设置雷达图的范围
            ax.set_ylim(0,100)
            # 标题
            plt.title(data.iloc[i]['KernelName'], pad=25)
            
            # 添加网格线
            ax.grid(True)
            # # 显示图形
            # plt.show()
            # 保存图片
            plt.savefig("../result/" + pic_name, dpi=200)
            print("绘制成功，输出结果已保存在: " + "../result/" + pic_name)
        

    def draw_bar_pic(self, data_x, data_y, title, y_label, pic_name):
        plt.ylabel(y_label)
        x_ticks_list = ['kernel' + str(x) for x in range(1, len(data_x)+1)]
        plt.bar(data_x, data_y, width=0.35)
        plt.xticks(data_x, x_ticks_list)
        plt.title(title)
        text_height_margin = max(data_y)/50
        # 加入具体数值的标签
        for x, y in enumerate(data_y):
            plt.text(x, y + text_height_margin, '%s' % int(round(data_y[x])), verticalalignment="bottom",horizontalalignment="center")
        # plt.show()
        plt.savefig("../result/" + pic_name, dpi=200)

    # 绘制热点函数图    
    def draw_bar_pic_two_bars(self, data_x, data_y1, data_y2, label1, label2, title, y_label, pic_name):
        print("绘制热点函数图中......")
        plt.figure(figsize=(20,10))
        plt.rcParams['figure.dpi'] = 100
        plt.ylabel(y_label)
        x_ticks_list = ['kernel' + str(x) for x in range(1, len(data_x)+1)]
        bar_width = 0.4
        x_loc = np.arange(len(data_x))#柱状图在横坐标上的位置
        plt.bar(x_loc, data_y1, label=label1, color='steelblue', width=bar_width)
    #     plt.bar(data_x, data_y2, bottom=data_y1, label=label2, color='indianred', width=0.8)
        
        plt.bar(x_loc + bar_width, height=data_y2, label=label2, color='indianred', width=bar_width)
        
        text_height_margin = max(data_y1)/50
        # 加入具体数值的标签
        for x, y in enumerate(data_y1):
            plt.text(x, y + text_height_margin, '%s' % int(round(data_y1[x])), verticalalignment="bottom",horizontalalignment="center")
            
        text_height_margin = max(data_y2)/50
        for x, y in enumerate(data_y2):
            plt.text(x + bar_width, y + text_height_margin, '%s' % int(round(data_y2[x])), verticalalignment="bottom",horizontalalignment="center")
        
        plt.xticks(x_loc + bar_width / 2, x_ticks_list)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
        plt.legend()
        plt.title(title)
        # plt.show()
        plt.savefig("../result/" + pic_name, dpi=200)
        print("绘制成功，输出结果已保存在: " + "../result/" + pic_name)

    def process_data(self, data):
        # 计算每次的运行时间
    #     data['elapsed_time'] = data['CompleteNs'] - data['DispatchNs']
        data['compute ability'] = (data['SQ_INSTS_SALU'] + data['SQ_INSTS_VALU']) / (data['GRBM_COUNT'] / 1300000*1000) / 842
    #     print(max(data['compute ability']))
        data['memory read']     = (data['SQ_INSTS_VMEM_RD'] + data['SQ_INSTS_SMEM'] - data['TA_FLAT_READ_WAVEFRONTS_sum']) / 100 / 41943 * 100
    #     print(max(data['momory read']))
        data['memory write']    = (data['SQ_INSTS_VMEM_WR'] - data['TA_FLAT_WRITE_WAVEFRONTS_sum']) / 6000
    #     print(max(data['memory write']))
        data['L2cache hit rate']= data['TCC_HIT_sum'] / (data['TCC_HIT_sum'] + data['TCC_MISS_sum']) * 100
    #     print(max(data['L2cache hit rate']))
        data['bandwidth usage'] = (data['TCC_EA_WRREQ_sum'] + data['TCC_EA_RDREQ_sum']) / (data['GRBM_COUNT'] / 1300000*1000) / 1000 / 5.5 * 100
    #     print(max(data['bandwidth usage']))
        return data

    def groupby_and_mean_data(self, data):
        # 计算每种核函数出现的频率（frequency）
        data_groupby_kernel = data.groupby('KernelName')
        # 根据核函数名字来分组，并求平均值和次数
        data_groupby_kernel_mean = data.groupby('KernelName').mean()
        data_groupby_kernel_sum = data.groupby('KernelName').sum()
        # 每种核函数的名字
        data_groupby_kernel_mean['KernelName'] = data_groupby_kernel.size().index
        # 每种核函数出现的频率
        data_groupby_kernel_mean['Count'] = data_groupby_kernel.size()
        kernel_total_num = sum(data_groupby_kernel.size())
        data_groupby_kernel_mean['Frequency'] = data_groupby_kernel_mean['Count'] / kernel_total_num * 100
        sum_of_time = sum(data['GRBM_COUNT'])
        data_groupby_kernel_mean['percentage of total time'] = data_groupby_kernel_sum['GRBM_COUNT'] / sum_of_time * 100

        return data_groupby_kernel_mean

    def calculate_radar_scores(self, data_groupby_kernel_mean_sorted):
        total_score = 0
        for i in range(0, len(data_groupby_kernel_mean_sorted)):
            score = 0
            kernel_name = data_groupby_kernel_mean_sorted.iloc[i]['KernelName']
            score += data_groupby_kernel_mean_sorted.iloc[i]['compute ability']
            score += data_groupby_kernel_mean_sorted.iloc[i]['memory read']
            score += data_groupby_kernel_mean_sorted.iloc[i]['memory write']
            score += data_groupby_kernel_mean_sorted.iloc[i]['L2cache hit rate']
            score += data_groupby_kernel_mean_sorted.iloc[i]['bandwidth usage']
            score /= 500
            print(str(kernel_name) + "的雷达分：" + str(score))
            total_score += score * (data_groupby_kernel_mean_sorted.iloc[i]['percentage of total time'] / 100)
        
        print("程序的总雷达分：" + str(total_score))

    def visualize_data(self, file_path):
        filename = file_path.split('/')[-1]
        if filename[-4:] != '.csv':
            print("error! the file is not a csv file!")
            sys.exit()
        filename = filename[:-4]
        # 分别读入三个文件中的指标
        data = read_performance_data(file_path)
        # 处理数据
        data = process_data(data)
        # 对数据分组，并且求均值
        data_groupby_kernel_mean = groupby_and_mean_data(data)
        data_groupby_kernel_mean_sorted = data_groupby_kernel_mean.sort_values('percentage of total time', inplace=False, ascending=False)

        current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        pic_name = current_time + "_" + filename
        # 绘制柱状图
        draw_bar_pic_two_bars(data_groupby_kernel_mean['KernelName'], \
                            data_groupby_kernel_mean['Frequency'], \
                            data_groupby_kernel_mean['percentage of total time'], \
                            '调用频率', '占总运行时间百分比', "各核函数调用频率及运行时间占总时间比例统计图", "百分比(%)", pic_name + "_hotbar.png")
        # 计算雷达分
        calculate_radar_scores(data_groupby_kernel_mean_sorted)
        # 绘制雷达图
        draw_radar_map(data_groupby_kernel_mean, pic_name +  "_radarmap.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input file information')
    parser.add_argument("-i", "--input_file", dest='input_file', type=str, help='The csv file includes performance counter result collected by Rocprof.')
    args = parser.parse_args()
    performance_analysis = Performance_data_analysis()
    performance_analysis.visualize_data(args.input_file)
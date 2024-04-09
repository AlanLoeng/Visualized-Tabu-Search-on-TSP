import random
import copy
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import sys

class TSP:
    def __init__(self, num_cities, initial_tabu_size=100, max_iter=250, num_runs=10, adjustment_interval=500, max_tabu_tenure=200, tabu_size_step=10, max_tabu_tenure_step=10, min_tabu_tenure=10 ,refresh_rate=200):
        self.num_cities = num_cities  #城市数量
        self.initial_tabu_size = initial_tabu_size  #禁忌表初始大小
        self.max_iter = max_iter  #每个初始解的迭代次数
        self.tabu_list = []  #禁忌表
        self.num_runs = num_runs  #初始解个数
        self.adjustment_interval = adjustment_interval
        self.best_distance_history = []  # 记录最佳距离历史
        self.tabu_size_history = []  # 记录禁忌长度历史
        self.max_tabu_tenure_history = []  # 记录禁忌期限历史
        self.coordinates = []
        # 初始化 neighbor_occurrence_count 字典
        self.neighbor_occurrence_count = {((i, j)): 0 for i in range(num_cities) for j in range(num_cities) if i != j}
        self.max_tabu_tenure = max_tabu_tenure  #禁忌表元素的存在期限
        self.tabu_size_step = tabu_size_step  #禁忌表大小
        self.max_tabu_tenure_step = max_tabu_tenure_step  #禁忌表元素的存在期限的调整步进
        self.min_tabu_tenure = min_tabu_tenure  #禁忌表元素的存在期限的调整下限
        self.refresh_rate = refresh_rate #刷新率

    def generate_random_coordinates(self):
        # 生成随机城市坐标
        # random.seed(18931226)
        return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(self.num_cities)]

    def calculate_distance(self, city1, city2):
        # 计算两个城市之间的距离
        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

    def create_distance_matrix(self, coordinates):
        # 创建基于坐标的距离矩阵
        distance_matrix = [[0] * self.num_cities for _ in range(self.num_cities)]
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                distance = self.calculate_distance(coordinates[i], coordinates[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        return distance_matrix

    def initial_solution(self):
        # 生成初始解
        start_city = random.randint(0, self.num_cities - 1)
        unvisited_cities = list(range(self.num_cities))
        unvisited_cities.remove(start_city)
        current_city = start_city
        solution = [start_city]

        while unvisited_cities:
            nearest_city = min(unvisited_cities,
                               key=lambda city: self.calculate_distance(self.coordinates[current_city], self.coordinates[city]))
            solution.append(nearest_city)
            current_city = nearest_city
            unvisited_cities.remove(nearest_city)

        return solution

    def calculate_total_distance(self, solution):
        # 计算解的总距离
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distance_matrix[solution[i]][solution[i + 1]]
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # 回到起始城市
        return total_distance

    def apply_move(self, solution, move):
        # 应用移动操作到解
        new_solution = copy.deepcopy(solution)
        new_solution[move[0]], new_solution[move[1]] = new_solution[move[1]], new_solution[move[0]]
        return new_solution

    def generate_neighbor(self, solution):
        # 生成邻居解，通过维护一个邻居出现次数表来选出被搜索的最少的邻居解，让搜搜尽量均匀
        least_occurred_neighbor = min(self.neighbor_occurrence_count.items(), key=lambda x: x[1])
        neighbor_move, occurrence_count = least_occurred_neighbor
        # 更新邻居的出现次数
        self.neighbor_occurrence_count[neighbor_move] = occurrence_count + 1
        return neighbor_move, occurrence_count

        # 禁忌搜索
    def tabu_search(self):
        # 运行禁忌搜索算法
        best_solution = None
        best_distance = float('inf')
        current_tabu_size = self.initial_tabu_size #初始化当前禁忌表大小为初始值
        max_tabu_tenure = self.max_tabu_tenure  # 初始的禁忌元素存在期限
        iteration_counter = 0
        print("最优解生成记录：")
        refreshed = False  # 刷新标记，防止同一次迭代中重复刷新
        tsp_visualization = TSPVisualization(self.coordinates, self.initial_solution(), self.initial_solution())  #负责数据可视化的对象
    
        # 循环执行多次禁忌搜索，以便在多次运行中找到全局最优解
        for _ in range(self.num_runs):
            # 产生一个初始解
            current_solution = self.initial_solution()
            current_distance = self.calculate_total_distance(current_solution)
    
            # 迭代搜索
            for iteration in range(self.max_iter):
                refreshed = False
                # 生成邻居解
                neighbor_move, occurrence_count = self.generate_neighbor(current_solution)
    
                # 确保邻居解不在禁忌列表中
                while neighbor_move in self.tabu_list:
                    neighbor_move, occurrence_count = self.generate_neighbor(current_solution)
    
                neighbor_solution = self.apply_move(current_solution, neighbor_move)
                neighbor_distance = self.calculate_total_distance(neighbor_solution)
    
                # 更新最优解
                if neighbor_distance < best_distance:
                    best_solution = neighbor_solution
                    best_distance = neighbor_distance
                    print(best_solution)
                    if not refreshed:
                        # 找到最优解时实时更新TSP解的图像
                        tsp_visualization.update_tsp_solution(best_solution, current_solution)
                        refreshed = True
    
                # 调整禁忌长度和期限
                if iteration_counter % self.adjustment_interval == 0:
                    current_tabu_size, max_tabu_tenure = self.adjust_tabu_size(current_tabu_size, best_distance,
                                                                               current_distance, max_tabu_tenure)
                    self.tabu_size_history.append(current_tabu_size)
                    self.max_tabu_tenure_history.append(max_tabu_tenure)
    
                # 添加禁忌移动，初始期限为0
                self.tabu_list.append((neighbor_move, 0))
    
                # 更新剔除禁忌表中的超期元素
                self.tabu_list = [(move, tenure + 1) for move, tenure in self.tabu_list if tenure < max_tabu_tenure]
    
                # 达到刷新周期时实时更新TSP解的图像
                if iteration_counter % self.refresh_rate == 0 and iteration_counter != 0 and not refreshed:
                    tsp_visualization.update_tsp_solution(best_solution, current_solution)
                    refreshed = True
    
                current_solution = neighbor_solution
                current_distance = neighbor_distance
    
                iteration_counter += 1
                self.best_distance_history.append(best_distance)  # 记录最佳距离
    
        # 显示最终结果
        tsp_visualization.plot_tabu_size_history(self.tabu_size_history)
        tsp_visualization.plot_max_tabu_tenure_history(self.max_tabu_tenure_history)
        tsp_visualization.plot_distance_history(self.best_distance_history)
        tsp_visualization.update_tsp_solution(best_solution, current_solution)  # 更新TSP解的图像
        return best_solution, best_distance


            
    def adjust_tabu_size(self, current_tabu_size, best_distance, current_distance, current_max_tabu_tenure):
        # 动态调整禁忌表长度和禁忌表元素的存在期限
        if current_distance > best_distance:
            new_tabu_size = min(current_tabu_size + self.tabu_size_step,
                               self.num_cities * (self.num_cities - 1) / 2)
            new_max_tabu_tenure = max(current_max_tabu_tenure - self.max_tabu_tenure_step, self.min_tabu_tenure)
        else:
            new_tabu_size = max(current_tabu_size - self.tabu_size_step, 1)
            new_max_tabu_tenure = min(current_max_tabu_tenure + self.max_tabu_tenure_step, 4000)

        return new_tabu_size, new_max_tabu_tenure

#可视化TSP和禁忌搜搜的数据变化
class TSPVisualization:
    def __init__(self, coordinates, best_solution, current_solution):
        self.coordinates = coordinates
        self.best_solution = best_solution
        self.current_solution = current_solution

        plt.rcParams['font.sans-serif'] = ['SimHei']
        self.fig_solution, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        plt.show(block=False)

    def plot_tsp_solution(self):
        # 绘制best_solution
        plt.sca(self.fig_solution.axes[0])
        plt.cla()
        x_best = [self.coordinates[i][0] for i in self.best_solution]
        y_best = [self.coordinates[i][1] for i in self.best_solution]
        x_best.append(self.coordinates[self.best_solution[0]][0])
        y_best.append(self.coordinates[self.best_solution[0]][1])
        plt.plot(x_best, y_best, marker='o', linestyle='-', color='b')
        plt.title('最优解')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')

        # 绘制current_solution
        plt.sca(self.fig_solution.axes[1])
        plt.cla()
        x_current = [self.coordinates[i][0] for i in self.current_solution]
        y_current = [self.coordinates[i][1] for i in self.current_solution]
        x_current.append(self.coordinates[self.current_solution[0]][0])
        y_current.append(self.coordinates[self.current_solution[0]][1])
        plt.plot(x_current, y_current, marker='o', linestyle='-', color='r')
        plt.title('当前解')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')

        plt.pause(0.0001)
       
        
    def plot_distance_history(self, best_distance_history):
        self.fig_distance_history = plt.figure(figsize=(8, 6))
        # 绘制最佳距离历史
        plt.plot(best_distance_history, marker='o', linestyle='-', color='r')
        plt.title('最优解的历史')
        plt.xlabel('迭代次数')
        plt.ylabel('最佳距离')
        plt.pause(0.0001)
        plt.show(block=False)

    def plot_tabu_size_history(self, tabu_size_history):
        self.fig_tabu_size_history = plt.figure(figsize=(8, 6))
        # 绘制禁忌表长度的历史
        plt.plot(tabu_size_history, marker='o', linestyle='-', color='g')
        plt.title('禁忌表长度的历史')
        plt.xlabel('调整次数')
        plt.ylabel('禁忌表长度')
        plt.pause(0.0001)
        plt.show(block=False)

    def plot_max_tabu_tenure_history(self, max_tabu_tenure_history):
        # 绘制禁忌表元素的存在期限的历史
        self.fig_max_tabu_tenure_history = plt.figure(figsize=(8, 6))
        plt.plot(max_tabu_tenure_history, marker='o', linestyle='-', color='purple')
        plt.title('禁忌表元素的存在期限的历史')
        plt.xlabel('调整次数')
        plt.ylabel('禁忌表元素的存在期限')
        plt.pause(0.0001)
        plt.show(block=False)

    def update_tsp_solution(self, new_best_solution, new_current_solution):
        # 更新TSP解
        self.best_solution = new_best_solution
        self.current_solution = new_current_solution
        self.plot_tsp_solution()
        
#自定义的非法输入Exception
class Invalid_input_Error(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("禁忌搜索算法求解TSP问题的模拟与实现")

        # 创建一个Text小部件以显示具有指定宽度和高度的日志
        self.log_text = tk.Text(root, height=55, width=70)
        self.log_text.grid(row=0, column=3, columnspan=2,rowspan=11, pady=10, padx=5)
        self.log_text.config(state=tk.DISABLED)  # 使Text小部件为只读
        # 保存原始的sys.stdout以便之后恢复
        self.original_stdout = sys.stdout

        # 将sys.stdout重定向到text_widget
        sys.stdout = self
        
        self.tsp_visualization = None  # 添加这行代码

    def __del__(self):
        # 在对象销毁时（例如窗口关闭），恢复sys.stdout
        sys.stdout = self.original_stdout

    def write(self, message):
        # 重定向print输出到text_widget,实现log功能
        self.redirect_print_to_text_widget(message)

    def flush(self):
        # 添加一个简单的flush方法，不执行任何操作
        pass

    def start_tsp_solver(self, event=None):
        plt.close('all')
        # 回调函数，启动具有用户指定参数的TSP求解器
        num_cities = 20
        initial_tabu_size = int(tabu_size_var.get())
        max_iter = int(max_iter_var.get())
        num_runs = int(num_runs_var.get())
        adjustment_interval = int(adjustment_interval_var.get())
        num_cities = int(num_cities_var.get())
        max_tabu_tenure = int(max_tabu_tenure_var.get())
        max_tabu_tenure_step = int(max_tabu_tenure_step_var.get())
        tabu_size_step = int(tabu_size_step_var.get())
        min_tabu_tenure = int(min_tabu_tenure_var.get())
        refresh_rate = int(refresh_rate_var.get())
        if(tabu_size_step<0 or max_tabu_tenure_step<0 or min_tabu_tenure<0 or refresh_rate<=0) :
            # 检查非法输入，这里只处理经测试会导致死循环，且不不能通过其他地方抛出exception处理的输入
            raise Invalid_input_Error("输入超过了标记的取值范围")

        tsp_solver = TSP(num_cities=num_cities, initial_tabu_size=initial_tabu_size, max_iter=max_iter, num_runs=num_runs,
                         adjustment_interval=adjustment_interval, max_tabu_tenure=max_tabu_tenure,
                         max_tabu_tenure_step=max_tabu_tenure_step, tabu_size_step=tabu_size_step, min_tabu_tenure=min_tabu_tenure, refresh_rate=refresh_rate)

        tsp_solver.coordinates = tsp_solver.generate_random_coordinates()
        distance_matrix = tsp_solver.create_distance_matrix(tsp_solver.coordinates)
        tsp_solver.distance_matrix = distance_matrix
        best_solution=None

        best_solution, best_distance = tsp_solver.tabu_search()

        print("最终最优解:", best_solution)
        print("最终最短距离:", best_distance)
        

    def redirect_print_to_text_widget(self, message):
        # 将print语句重定向到Text小部件
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.yview(tk.END)  # 滚动到底部


if __name__ == "__main__":

    # GUI设置
    root = tk.Tk()
    root.title("禁忌搜索算法求解TSP问题的模拟与实现")
    app = App(root)

    # 参数输入
    ttk.Label(root, text="初始禁忌表长度:").grid(row=0, column=0, padx=5, pady=5)
    tabu_size_var = tk.StringVar(value="100")
    ttk.Entry(root, textvariable=tabu_size_var).grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="每个初始解的迭代次数:").grid(row=1, column=0, padx=5, pady=5)
    max_iter_var = tk.StringVar(value="400")
    ttk.Entry(root, textvariable=max_iter_var).grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(root, text="初始解个数:").grid(row=2, column=0, padx=5, pady=5)
    num_runs_var = tk.StringVar(value="300")
    ttk.Entry(root, textvariable=num_runs_var).grid(row=2, column=1, padx=5, pady=5)

    ttk.Label(root, text="参数调整间隔:").grid(row=3, column=0, padx=5, pady=5)
    adjustment_interval_var = tk.StringVar(value="500")
    ttk.Entry(root, textvariable=adjustment_interval_var).grid(row=3, column=1, padx=5, pady=5)

    ttk.Label(root, text="城市数量:").grid(row=4, column=0, padx=5, pady=5)
    num_cities_var = tk.StringVar(value="20")
    ttk.Entry(root, textvariable=num_cities_var).grid(row=4, column=1, padx=5, pady=5)

    ttk.Label(root, text="禁忌表元素的初始存在期限:").grid(row=5, column=0, padx=5, pady=5)
    max_tabu_tenure_var = tk.StringVar(value="200")
    ttk.Entry(root, textvariable=max_tabu_tenure_var).grid(row=5, column=1, padx=5, pady=5)

    ttk.Label(root, text="禁忌表长度限制的调整步进(>=0):").grid(row=6, column=0, padx=5, pady=5)
    tabu_size_step_var = tk.StringVar(value="50")
    ttk.Entry(root, textvariable=tabu_size_step_var).grid(row=6, column=1, padx=5, pady=5)

    ttk.Label(root, text="禁忌表元素的存在期限的调整步进(>=0):").grid(row=7, column=0, padx=5, pady=5)
    max_tabu_tenure_step_var = tk.StringVar(value="50")
    ttk.Entry(root, textvariable=max_tabu_tenure_step_var).grid(row=7, column=1, padx=5, pady=5)

    ttk.Label(root, text="禁忌表元素的存在期限的最小值(>=0):").grid(row=8, column=0, padx=5, pady=5)
    min_tabu_tenure_var = tk.StringVar(value="10")
    ttk.Entry(root, textvariable=min_tabu_tenure_var).grid(row=8, column=1, padx=5, pady=5)
    
    ttk.Label(root, text="刷新周期(多少次迭代更新一帧)(输入大于0):").grid(row=9, column=0, padx=5, pady=5)
    refresh_rate_var = tk.StringVar(value="3000")
    ttk.Entry(root, textvariable=refresh_rate_var).grid(row=9, column=1, padx=5, pady=5)

    ttk.Button(root, text="开始求解TSP", command=app.start_tsp_solver).grid(row=10, column=0, columnspan=2, pady=10)
    
    root.mainloop()
#最终提交的可执行文件是使用pyinstaller打包的
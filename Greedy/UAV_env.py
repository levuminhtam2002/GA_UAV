import math
import random

import numpy as np


class UAVEnv(object):
    height = ground_length = ground_width = 100  # Chiều dài và chiều rộng của địa điểm và chiều cao chuyến bay của UAV cũng là 100m 
    sum_task_size = 100 * 1048576  # Tổng số tác vụ điện toán 60 Mbits -> 60 80 100 120 140
    loc_uav = [50, 50]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6  # Băng thông 1MHz
    p_noisy_los = 10 ** (-13)  # Công suất tiếng ồn -100dBm
    p_noisy_nlos = 10 ** (-11)  # Công suất tiếng ồn 80
    flight_speed = 50.  # Tốc độ bay 50m/s
    # f_ue = 6e8  # Tần số tính toán của UE 0.6GHz
    f_ue = 2e8  # Tần số tính toán của UE 0.6GHz
    f_uav = 1.2e9  # Tần số tính toán của UAV 1.2GHz
    r = 10 ** (-27)  # Hệ số ảnh hưởng của cấu trúc chip lên xử lý cpu
    s = 1000  # Số chu kỳ cpu cần để xử lý mỗi bit
    p_uplink = 0.1  # Công suất truyền dẫn lên 0.1W
    alpha0 = 1e-5  # Tăng ích kênh tham chiếu khi khoảng cách là 1m -30dB = 0.001, -50dB = 1e-5
    T = 320  # Chu kỳ 320s
    t_fly = 1
    t_com = 7
    delta_t = t_fly + t_com  # 1s bay, 7s tiếp theo để dừng lại và tính toán
    v_ue = 1    # Tốc độ di chuyển của UE 1m/s
    slot_num = int(T / delta_t)  # 40 quãng
    m_uav = 9.65  # Khối lượng UAV/kg
    e_battery_uav = 500000  # Năng lượng pin UAV: 500kJ. ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning

    #################### ues ####################
    M = 10  # Số lượng UE
    block_flag_list = np.random.randint(0, 2, M)  # Tình trạng tắc của UE
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  # Thông tin vị trí: x là ngẫu nhiên từ 0-100
    # task_list = np.random.randint(1572864, 2097153, M)  # Tác vụ tính toán ngẫu nhiên 1,5~2Mbits -> tổng kích thước tác vụ tương ứng 60
    task_list = np.random.randint(2097153, 2621440, M)  # Tác vụ tính toán ngẫu nhiên 2~2.5Mbits -> 80

    action_bound = [-1, 1]  # Tương ứng với hàm kích hoạt tanh
    action_dim = 4  # Đầu tiên biểu thị id của UE; giữa biểu thị góc bay và khoảng cách; cuối biểu thị tốc độ hoàn thành tác vụ hiện tại trên UE (tỷ lệ offloading task trên UE)
    state_dim = 4 + M * 4  # Pin UAV còn lại, vị trí UAV, kích thước tác vụ còn lại, tất cả vị trí UE, tất cả kích thước tác vụ UE, tất cả cờ tắc UE
    act = np.random.uniform(0,1,size = (M,))

    def __init__(self):
        # Pin UAV còn lại, vị trí UAV, kích thước tác vụ còn lại, tất cả vị trí UE, tất cả kích thước tác vụ UE, tất cả cờ tắc UE
        self.start_state = np.append(self.e_battery_uav, self.loc_uav)
        self.start_state = np.append(self.start_state, self.sum_task_size)
        self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
        self.start_state = np.append(self.start_state, self.task_list)
        self.start_state = np.append(self.start_state, self.block_flag_list)
        self.state = self.start_state

    def reset_env(self):
        self.sum_task_size = 100 * 1048576  # Tổng số tác vụ điện toán 60 Mbits -> 60 80 100 120 140
        self.e_battery_uav = 500000  # Năng lượng pin UAV: 500kJ
        self.loc_uav = [50, 50]
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])  # Thông tin vị trí: x là ngẫu nhiên từ 0-100
        self.reset_step()

    def reset_step(self):
        # self.task_list = np.random.randint(1572864, 2097153, self.M)  # Tác vụ tính toán ngẫu nhiên 1,5~2Mbits
        # self.task_list = np.random.randint(2097152, 2621441, self.M)  # Tác vụ tính toán ngẫu nhiên 2~2.5Mbits
        # self.task_list = np.random.randint(2621440, 3145729, self.M)  # Tác vụ tính toán ngẫu nhiên 2.5~3Mbits
        self.task_list = np.random.randint(2621440, 3145729, self.M)  # Tác vụ tính toán ngẫu nhiên 2.5~3Mbits
        # self.task_list = np.random.randint(3145728, 3670017, self.M)  # Tác vụ tính toán ngẫu nhiên 3~3.5Mbits
        # self.task_list = np.random.randint(3670016, 4194305, self.M)  # Tác vụ tính toán ngẫu nhiên 3.5~4Mbits
        self.block_flag_list = np.random.randint(0, 2, self.M)  # Tình trạng tắc của UE

    def reset(self):
        self.reset_env()
        # Pin UAV còn lại, vị trí UAV, kích thước tác vụ còn lại, tất cả vị trí UE, tất cả kích thước tác vụ UE, tất cả cờ tắc UE
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    def _get_obs(self):
        # Pin UAV còn lại, vị trí UAV, kích thước tác vụ còn lại, tất cả vị trí UE, tất cả kích thước tác vụ UE, tất cả cờ tắc UE
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self.state

    def step(self, action):  # 0: Chọn id của UE; 1: Hướng theta; 2: Khoảng cách d; 3: Tỷ lệ offloading
        step_redo = False
        is_terminal = False
        offloading_ratio_change = False
        reset_dist = False
        action = (action + 1) / 2  # Các hành động đặt phạm vi giá trị từ -1~1 -> hành động từ 0~1. Tránh huấn luyện hàm tanh mạng tác nhân luôn lấy ranh giới 0 khi giới hạn hành động ban đầu là [0,1]

        ################# Tìm đối tượng dịch vụ tối ưu UE ###################### 
        # Cải thiện ddpg và thêm một lớp vào lớp đầu ra để xuất ra các hành động rời rạc (kết quả thực hiện không chính xác)
        # Có lỗi khi sử dụng thuật toán khoảng cách gần nhất, nếu sử dụng khoảng cách gần nhất, máy bay không người lái sẽ luôn đậu phía trên đầu (sai)
        # Thăm dò ngẫu nhiên: Đầu tiên tạo hàng đợi số ngẫu nhiên, xóa UE sau khi dịch vụ hoàn thành và tạo lại ngẫu nhiên nếu hàng đợi trống (lỗi logic)
        # Các biến điều khiển được ánh xạ tới phạm vi giá trị của từng biến
        if action[0] == 1:
            ue_id = self.M - 1
        else:
            ue_id = int(self.M * action[0])  # Random chọn UE

        theta = action[1] * np.pi * 2  # Góc
        offloading_ratio = action[3]  # Tỷ lệ giảm tải
        task_size = self.task_list[ue_id]
        block_flag = self.block_flag_list[ue_id]

        # Khoảng cách chuyến bay
        dis_fly = action[2] * self.flight_speed * self.t_fly  # 1s khoảng cách bay
        # Năng lượng tiêu thụ bay
        e_fly = (dis_fly / self.t_fly) ** 2 * self.m_uav * self.t_fly * 0.5  # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning

        # Vị trí UAV sau khi bay
        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        loc_uav_after_fly_x = self.loc_uav[0] + dx_uav
        loc_uav_after_fly_y = self.loc_uav[1] + dy_uav

        # Tính toán tiêu thụ năng lượng
        t_server = offloading_ratio * task_size / (self.f_uav / self.s)  # Độ trễ tính toán trên máy chủ UAV Edge
        e_server = self.r * self.f_uav ** 3 * t_server  # Mức tiêu thụ năng lượng trên UAV edge

        if self.sum_task_size == 0:  # Tất cả các tác vụ tính toán đã hoàn thành
            is_terminal = True
            reward = 0
        elif self.sum_task_size - self.task_list[ue_id] < 0:  # Nhiệm vụ tính toán của bước cuối cùng không khớp với nhiệm vụ tính toán của UE
            self.task_list = np.ones(self.M) * self.sum_task_size
            reward = 0
            step_redo = True
        elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > self.ground_width or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > self.ground_length:  # Vị trí UAV sai
            # Nếu vượt quá biên giới, đặt khoảng cách bay về không
            reset_dist = True
            delay = self.com_delay(self.loc_ue_list[ue_id], self.loc_uav, offloading_ratio, task_size, block_flag)  # Tính toán độ trễ
            reward = -delay
            # Cập nhật trạng thái tiếp theo
            self.e_battery_uav = self.e_battery_uav - e_server  # Pin UAV còn lại
            self.reset2(delay, self.loc_uav[0], self.loc_uav[1], offloading_ratio, task_size, ue_id)
        elif self.e_battery_uav < e_fly or self.e_battery_uav - e_fly < e_server:  # Pin UAV không thể hỗ trợ tính toán
            delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   0, task_size, block_flag)  # Tính toán độ trễ
            reward = -delay
            self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, 0, task_size, ue_id)
            offloading_ratio_change = True
        else:  # Pin hỗ trợ bay, và nhiệm vụ tính toán hợp lý, và nhiệm vụ tính toán có thể được thực hiện trong năng lượng còn lại
            delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   offloading_ratio, task_size, block_flag)  # Tính toán độ trễ
            reward = -delay
            # Cập nhật trạng thái tiếp theo
            self.e_battery_uav = self.e_battery_uav - e_fly - e_server  # Pin UAV còn lại
            self.loc_uav[0] = loc_uav_after_fly_x  # Vị trí UAV sau khi bay
            self.loc_uav[1] = loc_uav_after_fly_y
            self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, offloading_ratio, task_size, ue_id)  # Đặt lại kích thước tác vụ UE, tổng kích thước tác vụ còn lại, vị trí UE và ghi vào tệp

        return self._get_obs(), reward, is_terminal, step_redo, offloading_ratio_change, reset_dist

    # Đặt lại kích thước tác vụ UE, tổng kích thước tác vụ còn lại, vị trí UE và ghi vào tệp
    def reset2(self, delay, x, y, offloading_ratio, task_size, ue_id):
        self.sum_task_size -= self.task_list[ue_id]  # Nhiệm vụ còn lại
        for i in range(self.M):  # Vị trí sau khi di chuyển ngẫu nhiên
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2  # Góc chuyển động ngẫu nhiên
            dis_ue = tmp[1] * self.delta_t * self.v_ue  # Khoảng cách chuyển động ngẫu nhiên
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
        self.reset_step()  # Tác vụ tính toán ngẫu nhiên 1~2Mbits, tình trạng tắc của UE
        # Ghi lại chi phí của UE
        file_name = 'output.txt'
        # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
        with open(file_name, 'a') as file_obj:
            file_obj.write("\nUE-" + '{:d}'.format(ue_id) + ", kích thước tác vụ: " + '{:d}'.format(int(task_size)) + ", tỷ lệ offloading:" + '{:.2f}'.format(offloading_ratio))
            file_obj.write("\nđộ trễ:" + '{:.2f}'.format(delay))
            file_obj.write("\nVị trí hover UAV:" + "[" + '{:.2f}'.format(x) + ', ' + '{:.2f}'.format(y) + ']')  # Xuất kết quả với hai chữ số thập phân

    # Tính toán chi phí
    def com_delay(self, loc_ue, loc_uav, offloading_ratio, task_size, block_flag):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  # Gain of the line of sight
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)  # Tốc độ đường truyền (bps)
        t_tr = offloading_ratio * task_size / trans_rate  # Độ trễ tải lên, 1B=8bit
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)  # Độ trễ tính toán trên máy chủ UAV Edge
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)  # Độ trễ tính toán cục bộ
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        return max([t_tr + t_edge_com, t_local_com])  # Yếu tố tác động thời gian của chuyến bay


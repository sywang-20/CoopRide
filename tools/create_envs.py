"""
Edited by Zilong Guo & Jerry Jin
"""
import sys
import numpy as np
sys.path.append('../')
from simulator.envs import *
from simulator.utilities import *
import os.path as osp

def create_OD_36(fleet_help=False):
    random.seed(0)
    np.random.seed(0)
    num_valid_grid=36
    mapped_matrix_int = np.arange(num_valid_grid)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(6,6))
    central_node_ids = []
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    idle_driver_location_mat = np.zeros((144, num_valid_grid))

    order_grid_param=np.random.randint(2,10,num_valid_grid)

    for ii in np.arange(144):
        time_dict = {}
        random_disturb= np.random.randint(-2,3,num_valid_grid)
        time_dict_param=order_grid_param+random_disturb
        time_dict_param[time_dict_param<0]=0
        #time_dict_param[time_dict_param<=2]=2
        for jj in np.arange(M * N):  # num of grids
            time_dict[jj] = [time_dict_param[jj].item()]  # mean is 2
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :] = [10] * num_valid_grid

    idle_driver_dist_time = []

    n_side = 6
    l_max = 3

    order_time = [0.2, 0.2, 0.15,
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]

    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],
                   [32.14, 16.20],
                   [40.99, 20.69],
                   [49.94, 25.61],
                   [58.98, 31.69],
                   [68.80, 37.25],
                   [79.40, 44.39]]

    order_real = []
    onoff_driver_location_mat = []
    for tt in np.arange(144):
        # order: begin_p, end_p, begin_t, _t, _p, wait_time, serve_type
        order_real += [[1, 5, tt, 1, 13.2, 2, -1], [0, 14, tt, 1, 13.2, 2, -1], [13, 1, tt, 1, 13.2, 2, -1],
                       [5, 3, tt, 1, 13.2, 2, -1], [14, 2, tt, 1, 13.2, 2, -1], [17, 14, tt, 1, 13.2, 2, -1],
                       [7, 1, tt, 1, 13.2, 2, -1], [5, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1],
                       [18, 5, tt, 1, 13.2, 2, -1], [12, 14, tt, 1, 13.2, 2, -1], [14, 1, tt, 1, 13.2, 2, -1],
                       [19, 23, tt, 1, 13.2, 2, -1], [14, 12, tt, 1, 13.2, 2, -1], [13, 14, tt, 1, 13.2, 2, -1],
                       [0, 1, tt, 1, 13.2, 2, -1], [11, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1]]
        onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452]])

    '''
    onoff_driver_location_mat : no use
    
    '''
    env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), fleet_help=fleet_help)

    return env, M, N, central_node_ids, num_valid_grid


def create_OD(fleet_help=False, scale = 1):
    random.seed(0)
    np.random.seed(0)
    num_valid_grid=100
    mapped_matrix_int = np.arange(num_valid_grid)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(10,10))
    central_node_ids = []
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    idle_driver_location_mat = np.zeros((144, num_valid_grid))

    order_grid_param=np.random.randint(0,10*scale,num_valid_grid)

    for ii in np.arange(144):
        time_dict = {}
        random_disturb= np.random.randint(int(-2*scale),int(round(3*scale)),num_valid_grid)
        time_dict_param=order_grid_param+random_disturb
        time_dict_param[time_dict_param<0]=0
        #time_dict_param[time_dict_param<=2]=2
        for jj in np.arange(M * N):  # num of grids
            time_dict[jj] = [time_dict_param[jj].item()]  # mean is 2
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :] = [int(10*scale)] * num_valid_grid

    idle_driver_dist_time = []

    total_driver_num = 0
    for i in range(M*N):
        total_driver_num+=idle_driver_location_mat[0][i]
    total_order_num = 0
    for t in range(144):
        for i in range(M*N):
            total_order_num+=order_num_dist[t][i][0]
    print('订单数量:{} , 司机数量:{}'.format(total_order_num,total_driver_num))

    n_side = 6
    l_max = 5

    order_time = [0.2, 0.2, 0.15,
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]

    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],
                   [32.14, 16.20],
                   [40.99, 20.69],
                   [49.94, 25.61],
                   [58.98, 31.69],
                   [68.80, 37.25],
                   [79.40, 44.39]]

    order_real = []
    onoff_driver_location_mat = []
    for tt in np.arange(144):
        # order: begin_p, end_p, begin_t, _t, _p, wait_time, serve_type
        order_real += [[1, 5, tt, 1, 13.2, 2, -1], [0, 14, tt, 1, 13.2, 2, -1], [13, 1, tt, 1, 13.2, 2, -1],
                       [5, 3, tt, 1, 13.2, 2, -1], [14, 2, tt, 1, 13.2, 2, -1], [17, 14, tt, 1, 13.2, 2, -1],
                       [7, 1, tt, 1, 13.2, 2, -1], [5, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1],
                       [18, 5, tt, 1, 13.2, 2, -1], [12, 14, tt, 1, 13.2, 2, -1], [14, 1, tt, 1, 13.2, 2, -1],
                       [19, 23, tt, 1, 13.2, 2, -1], [14, 12, tt, 1, 13.2, 2, -1], [13, 14, tt, 1, 13.2, 2, -1],
                       [0, 1, tt, 1, 13.2, 2, -1], [11, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1]]
        onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452]])

    '''
    onoff_driver_location_mat : no use
    
    '''
    env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), fleet_help=fleet_help)

    return env, M, N, central_node_ids, num_valid_grid

def create_env_parameter(fleet_help=False):
    random.seed(0)
    np.random.seed(0)
    num_valid_grid=100
    mapped_matrix_int = np.arange(num_valid_grid)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(10,10))
    central_node_ids = []
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    idle_driver_location_mat = np.zeros((144, num_valid_grid))

    order_grid_param=np.random.randint(0,10,num_valid_grid)

    for ii in np.arange(144):
        time_dict = {}
        random_disturb= np.random.randint(-2,3,num_valid_grid)
        time_dict_param=order_grid_param+random_disturb
        time_dict_param[time_dict_param<0]=0
        #time_dict_param[time_dict_param<=2]=2
        for jj in np.arange(M * N):  # num of grids
            time_dict[jj] = [time_dict_param[jj].item()]  # mean is 2
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :] = [10] * num_valid_grid

    idle_driver_dist_time = []

    n_side = 6
    l_max = 5

    order_time = [0.2, 0.2, 0.15,
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]

    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],
                   [32.14, 16.20],
                   [40.99, 20.69],
                   [49.94, 25.61],
                   [58.98, 31.69],
                   [68.80, 37.25],
                   [79.40, 44.39]]

    order_real = []
    onoff_driver_location_mat = []
    for tt in np.arange(144):
        # order: begin_p, end_p, begin_t, _t, _p, wait_time, serve_type
        order_real += [[1, 5, tt, 1, 13.2, 2, -1], [0, 14, tt, 1, 13.2, 2, -1], [13, 1, tt, 1, 13.2, 2, -1],
                       [5, 3, tt, 1, 13.2, 2, -1], [14, 2, tt, 1, 13.2, 2, -1], [17, 14, tt, 1, 13.2, 2, -1],
                       [7, 1, tt, 1, 13.2, 2, -1], [5, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1],
                       [18, 5, tt, 1, 13.2, 2, -1], [12, 14, tt, 1, 13.2, 2, -1], [14, 1, tt, 1, 13.2, 2, -1],
                       [19, 23, tt, 1, 13.2, 2, -1], [14, 12, tt, 1, 13.2, 2, -1], [13, 14, tt, 1, 13.2, 2, -1],
                       [0, 1, tt, 1, 13.2, 2, -1], [11, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1]]
        onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452]])

    '''
    onoff_driver_location_mat : no use
    
    '''

    return mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,order_time, order_price, l_max, M, N, n_side, 144, 1, np.array(order_real),np.array(onoff_driver_location_mat)


def dispatch_env(fleet_help):
    mapped_matrix_int = np.array([[0, 1, 2, -100, -100],
                                  [5, 6, 7, -100, -100],
                                  [-100, 11, 12, 13, 14],
                                  [-100, -100, 17, 18, 19],
                                  [-100, -100, -100, 23, -100]])
    central_node_ids = [6, 18]
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    num_valid_grid = 14
    idle_driver_location_mat = np.zeros((144, 14))

    for ii in np.arange(144):
        time_dict = {}
        for jj in np.arange(M * N):  # num of grids
            time_dict[jj] = [2]  # mean is 2
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :] = [2] * num_valid_grid

    idle_driver_dist_time = [[2, 1] for _ in np.arange(144)]

    n_side = 6
    l_max = 2

    order_time = [0.2, 0.2, 0.15,
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]

    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],
                   [32.14, 16.20],
                   [40.99, 20.69],
                   [49.94, 25.61],
                   [58.98, 31.69],
                   [68.80, 37.25],
                   [79.40, 44.39]]

    order_real = []
    onoff_driver_location_mat = []
    for tt in np.arange(144):
        # order: begin_p, end_p, begin_t, _t, _p, wait_time, serve_type
        order_real += [[1, 5, tt, 1, 13.2, 2, -1], [0, 14, tt, 1, 13.2, 2, -1], [13, 1, tt, 1, 13.2, 2, -1],
                       [5, 3, tt, 1, 13.2, 2, -1], [14, 2, tt, 1, 13.2, 2, -1], [17, 14, tt, 1, 13.2, 2, -1],
                       [7, 1, tt, 1, 13.2, 2, -1], [5, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1],
                       [18, 5, tt, 1, 13.2, 2, -1], [12, 14, tt, 1, 13.2, 2, -1], [14, 1, tt, 1, 13.2, 2, -1],
                       [19, 23, tt, 1, 13.2, 2, -1], [14, 12, tt, 1, 13.2, 2, -1], [13, 14, tt, 1, 13.2, 2, -1],
                       [0, 1, tt, 1, 13.2, 2, -1], [11, 14, tt, 1, 13.2, 2, -1], [14, 7, tt, 1, 13.2, 2, -1]]
        onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452]])

    '''
    onoff_driver_location_mat : no use
    
    '''
    env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), fleet_help=fleet_help)

    return env, M, N, central_node_ids, num_valid_grid


def fleet_env(fleet_help):
    mapped_matrix_int = np.array([[0, 1, 2, -100, -100],
                                  [5, 6, 7, -100, -100],
                                  [-100, 11, 12, 13, 14],
                                  [-100, -100, 17, 18, 19],
                                  [-100, -100, -100, 23, -100]])
    central_node_ids = [6, 18]
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    num_valid_grid = 14
    idle_driver_location_mat = np.zeros((144, 14))

    for ii in np.arange(144):
        time_dict = {}
        for jj in np.arange(M * N):  # num of grids
            time_dict[jj] = [2]  # mean is 2
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :7] = [10] * 7

    idle_driver_dist_time = [[2, 1] for _ in np.arange(144)]

    n_side = 6
    l_max = 2

    order_time = [0.2, 0.2, 0.15,
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]

    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],
                   [32.14, 16.20],
                   [40.99, 20.69],
                   [49.94, 25.61],
                   [58.98, 31.69],
                   [68.80, 37.25],
                   [79.40, 44.39]]

    order_real = []
    onoff_driver_location_mat = []
    for tt in np.arange(144):
        area1 = [0,1,2,5,6,7,11]
        area2 = [12,13,14,17,18,19,23]
        # order: begin_p, end_p, begin_t, _t, _p, wait_time, serve_type
        for i in range(100):
            order_real += [[random.choice(area2), random.choice(area1), tt, 1, 13.2, 2, -1]]
        for i in range(5):
            order_real += [[random.choice(area1), random.choice(area1), tt, 1, 13.2, 2, -1]]
        for i in range(5):
            order_real += [[random.choice(area2), random.choice(area2), tt, 1, 13.2, 2, -1]]
        onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452]])
    env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), fleet_help=fleet_help)

    return env, M, N, central_node_ids, num_valid_grid


def two_dis(fleet_help):
    mapped_matrix_int = np.array([[0, 1, 2, -100, -100],
                                  [5, 6, 7, -100, -100],
                                  [-100, 11, 12, 13, 14],
                                  [-100, -100, 17, 18, 19],
                                  [-100, -100, -100, 23, -100]])
    central_node_ids = [6, 18]
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    num_valid_grid = 14
    idle_driver_location_mat = np.zeros((144, 14))

    for ii in np.arange(144):
        time_dict = {}
        for jj in np.arange(M * N):  # num of grids
            time_dict[jj] = [2]  # mean is 2
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :7] = [10] * 7

    idle_driver_dist_time = [[2, 1] for _ in np.arange(144)]

    n_side = 6
    l_max = 2

    order_time = [0.2, 0.2, 0.15,
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]

    order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                   [15.02, 6.90],  # mean and std of order price when duration is 20 min
                   [23.22, 11.63],
                   [32.14, 16.20],
                   [40.99, 20.69],
                   [49.94, 25.61],
                   [58.98, 31.69],
                   [68.80, 37.25],
                   [79.40, 44.39]]

    order_real = []
    onoff_driver_location_mat = []
    for tt in np.arange(144):
        area1 = [0, 1, 2, 5, 6, 7, 11]
        area2 = [12, 13, 14]
        area3 = [17, 18, 19, 23]
        # order: begin_p, end_p, begin_t, _t, _p, wait_time, serve_type
        for i in range(50):
            order_real += [[random.choice(area2), random.choice(area1), tt, 1, 13.2, 2, -1]]
        for i in range(50):
            order_real += [[random.choice(area3), random.choice(area1), tt, 1, 13.2, 2, -1]]
        for i in range(5):
            order_real += [[random.choice(area1), random.choice(area1), tt, 1, 13.2, 2, -1]]
        for i in range(5):
            order_real += [[random.choice(area2+area3), random.choice(area2+area3), tt, 1, 13.2, 2, -1]]
        onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452]])
    env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), fleet_help=fleet_help)

    return env, M, N, central_node_ids, num_valid_grid


if __name__ == '__main__':
    '''
    param=create_env_parameter()
    env=CityReal(*param)
    env.reset()
    print(0)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
    print(sys.path)
    '''
    create_OD(scale=3)
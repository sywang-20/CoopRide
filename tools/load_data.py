import numpy as np
import pickle 
import sys 
sys.path.append('../')
from simulator.envs import *

def load_envs_DiDi121(driver_num=2000):
    with open('../data/DiDi/DiDi_day1_grid121.pkl', 'rb') as handle:
        data_param = pickle.load(handle) 
    dura_param = data_param['duration'] # 每
    price_param = data_param['price']   # 每级邻居的价格    第0级表示自己网格 0~l_max
    neighbor = data_param['neighbor']    # neighbor>=100 表示不可达的订单
    order_param = data_param['order']   # shape=(11,11,144)  表示 (出发地，目的地，出发时间)
    l_max = 6   # 最大通勤跨邻居数
    M,N = 11,11
    price_param[:,1]/=2     # 减小订单的方差
    np.random.seed(0)
    # 减小order param 数量
    commute1 = order_param.astype(np.float32)
    commute1[(neighbor==0)] = 0
    index = commute1>=3
    commute1[index] = (commute1[index]-3)*0.2+3
    index = commute1>=2
    commute1 = np.round(commute1/2+0.1)
    commute1[neighbor==100]=0
    random_delete = np.random.randint(0,3,(commute1.shape))
    random_delete[commute1.sum(-1).sum(-1)<3000] = 0
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 添加与邻居相关的随机数据
    random_grid_num = np.random.randint(1,6,M*N)
    random_prob = np.zeros((M*N,M*N))
    prob_list = [0.05,0.2,0.4,0.15,0.1,0.05,0.05]
    for k in range(7):
        index= neighbor==k
        random_prob+= prob_list[k]/np.sum(index,axis=-1,keepdims=True)*index
    random_add = np.zeros(commute1.shape)
    for i in range(M*N):
        sample = np.random.choice(M*N,size = (random_grid_num[i],commute1.shape[-1]),replace=True, p=random_prob[i])    
        for t in range(commute1.shape[-1]):
            random_add[i,sample[:,t],t] = 1
    commute1+=random_add
    # 删除数量多的
    random_delete = np.random.randint(0,2,(commute1.shape))
    #random_delete[random_delete<=1] = 0
    index = commute1.sum(1)<=25
    random_delete = random_delete.swapaxes(1,2)
    random_delete[index] = 0
    random_delete = random_delete.swapaxes(1,2)
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 全部删除一点点
    random_delete = np.random.randint(0,5,(commute1.shape))
    random_delete[random_delete<=3] = 0
    random_delete[random_delete>3] = 1
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    order_param = commute1.astype(np.int32)
    # 初始化司机数量
    driver_param=np.zeros(M*N,dtype=np.int32)+1
    order_num = np.sum(np.sum(order_param, axis=1), axis=1)
    driver_param[order_num>=100]= driver_param[order_num>=100]+3
    driver_param[order_num>=400]= driver_param[order_num>=400]+3
    driver_param[order_num>=800]= driver_param[order_num>=800]+3
    driver_param = driver_param*driver_num/np.sum(driver_param)
    driver_param = driver_param.astype(np.int32)
    random_add = np.random.choice(M*N, driver_num-np.sum(driver_param), replace = True)
    for dri in random_add:
        driver_param[dri] += 1
    # 统计数量特别多的网格id
    large_grid_dist={i:0 for i in range(121)}
    a= np.sum(order_param, axis=1)
    for i in range(144):
        b = np.where(a[:,i]>=100)[0]
        for n in b:
            large_grid_dist[n]+=1
    large_grid=[]
    for k,v in large_grid_dist.items():
        if v>0:
            large_grid.append(k)
    print('订单数量: {} , 司机数量: {}'.format( np.sum(order_num), np.sum(driver_param)))

    # 处理为envs的参数
    mapped_matrix_int = np.arange(M*N)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(M,N))
    order_num = np.sum(order_param, axis=1)
    order_num_dict = []
    for t in range(144):
        order_num_dict.append( {i:[order_num[i,t]] for i in range(M*N)} )
    idle_driver_location_mat = np.zeros((144, M*N))
    for t in range(144):
        idle_driver_location_mat[t] = driver_param
    order_time = [0.2, 0.2, 0.15,       # 没用
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]
    n_side = 6
    order_real = []
    onoff_driver_location_mat=[]
    env = CityReal(mapped_matrix_int, order_num_dict, [], idle_driver_location_mat,
                   order_time, price_param, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), neighbor_dis = neighbor , order_param=order_param ,fleet_help=False)
    return env, M, N, None, M*N



def load_envs_NYU143(driver_num=2000):
    with open('../data/NYU/NYU_grid143.pkl', 'rb') as handle:
        data_param = pickle.load(handle) 
    price_param = data_param['price']   # 每级邻居的价格    第0级表示自己网格 0~l_max
    neighbor = data_param['neighbor']    # neighbor>=100 表示不可达的订单
    order_param = data_param['order']   # shape=(11,11,144)  表示 (出发地，目的地，出发时间)
    M,N = data_param['shape']
    l_max = 6   # 最大通勤跨邻居数
    # 减小order param 数量
    np.random.seed(0)
    commute1 = order_param.astype(np.float32)
    commute1[(neighbor==0)] = 0
    index = commute1>=3
    commute1[index] = (commute1[index]-3)*0.2+3
    index = commute1>=2
    commute1 = np.round(commute1/2+0.1)
    commute1[neighbor==100]=0
    random_delete = np.random.randint(0,3,(commute1.shape))
    random_delete[commute1.sum(-1).sum(-1)<3000] = 0
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    random_delete = np.random.randint(0,2,(commute1.shape[1],commute1.shape[2]))
    commute1[68]-=random_delete
    commute1[commute1<0] = 0
    # 添加与邻居相关的随机数据
    random_grid_num = np.random.randint(1,6,M*N)
    random_prob = np.zeros((M*N,M*N))
    prob_list = [0.05,0.2,0.4,0.15,0.1,0.05,0.05]
    for k in range(7):
        index= neighbor==k
        random_prob+= prob_list[k]/np.sum(index,axis=-1,keepdims=True)*index
    random_add = np.zeros(commute1.shape)
    for i in range(M*N):
        sample = np.random.choice(M*N,size = (random_grid_num[i],commute1.shape[-1]),replace=True, p=random_prob[i])    
        for t in range(commute1.shape[-1]):
            random_add[i,sample[:,t],t] = 1
    commute1+=random_add
    # 删除数量多的
    random_delete = np.random.randint(0,2,(commute1.shape))
    #random_delete[random_delete<=1] = 0
    index = commute1.sum(1)<=25
    random_delete = random_delete.swapaxes(1,2)
    random_delete[index] = 0
    random_delete = random_delete.swapaxes(1,2)
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 全部删除一点点
    random_delete = np.random.randint(0,5,(commute1.shape))
    random_delete[random_delete<=3] = 0
    random_delete[random_delete>3] = 1
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    order_param = commute1.astype(np.int32)
    # 初始化司机数量
    driver_param=np.ones(M*N,dtype=np.int32)
    driver_param = driver_param*driver_num/np.sum(driver_param)
    driver_param = driver_param.astype(np.int32)
    random_add = np.random.choice(M*N, driver_num-np.sum(driver_param), replace = True)
    for dri in random_add:
        driver_param[dri] += 1
    # 统计数量特别多的网格id
    large_grid_dist={i:0 for i in range(121)}
    a= np.sum(order_param, axis=1)
    for i in range(144):
        b = np.where(a[:,i]>=100)[0]
        for n in b:
            large_grid_dist[n]+=1
    large_grid=[]
    for k,v in large_grid_dist.items():
        if v>0:
            large_grid.append(k)
    order_num = order_param.sum()
    print('订单数量: {} , 司机数量: {}'.format( np.sum(order_num), np.sum(driver_param)))

    # 处理为envs的参数
    mapped_matrix_int = np.arange(M*N)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(M,N))
    order_num = np.sum(order_param, axis=1)
    order_num_dict = []
    for t in range(144):
        order_num_dict.append( {i:[order_num[i,t]] for i in range(M*N)} )
    idle_driver_location_mat = np.zeros((144, M*N))
    for t in range(144):
        idle_driver_location_mat[t] = driver_param
    order_time = [0.2, 0.2, 0.15,       # 没用
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]
    n_side = 6
    order_real = []
    onoff_driver_location_mat=[]
    env = CityReal(mapped_matrix_int, order_num_dict, [], idle_driver_location_mat,
                   order_time, price_param, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), neighbor_dis = neighbor , order_param=order_param ,fleet_help=False)
    return env, M, N, None, M*N



if __name__ == '__main__':     

    load_envs_DiDi121(2000)
    #load_envs_NYU143(2000)
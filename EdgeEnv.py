# -*- coding: utf-8 -*-
"""
SysAdmin env 
"""
import networkx as nx
import numpy as np
from gym.spaces import MultiDiscrete
from utilities import action_comb
import itertools

class  build_env():
    '''
        Create Edge environment
        Parameters:
        ----------------------------------------------
        N: number of Edge servers
        net_struc: network structure: "grid"
    ''' 
    def __init__(self):
        self.N = 4
        fea_set = [0,1,2]
        action_temp = [fea_set]*self.N
        self.action_comb = [list(_) for _ in list(itertools.product(*action_temp))]

        self.net_struc = 'grid'
        if  self.net_struc == "grid":
            self.nxy = [2, 2]
            self.lenxy = [3*30,3*30]
        self.eps_len = 24
        self.graph = nx.Graph()
        self.s_N_t = [None]*self.N
        self.ser_demand_exp = None
        self.done = False
        self.comm_range = 43
        self.time = 0
        self.exp_demand = {"residential": [5.0, 4.0, 3.0, 2.0, 0.5, 0.5, 1.5, 3.0, 6.0, 5.0, 4.5, 2.5, 
                                    1.5, 0.5, 0.5, 1.0, 3.0, 6.0, 10.0, 15.0, 22.0, 20.0, 12.0, 8.0],
                    "school": [0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 2.5, 4.5, 8.0, 8.5, 8.5, 7.5,
                               9.0, 8.5, 8.0, 8.0, 7.5, 7.0, 6.0, 2.5, 0.5, 0.0, 0.0, 0.0 ],
                    "commercial": [0.5, 0.5, 0.5, 1.0, 1.5, 3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                                   4.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 7.5, 3.5, 1.5, 0.5],
                    "public": [0.5, 0.5, 1.5, 2.5, 4.5, 7.5, 9.5, 10.5, 10.5, 7.5, 2.5, 1.0, 
                               1.0, 0.5, 0.5, 2.5, 5.5, 6.5, 6.5, 5.5, 1.0, 0.5, 0, 0]
                    }
        self.maximum_ser_delay = 5
        self.prev_edge_delay = [0.0]*self.N
        self.edge_delay = [0.0]*self.N
        self.edge_unit_price = [0.0]*self.N
        self.user_num = 0
        self.edge_x_loc = None
        self.edge_y_loc = None

    def get_neighbor_action(self):
        self.neighbor_action_comb_N = [None]*self.N
        for n in range(self.N):
            neighbors = [_ for _ in self.graph.neighbors(n)]
            num_neighbors = len(neighbors)
            fea_set = [0,1,2]
            action_temp = [fea_set]*num_neighbors
            self.neighbor_action_comb_N[n] = [list(_) for _ in list(itertools.product(*action_temp))]
            
        
    def topo_build(self):
        self.graph.add_nodes_from(range(self.N))
        x_temp = np.linspace(30,60,self.nxy[0])
        y_temp = np.linspace(30,60,self.nxy[1])
        x_loc, y_loc = np.meshgrid(x_temp, y_temp)
        x_loc = np.reshape(x_loc, self.N, 1)
        y_loc = np.reshape(y_loc, self.N, 1) 
        self.edge_x_loc = x_loc
        self.edge_y_loc = y_loc
        for n in range(self.N):
            x_dis = x_loc - x_loc[n];
            y_dis = y_loc - y_loc[n];
            dis = np.sqrt(x_dis**2 + y_dis**2)
            neighbor_n = [idx for (idx,val) in zip(range(self.N), dis) if val <= 30]            
            for _ in neighbor_n:
                self.graph.add_edge(n,_)
        
        edge_loc_type = [None]*self.N
        for n in range(self.N):
            if self.edge_x_loc[n] <= 0.3*self.lenxy[0] and self.edge_y_loc[n] <= 0.5*self.lenxy[1]:
                edge_loc_type[n] = "school"
            elif self.edge_x_loc[n] > 0.3*self.lenxy[0] and self.edge_y_loc[n] <= 0.5*self.lenxy[1]:
                edge_loc_type[n] = "commercial"
            elif self.edge_x_loc[n] <= 0.5*self.lenxy[0] and self.edge_y_loc[n] >= 0.5*self.lenxy[1]:
                edge_loc_type[n] = "residential"
            elif self.edge_x_loc[n] >= 0.5*self.lenxy[0] and self.edge_y_loc[n] >= 0.5*self.lenxy[1]:
                edge_loc_type[n] = "public"
        
        self.edge_loc_type = edge_loc_type        
                
    def system_update(self):
        exp_user_num = 30
        self.user_num = np.random.poisson(exp_user_num)
        self.edge_bandwidth = np.random.normal(loc=20, scale=6.0, size=self.N)
        self.edge_bandwidth = self.edge_bandwidth.clip(1,np.inf)
        
        
        #for n in range(self.N):
        #    exp_demand_n = self.exp_demand[self.edge_loc_type[n]][self.time]
        #    self.edge_unit_price = np.random.normal(loc = 15 + exp_demand_n, scale = 4.0, size=self.N).clip(1,np.inf)
        self.edge_unit_price = np.random.random(4)*40
        

        
        
        user_num = self.user_num
        user_x_loc = np.random.random(user_num)*self.lenxy[0]
        user_y_loc = np.random.random(user_num)*self.lenxy[1]


        ## determine the area type
        user_loc_type = [None]*user_num
        asso_edge = [None]*user_num
        user_ser_demand = [None]*user_num
        for m in range(user_num):
            if user_x_loc[m] <= 0.3*self.lenxy[0] and user_y_loc[m] <= 0.5*self.lenxy[1]:
                user_loc_type[m] = "school"
            elif user_x_loc[m] > 0.3*self.lenxy[0] and user_y_loc[m] <= 0.5*self.lenxy[1]:
                user_loc_type[m] = "commercial"
            elif user_x_loc[m] <= 0.5*self.lenxy[0] and user_y_loc[m] >= 0.5*self.lenxy[1]:
                user_loc_type[m] = "residential"
            elif user_x_loc[m] >= 0.5*self.lenxy[0] and user_y_loc[m] >= 0.5*self.lenxy[1]:
                user_loc_type[m] = "public"
                
            ## get associate     
            user_edge_x_dis = user_x_loc[m] - self.edge_x_loc
            user_edge_y_dis = user_y_loc[m] - self.edge_y_loc
            user_edge_dis = np.sqrt(user_edge_x_dis**2 + user_edge_y_dis**2)
            reachable_edge = [idx for (idx,val) in zip(range(self.N), user_edge_dis) if val <= self.comm_range]            
            prev_edge_delay = self.prev_edge_delay
            temp_prev_delay = [prev_edge_delay[_] for _ in reachable_edge]
            temp_asso_prob = np.exp(-1*np.array(temp_prev_delay))/sum(np.exp(-1*np.array(temp_prev_delay)))
            if np.isnan(temp_asso_prob).any():
                debug = 1
                
            asso_edge[m] = np.random.choice(reachable_edge, 1, p = temp_asso_prob)
            
            ## get service demand of a user
            exp_demand_m = self.exp_demand[user_loc_type[m]][self.time]
            
            alpha_ratio = 0.2
            if self.prev_edge_delay[asso_edge[m][0]] > self.maximum_ser_delay:
                modified_exp_demand_m = alpha_ratio *exp_demand_m
            else:
                modified_exp_demand_m = max(alpha_ratio*exp_demand_m, 
                                            (self.maximum_ser_delay - self.prev_edge_delay[asso_edge[m][0]])/self.maximum_ser_delay *exp_demand_m)
            user_ser_demand[m] = np.random.poisson(modified_exp_demand_m)
        self.user_ser_demand = user_ser_demand
        self.asso_edge = asso_edge
        
        #number of users connected to edge 
        self.edge_user_num = [None]*self.N
        for n in range(self.N):
            edge_user_x_dis = user_x_loc - self.edge_x_loc[n]
            edge_user_y_dis = user_y_loc - self.edge_y_loc[n]
            edge_user_dis = np.sqrt(edge_user_x_dis**2 + edge_user_y_dis**2)
            reachable_user = [idx for (idx,val) in zip(range(self.user_num), edge_user_dis) if val <= self.comm_range] 
            self.edge_user_num[n] = len(reachable_user)
            
    def get_edge_reward(self, a_N):
        ## calculated service delay
        self.edge_ser_demand = [0.0]*self.N
        self.user_ser_delay = [0.0]*self.N
        task_unit_size = 1 #MB
        unit_resource = 40
        for n in range(self.N):
            user_to_n = [user_idx for (user_idx,val) in zip(range(self.user_num), self.asso_edge) if val == n]   
            demand_to_n = [self.user_ser_demand[user_idx] for user_idx in user_to_n]
            self.edge_ser_demand[n] = sum(demand_to_n)
            if a_N[0] == 0:
                self.edge_delay[n] = 0
            else:
                self.edge_delay[n] = 1/max(a_N[n]*unit_resource - self.edge_ser_demand[n],0.2) + min(5, task_unit_size/(self.edge_bandwidth[n]/max(1,len(user_to_n))))
            
        ## calculate reward
        self.edge_reward = [0.0]*self.N
        task_unit_reward = 5
        edge_unit_price = self.edge_unit_price
        for n in range(self.N):
            if a_N[n] == 0:
                self.edge_reward[n] = 0
            else:
                self.edge_reward[n] =0.1*((task_unit_reward - 0.7*self.edge_delay[n])*self.edge_ser_demand[n] - a_N[n]*edge_unit_price[n])           
    
    
    def get_state(self):
        for n in range(self.N):
            self.s_N_t[n] =[self.prev_edge_delay[n],self.edge_user_num[n], self.edge_bandwidth[n], self.edge_unit_price[n], self.time]
        return self.s_N_t    
        
    def reset(self):
        self.time = 0
        self.done = False
        self.prev_edge_delay = [0.0]*self.N
        self.system_update()

    def step(self, a_N):
        #calculate reward
        self.get_edge_reward(a_N)
        rew_N_t = self.edge_reward
        self.prev_edge_delay = self.edge_delay
        if self.time == 23:
            self.done = True
            done = self.done    
            self.reset()
        else:
            self.time += 1
            done = self.done
            self.system_update()
            
        s_N_t1 = self.get_state()
            
        return s_N_t1, rew_N_t, done 
                




   
    

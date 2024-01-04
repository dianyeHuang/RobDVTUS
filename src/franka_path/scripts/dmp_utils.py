#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-10-11 15:45:25
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-01-04 19:08:33
Description: 
'''

import numpy as np
import tf.transformations as t
from math import exp, sqrt, cos, sin, acos, floor
import matplotlib.pyplot as plt

class Quaternion:   
    def __init__(self, *args):
        if len(args) == 1:    
            q_arr = args[0]         # q_arr:np.array (qx, qy, qz, qw)
            self.eta = q_arr[3]
            self.epsilon = q_arr[:3:]
        elif len(args) == 2:   
            self.eta = args[0]      # eta:float (qw)
            self.epsilon = args[1]  # epsilon:np.array (qx, qy, qz)

    def skrew_mat(self, vec):
        x, y, z = vec
        return np.array([[ 0, -z,  y],
                        [ z,  0, -x],
                        [-y,  x,  0]]) 

    def __str__(self):
        op = ['i ', 'j ', 'k']
        result = ''
        if self.eta < -1e-8 or self.eta > 1e-8:
            result += str(round(self.eta, 4)) + ' '
        else:
            result += '0.0 '
        for i in range(3):
            val = self.epsilon[i]
            if (val < -1e-8) or (val > 1e-8):
                result += str(round(val, 4)) + op[i]
            else:
                result += '0.0' + op[i]
        return result

    def __add__(self, q):
        real = self.eta + q.eta
        imag = self.epsilon + q.epsilon
        return Quaternion(real, imag)
    
    def __sub__(self, q):
        real = self.eta - q.eta
        imag = self.epsilon - q.epsilon
        return Quaternion(real, imag)

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            real = self.eta*q.eta-np.dot(self.epsilon, q.epsilon)
            imag = self.eta*q.epsilon + q.eta*self.epsilon + \
                np.dot(self.skrew_mat(self.epsilon),q.epsilon)
            return Quaternion(real, imag)
        if isinstance(q, float) or isinstance(q, int):
            return Quaternion(q*self.eta, q*self.epsilon)
    
    def copy(self):
        return Quaternion(self.eta, self.epsilon)

    def conj(self):
        return Quaternion(self.eta, -1.0*self.epsilon)
    
    def wxyz_vec(self):
        return np.vstack((np.array(self.eta),self.epsilon.reshape(3,1)))
    
    def xyzw_vec(self):
        return np.array(self.epsilon.tolist() + [self.eta])

# weighted truncated approximators
class TruncGaussApprox:
    '''
    Description: 
        A DMP approximator with truncated gaussian function and 
        scaled bias term, the weightings are learned by Local 
        Weighted Regression method. 
    @ param : n_kfs{float}    --number of kernel functions
    @ param : flag_dmp{bool}  --if use as original dmp approximator 
    @ param : th{float}       --gaussian width, 
    @ param : k{float}        --threshold for guassian function k standard
                                deviation of the center would be truncated
    '''    
    def __init__(self, n_kfs:float, flag_dmp=False, th=2, k=2):   
        # meta_parameters
        if n_kfs % 2 != 0:      # make the number of kernel function odd
            self.n_kfs = n_kfs+1
        else:
            self.n_kfs = n_kfs
        delta_c = 1/(self.n_kfs-1)
        self.centers =  np.array(range(self.n_kfs))*delta_c # evenly spaced
        self.width = th/delta_c**2
        self.thresh = k/sqrt(self.width) 
        self.flag_dmp = flag_dmp
        # model parameters
        self.weights = list()
        self.biases = list()

    def truncated_gaussian(self, x, center, width, thresh):
        '''
        Description: 
            a truncated gaussian function
        @ param : x{float}      --input variable
        @ param : center{float} --center of the kernel function
        @ param : width{float}  --width of the kernel function
        @ param : thresh{float} --deviation of the center would be truncated
        @ return: {float} 
        '''        
        if abs(x-center) <= thresh:
            return exp(-width/2*(x-center)**2)
        return 0
    
        # return exp(-width/2*(x-center)**2)
    
    
    def bias_activation(self, x):
        '''
        Description: 
            return a weighting scales for the bias term
        ''' 
        if self.flag_dmp:
            a = 2000
            return 1 - exp(-a*x**2)
        return x # TODO

    def forward(self, x, flag_s=False):
        '''
        Description: 
        @ param : x{float} --input variable
        @ return: {float}  --output of the approximator 
        '''        
        sum_psi = 0
        weighted_sum_psi = 0
        for c, w, b in zip(self.centers, self.weights, self.biases):
            psi = self.truncated_gaussian(x, c, self.width, self.thresh)
            if flag_s:
                x = self.input_modulation(x)
            As = self.bias_activation(x)
            sum_psi += psi
            weighted_sum_psi += psi*(w*x + As*b)
        return weighted_sum_psi/sum_psi
    
    def input_modulation(self, x, a=20000):
        '''
        Description: 
            A mapping that modulates the input x so that x->1 when xM(x)->0
        @ param : a{float} -- scaling factor for exponential function
        @ return: {float}  -- xM(x)
        '''        
        return x*(1-exp(a*(x-1))) # TODO new things
        # return x
    
    def learn_params(self, x_input, y_output, flag_s=False):
        '''
        Description: 
            learn the weightings of the approximator given the input and 
            output data, i.e. mapping x to y, y_output=approximator(x_input)
        @ param : x_input{list}   
        @ param : y_output{list}  
        @ param : flag_s{bool}  -- if x start from 0 then True, otherwise False
        @ return: {bool}        -- True->successfully learns the params, False->otherwise
        '''        
        # LWR method 
        self.weights = list()
        self.biases = list()
        
        for c in self.centers:
            Sx2, Sxy = 0, 0
            Spsi = 0
            for x, f in zip(x_input, y_output):
                psi = self.truncated_gaussian(x, c, 
                                self.width, self.thresh)
                if flag_s:
                    x = self.input_modulation(x)
                Sx2 += psi*x*x
                Sxy += psi*x*f
                Spsi += psi
                
            if Sx2 > -1e-8 and Sx2 < 1e-8: # exit
                return False
            w =  Sxy/Sx2 
            
            self.weights.append(w)
            self.biases.append(0)
        return True
    
    def rollout(self, x_vec, flag_s=False):
        '''
        Description: 
            rollout a sequence of outputs of the approximator 
        @ param : x_vec{list}   --a list of input variables 
        @ return: y_list{list}  --a list of approximator output
        '''        
        y_list = list()
        for x in x_vec:
            y_list.append(self.forward(x, flag_s))
        return y_list

class AL_PosDMP:
    def __init__(self, n_kfns=31, th=2, k=2, flag_dmp=False):
        self.x_approx = TruncGaussApprox(n_kfns, th=th, k=k, flag_dmp=flag_dmp)
        self.y_approx = TruncGaussApprox(n_kfns, th=th, k=k, flag_dmp=flag_dmp)    
        self.z_approx = TruncGaussApprox(n_kfns, th=th, k=k, flag_dmp=flag_dmp)
        self.x0 = None
        self.xg = None
        
        self.dist_length = None
    
    def learn_params(self, input:np.array, traj:np.array, flag_s=True):
        '''
        Description: 
            fitting the traj. with input, learn the weightings
        @ param : input{np.array} -- Nx1 np.array
        @ param : traj{np.array}  -- Nx3 np.array position info
        '''     
        self.x0 = traj[0]
        self.xg = traj[-1]
        x_output = traj[:, 0] - input*self.xg[0] - (1-input)*self.x0[0]
        y_output = traj[:, 1] - input*self.xg[1] - (1-input)*self.x0[1]
        z_output = traj[:, 2] - input*self.xg[2] - (1-input)*self.x0[2]
        self.x_approx.learn_params(x_input=input, y_output=x_output, flag_s=flag_s)
        self.y_approx.learn_params(x_input=input, y_output=y_output, flag_s=flag_s)
        self.z_approx.learn_params(x_input=input, y_output=z_output, flag_s=flag_s) 
    
    def rollout(self, x_vec, xg=None, flag_s=True):
        if xg is None:
            xg = self.xg
        x_rollout = self.x_approx.rollout(x_vec=x_vec, flag_s=flag_s) + x_vec*xg[0] + (1-x_vec)*self.x0[0]
        y_rollout = self.y_approx.rollout(x_vec=x_vec, flag_s=flag_s) + x_vec*xg[1] + (1-x_vec)*self.x0[1]
        z_rollout = self.z_approx.rollout(x_vec=x_vec, flag_s=flag_s) + x_vec*xg[2] + (1-x_vec)*self.x0[2]
        return np.array([x_rollout, y_rollout, z_rollout]).transpose()
    
    def forward(self, s, flag_s=True):
        '''
        Description: 
            get reproduced positions
        @ param : s{float}           -- s\in[0, 1] 
        @ return: pose_arr{np.array} -- 1x3 np.array 
        '''        
        pos_arr = np.array([self.x_approx.forward(s, flag_s=flag_s),
                            self.y_approx.forward(s, flag_s=flag_s),
                            self.z_approx.forward(s, flag_s=flag_s)
                    ]).reshape(1, 3) + s*self.xg + (1-s)*self.x0
        return pos_arr
    
    def get_vels(self, traj:np.array, sfreq:float, vel_lim = 0.005):
        '''
        Description: 
            get velocities from the input trajectory, currently only
            the cartesian trajectory is considered. 
        @ param : traj{np.array}  -- recrorded trajecoty at sfreq frequency
        @ param : sfreq{float}    -- sampling frequency
        @ param : vel_lim{float}    -- velocity limit
        @ return: vels{np.array}  -- nx3 velocity array, n is the number of recorded positions 
        '''
        new_traj = list()
        max_dist = vel_lim/sfreq
        cur_pos = traj[0]
        for next_pos in traj[1:, :]:
            delta = next_pos - cur_pos
            dist = np.sqrt(np.sum(np.square(delta)))
            if dist > max_dist:
                # interpolation between current pos and next pos
                dir = delta/dist
                num_interpolate = floor(dist/max_dist)
                for i in range(num_interpolate+1):
                    new_traj.append(cur_pos+i*max_dist*dir)
            else:
                new_traj.append(cur_pos)
            cur_pos = next_pos.copy()
        new_traj.append(traj[-1])
        new_traj = np.array(new_traj)
        next = new_traj[1:].copy()
        next = np.concatenate((next, next[-1].reshape(1, 3)), axis=0)
        vels = (next - new_traj) * sfreq
        return vels, new_traj
    
    def get_dist(self, vels:np.array, sfreq:float):
        dist_list = list()
        vel_scalars = np.sqrt(np.sum(np.square(vels), axis=1))
        dist_acc = 0
        dist_list.append(dist_acc)
        for v in vel_scalars:
            dist_acc += v/sfreq 
            dist_list.append(dist_acc)
        return dist_list[:-1], list(vel_scalars)
    
    def fit(self, sampleHz, position_list, flag_visresult=False, vel_lim=0.005):
        pos_arr = np.array(position_list).reshape(-1, 3) 
        vel_3d_arr, pos_arr = self.get_vels(traj=pos_arr, sfreq=sampleHz, vel_lim=vel_lim)
        s_list, _ = self.get_dist(vel_3d_arr, sampleHz)
        total_dist = s_list[-1]
        s_arr = np.array(s_list)/total_dist
        self.learn_params(s_arr, pos_arr)
        
        if flag_visresult:
            plt_time = np.array(range(pos_arr.shape[0]))/sampleHz
            repro_arr = self.rollout(s_arr)
            plt.figure('Position encoding')
            
            plt.subplot(131)  
            plt.plot(plt_time, repro_arr[:, 0], color='blue')
            plt.plot(plt_time, pos_arr[:, 0], linestyle='--', color='black')
            plt.ylabel('position x (m)')
            plt.grid()
            
            plt.subplot(132)
            plt.plot(plt_time, repro_arr[:, 1], color='blue')
            plt.plot(plt_time, pos_arr[:, 1], linestyle='--', color='black')
            plt.ylabel('position y (m)')
            plt.grid()
            
            plt.subplot(133)
            plt.plot(plt_time, repro_arr[:, 2], color='blue')
            plt.plot(plt_time, pos_arr[:, 2], linestyle='--', color='black')
            plt.xlabel('Time (s)')
            plt.ylabel('position z (m)')
            plt.grid()
        
        return total_dist, s_arr, pos_arr

class AL_OrtDMP:
    def __init__(self, n_kfns=31, th=2, k=2, flag_dmp=False):
        self.ZERO_THRESHOLD = 1e-7
        self.approximators = [  TruncGaussApprox(n_kfns, th=th, k=k, flag_dmp=flag_dmp),
                                TruncGaussApprox(n_kfns, th=th, k=k, flag_dmp=flag_dmp),    
                                TruncGaussApprox(n_kfns, th=th, k=k, flag_dmp=flag_dmp)]
        self.eQ0 = None
        self.eQg = None

    def learn_params(self, input:np.array, output:np.array, flag_s=True):
        '''
        Description: 
            fitting the traj. with input, learn the weightings
        @ param : input{np.array} -- Nx1 np.array
        @ param : traj{np.array}  -- Nx3 np.array position info
        '''     
        self.eQ0 = output[0]
        self.eQg = output[-1]
        x_output = output[:, 0] - input*self.eQg[0] - (1-input)*self.eQ0[0]
        y_output = output[:, 1] - input*self.eQg[1] - (1-input)*self.eQ0[1]
        z_output = output[:, 2] - input*self.eQg[2] - (1-input)*self.eQ0[2]
        self.approximators[0].learn_params(x_input=input, y_output=x_output, flag_s=flag_s)
        self.approximators[1].learn_params(x_input=input, y_output=y_output, flag_s=flag_s)
        self.approximators[2].learn_params(x_input=input, y_output=z_output, flag_s=flag_s)
    
    def ort_slerp(self, start_quat:list, end_quat:list, num_inter:float):
        quat_list = list()
        t_list = list(np.arange(0, 1, 1/num_inter))
        for t_inter in t_list:
            quat_list.append(t.quaternion_slerp(start_quat, end_quat, t_inter))
        return quat_list
    
    def get_interpolated_quat(self, traj:np.array, sfreq:float, quat:np.array, vel_lim = 0.005):
        '''
        Description: 
            get velocities from the input trajectory, currently only
            the cartesian trajectory is considered. 
        @ param : traj{np.array}  -- recrorded trajecoty at sfreq frequency
        @ param : sfreq{float}    -- sampling frequency
        @ param : vel_lim{float}    -- velocity limit
        @ return: vels{np.array}  -- nx3 velocity array, n is the number of recorded positions 
        '''
        # new_traj = list()
        new_quat = list()
        max_dist = vel_lim/sfreq
        cur_pos = traj[0]
        cur_quat = quat[0]
        for next_pos, next_quat in zip(traj[1:, :], quat[1:]):
            delta = next_pos - cur_pos
            dist = np.sqrt(np.sum(np.square(delta)))
            if dist > max_dist:
                # interpolation between current pos and next pos
                num_interpolate = floor(dist/max_dist)
                quat_list = self.ort_slerp(cur_quat, next_quat, num_interpolate+1)
                new_quat.extend(quat_list)
            else:
                new_quat.append(cur_quat)
            cur_pos = next_pos.copy()
            cur_quat = next_quat
        new_quat.append(quat[-1])
        return new_quat
    
    def fit(self, s_arr, quat_list, pos_list=None, sampleHz=20, vel_lim=None, flag_visresult=False, flag_check_quat=True):
        if flag_check_quat:  # quat correct
            quat_list = self.correct_quat(quat_list)
        
        # interpolate quaternion trajectory
        if pos_list is not None:
            quat_list = self.get_interpolated_quat( traj=np.array(pos_list), sfreq=sampleHz, 
                                                    quat=np.array(quat_list), vel_lim=vel_lim)
        
        eQ = self.quat_process(quat_list)
        self.Q0_vec, self.Qg_vec = quat_list[0], quat_list[-1]
        self.learn_params(input=s_arr, output=eQ)
        
        if flag_visresult:
            plt_step = np.array(range(eQ.shape[0]))/sampleHz
            repro_eQ = self.rollout(s_arr, flag_get_quat=False)
            plt.figure('eQ encoding')
            plt.subplot(131)
            plt.plot(plt_step, eQ[:, 0], linestyle='--', color='black')
            plt.plot(plt_step, repro_eQ[:, 0], color='blue')
            plt.ylabel('Magnitude eQ[0]')
            plt.grid()
            plt.subplot(132)
            plt.plot(plt_step, eQ[:, 1], linestyle='--', color='black')
            plt.plot(plt_step, repro_eQ[:, 1], color='blue')
            plt.ylabel('Magnitude eQ[1]')
            plt.grid()
            plt.subplot(133)
            plt.plot(plt_step, eQ[:, 2], linestyle='--', color='black')
            plt.plot(plt_step, repro_eQ[:, 2], color='blue')
            plt.xlabel('Time(s)')
            plt.ylabel('Magnitude eQ[2]')
            plt.grid()
        return eQ, np.array(quat_list)
        
    def quat_process(self, q_list):
        # initial and goal quaternion
        q_arr = np.array(q_list)
        self.Q0 = Quaternion(q_arr[0])
        self.Qg = Quaternion(q_arr[-1])
        
        # -- get eQ
        eQ = list()
        for q in q_arr:
            Qc = Quaternion(q) # current quaternion
            eQ_tmp = self.e_Q_metric(self.Qg, Qc)
            eQ.append(eQ_tmp)
        eQ_arr = np.array(eQ)
        # return eQ_arr[:-2:,:]
        return eQ_arr
    
    def rollout(self, x_vec, xg=None, flag_s=True, flag_get_quat=True):
        if xg is None:
            xg = self.eQg
        x_rollout = self.approximators[0].rollout(x_vec=x_vec, flag_s=flag_s) + x_vec*xg[0] + (1-x_vec)*self.eQ0[0]
        y_rollout = self.approximators[1].rollout(x_vec=x_vec, flag_s=flag_s) + x_vec*xg[1] + (1-x_vec)*self.eQ0[1]
        z_rollout = self.approximators[2].rollout(x_vec=x_vec, flag_s=flag_s) + x_vec*xg[2] + (1-x_vec)*self.eQ0[2]
        eQ_arr = np.array([x_rollout, y_rollout, z_rollout]).transpose()
        if flag_get_quat:   # TODO to be accelerated
            quat_list = list()
            for eq in eQ_arr:
                quat = self.exponential_map(0.5*eq).conj()*self.Qg
                quat_list.append(quat.xyzw_vec())
            return np.array(quat_list)
        else:
            return eQ_arr
    
    def forward(self, s, flag_s=True):
        '''
        Description: 
            get reproduced positions
        @ param : s{float}           -- s\in[0, 1] 
        @ return: pose_arr{np.array} -- 1x3 np.array 
        '''        
        eQ = np.array([ self.approximators[0].forward(s, flag_s=flag_s),
                        self.approximators[1].forward(s, flag_s=flag_s),
                        self.approximators[2].forward(s, flag_s=flag_s)
                    ]) + s*self.eQg + (1-s)*self.eQ0
        quat = self.exponential_map(0.5*eQ).conj()*self.Qg
        return quat.xyzw_vec()
    
    def correct_quat(self, quat_list):
        qlast = np.array(quat_list[0])
        for idx in range(1, len(quat_list)):
            if np.linalg.norm(qlast-
                np.array(quat_list[idx])) > 0.5:
                quat_list[idx] = [  -quat_list[idx][0], -quat_list[idx][1],
                                    -quat_list[idx][2], -quat_list[idx][3]]
            qlast =  np.array(quat_list[idx])
        return quat_list
    
    def e_Q_metric(self, q1:Quaternion, q2:Quaternion):
        return 2*self.logarithmic_map(q1*q2.conj())  # 2log(q1*q2.conj()) 
    
    def logarithmic_map(self, q:Quaternion):
        qe_norm = np.linalg.norm(q.epsilon)
        if qe_norm < self.ZERO_THRESHOLD:
            return np.array([0,0,0])
        return acos(q.eta)*q.epsilon/qe_norm
    
    def numerical_derivative(self, data_arr:np.array, Ts:float):
        (_, cols) = data_arr.shape
        arr_last = data_arr[:-2:, :]
        arr_next = data_arr[2::, :]
        darr = (arr_next - arr_last)/(2*Ts)
        darr = np.vstack((np.zeros((1,cols)), darr))
        darr_res = darr.copy()
        for i in range(cols): 
            darr_res[:, i] = self.mid_filter(darr[:, i])
        return darr_res
    
    def mid_filter(self, traj):
        n_traj = len(traj)
        for i in  range(1,n_traj-1,1):
            if traj[i] >=  traj[i-1]:
                seq = [traj[i-1], traj[i]]
            else:
                seq = [traj[i], traj[i-1]]
            if traj[i+1] >=  seq[0]:
                if traj[i+1]  >= seq[1]:
                    mid = seq[1]
                else:
                    mid = traj[i+1]
            else:
                mid = seq[0]
            traj[i] = mid
        return traj
    
    def exponential_map(self, r:np.array):
        r_norm = np.linalg.norm(r)
        if r_norm < self.ZERO_THRESHOLD:
            return Quaternion(1, np.array([0, 0, 0]))
        return Quaternion(cos(r_norm), sin(r_norm)/r_norm*r)


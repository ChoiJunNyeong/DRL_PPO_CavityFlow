# 2021-09-29
from ftplib import FTP
import numpy as np
import os

class communication:
    def login(self,num):
        self.ftp = FTP()
        self.ftp.connect(host='10.74.18.5'+str(num))
        self.ftp.login(user='JUN', passwd='123456')
        self.ftp.sendcmd('PASV') #passive mode로 사용한다고 전달
        self.home = '/home1/JUN/Research_Cavity_DRL/Phase02_MainSimulation/02MainSimulation/0000Control_Cavity_LES_RE12000_LzD1/00DATA_05DRL'
        # self.home = '/home1/JUN/Research_Cavity_DRL/Phase02_MainSimulation/02MainSimulation/0000Control_Cavity_LES_RE12000/00DATA_05DRL'
        self.ftp.cwd(self.home) #home directory로 이동
        
    def server_cwd(self):
        self.ftp.cwd(self.home) #디렉토리 변경(branch+number)
        
    def server_list(self):   #현재 디렉토리 파일 목록
        return self.ftp.nlst() #server의 파일목록

    def client_list(self):
        return os.listdir(path=os.getcwd())  #os.getcwd의 파일목록
    
    def download(self,a,b):
        # server의 a 파일을 client에 b의 이름으로 download
        # 이후 a 파일은 server에서 삭제
        with open('./'+b, 'wb') as temp: #open(filename,"wb"),wb:binary형태 write
            self.ftp.retrbinary('RETR '+a,temp.write) #a파일을 client temp에 저장.
        self.ftp.delete(a)   #a는 삭제
        
    def upload(self,a,b):
        # client의 a 파일을 server에 b의 이름으로 upload
        # 이후 a 파일은 client에서 삭제
        with open('./'+a, 'rb') as temp:  #open(filename,"rb"),rb:binary형태 read, t:test형태
            self.ftp.storbinary('STOR '+b, temp) #temp파일을 b이름으로 upload
        os.remove('./'+a) #a는 삭제

    def write(self, contents):
        # contents = contents.tolist()
        # contents는 list형태
        with open('./'+'made_in_client.txt', 'w') as temp:
            for i in range(len(contents)):  #range(0,len(contents))
                temp.write(str(contents[i])) #parr이 1개면 1개만 받음.
                if i != len(contents)-1:
                    temp.write('\n')
    
    def read(self,num):
        with open('server_to_client_'+str(num)+'.txt', 'r') as temp:
            states = []
            while True: #줄 끝에 도달하면 break.
                info = temp.readline().strip('\n')
                if not info: break
                info = info.split(':')
                if len(info) != 1:
                    name,val = info[0].replace(' ',''),float(info[1].replace(' ',''))
                    if name == 'STATE':
                        states.append(val)
                    elif name == 'REWARD':
                        reward = val
                    elif name == 'DONE':
                        done = int(val)
        os.remove('server_to_client_'+str(num)+'.txt')
        return np.array(states), reward, done
    
    def read_init(self,num,s_dim,a_dim,flag):
        with open('initializer_'+str(num)+'.txt', 'r') as temp:
            states = []
            while True: 
                info = temp.readline().strip('\n') #strip : 줄 끝 줄바꿈 문자 없앰, split: 문자 기준으로 나누기.
                if not info: break
                info = info.split(':')
                if len(info) != 1:
                    name,val = info[0].replace(' ',''), float(info[1].replace(' ',''))
                    if name == 'STATE':
                        states.append(val)            

            if flag == 1:
                before = ['ep_#','action_#','machine_#']
                after = ['Reward','Done']
                with open('history.txt','a') as temp:
                    temp.write('%')
                    for item in before:
                        temp.write("{0:>12}".format(item)) #{}안의 숫자는 format인덱스임.
                    for i in range(1,s_dim+1):
                        temp.write("{0:>12}".format("STATE_"+str(i))) #0:>?는 왼쪽정렬, 0:<?는 오른쪽 정렬, 0:^?는 가운데 정렬 
                    for i in range(1,a_dim+1):
                        temp.write("{0:>12}".format("ACTION_"+str(i)))
                    for item in after:
                        temp.write("{0:>12}".format(item))
                    temp.write('\n')
        os.remove('initializer_'+str(num)+'.txt')
            
        return np.array(states)
    
    def read_env_param(self): #젤 처음 env_param읽기
        with open('env_param.txt','r') as temp:
            output = []            
            while True:
                line = temp.readline().strip('\n').split()
                if not line: break
                for item in line:
                    if item != '':
                        output.append(int(item))
            
        params = ['% max_episodes', '% action_num', '% state_dim', 
                  '% action_dim', '% Update_epi_num', '% K_epochs times',
                  '% Simulation Parallel']
        with open('history.txt','a') as temp:  #history에 기록
            for i in range(len(params)):
                temp.write("{} : {}".format(params[i],output[i]))
                print("{} : {}".format(params[i],output[i]))
                if i != len(params)-1:  #한칸씩 띄워 쓰기 
                    temp.write("\n")
                else:
                    temp.write("\n\n")   #끝나면 두칸 띄워 쓰기
                    print('')
            temp.write('\n')
        os.remove('env_param.txt')     #env_param읽고나서 client에서 삭제.
        return np.array(output)
            
    def history_ep_ts(self, ep, timestep,machine):
        with open('history.txt', 'a') as temp:
            temp.write(" {0:12}{1:12}{2:12}".format(ep,timestep,machine))
        
    def history_state(self, states):
        with open('history.txt', 'a') as temp:
            for item in states:
                temp.write("{0:12.6f}".format(item))
                
    def history_action(self, actions):
        with open('history.txt', 'a') as temp:
            for item in actions:
                temp.write("{0:12.6f}".format(item))
        
    def history_reward_done(self, reward, done):
        with open('history.txt', 'a') as temp:
            temp.write("{0:12.3f}{1:12}".format(reward,int(done)))
            temp.write("\n")
            
    def logging(self, i, i_episode, action_step, log_avg_reward):
        log_dir = "DRL(PPO)_logs" + '/' + "Cavity" 
        log_f_name = log_dir + '/PPO_' + "Cavity" + "_log_" + str(i) + ".csv"
        with open(log_f_name, 'a') as temp:
            temp.write('{},{},{}\n'.format(i_episode, action_step, log_avg_reward))
    
    
    # def history_write(self, string):
    #     with open('history.txt', 'a') as temp:
    #         temp.write(string)
            
    # def clear_before_start(self, sv_txts, cl_txts):
    #     cl_lists = os.listdir()
    #     sv_lists = self.ftp.nlst()
    #     for name in cl_txts:
    #         if name in cl_lists:
    #             os.remove('./'+name)
    #     for name in sv_txts:
    #         if name in sv_lists:
    #             self.ftp.delete(name)
                
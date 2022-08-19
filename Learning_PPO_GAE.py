import sys
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
# import gym
import numpy as np
import os
from Learning_python_fortran import communication as Commu
import time
import datetime
# import fish_evasion as fish
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #cpu or gpu로 학습하기.

class Memory:
    # structure to store data for each update
    def __init__(self):
        self.actions = [] #빈 list 설정
        self.states = []
        self.states_p = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:] #[:]이렇게 list설정하면 똑같지만 다른 주소를 가리키면서 복사됨.
        del self.states[:]
        del self.states_p[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):  #nn 모듈 상속
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__() #nn.module의 init를 상속받게 함. 항상해줘야함!
        # action mean range -1 to 1
        self.actor = nn.Sequential(        #nn.linear 편하게 구성.
                nn.Linear(state_dim, 512),  #input에서 1st layer로 
                nn.ReLU(),                 #1st layer의 actication fuction
                nn.Linear(512, 512),         #1st layer에서 2nd layer로 (neuron 갯수)
                nn.ReLU(),                 #2nd layer의 actication fuction
                nn.Linear(512, action_dim), #2nd layer에서 output layer로
                # nn.Tanh()                  #output layer의 actication fuction
                )        
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)           #value니깐 output이 항상 1이다.
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device) #full : 후자를 전자로 구성.
        self.action_dim = action_dim
        
    def forward(self):         
        raise NotImplementedError          #무조건 구현되어야하는 곳이 안되고 넘어갈때 오류 발생시키려고!
    
    def act(self, state):
        # operations per decision time step
        action_mean = self.actor(state)                   #diag: 값들을 대각행렬로 만들어줌.(Stocahstic policy를 위해 사용, 연속공간에서 주로 사용됨.)
        action_mean = torch.clamp(action_mean, -2.0, 2.0)  
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  #unsqueeze : 두번째 차원에 1차원을 추가. ex) scalar(3)이면 tensor([1,3])이됨.        
        dist = MultivariateNormal(action_mean, cov_mat)   #출력한 action 확률을 토대로 확률밀도함수를 만듬
        action = dist.sample()                            #확률밀도함수에서 action을 sampling함 (하나의 action이 추출된다.)
        action = torch.clamp(action, -2.0, 2.0) 
        action_logprob = dist.log_prob(action)            #log-likelihood
        print("Action mean[1~5] :",action_mean) #action 값.
        
        # Write action mean
        a_mean_record = action_mean.tolist()
        with open('DRL(PPO)_logs/Cavity/Action_mean.txt', 'a') as temp:
            temp.write(str(a_mean_record)) 
            temp.write("\n")   
            
        return action.detach(), action_logprob.detach()
    
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # for single action continuous environments
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy  

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr_actor, lr_critic, gamma, lmbda, K_epochs, eps_clip):
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])        
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()   #loss 함수로 mse를 사용.
    
    def select_action(self, state, memory):    

        with torch.no_grad():
            action, action_logprob = self.policy_old.act(torch.FloatTensor(state))   #floattensor : tensor로 만들어줌.
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.detach().cpu().numpy().flatten() #여기서 action값을 뽑음. (flatten : 1차원 형태의 배열로 변환.)
    
    def update(self, memory): 
    
        # Ganeralized advantages Estimations
        s = torch.FloatTensor(memory.states)
        s_prime = torch.FloatTensor(memory.states_p)
        r = torch.FloatTensor(memory.rewards)
        done_mask = torch.FloatTensor(memory.is_terminals)
        a = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device) 
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):  #update를 할때 k_epochs만큼 나눠서 update를함. 그래서 천천히 policy가 좋은방향으로 바뀜.(PPO 특징)
            td_target = r + self.gamma * self.policy.critic(s_prime) * done_mask  
            delta = td_target - self.policy.critic(s)        
            delta = delta.detach().numpy()
            advantages_lst = []
            advantages = 0.0
            for delta_t, done_mask_t in zip(reversed(delta), reversed(done_mask)):
                if done_mask_t == 0.0:
                    advantages = 0.0
                advantages = self.gamma * self.lmbda * advantages + delta_t[0]
                advantages_lst.insert(0, [advantages])        
            advantages = torch.squeeze(torch.tensor(advantages_lst, dtype=torch.float))  #squeeze : 차원 1개를 없앰. 여기서 차원없애서 update에서 차원이 맞아짐.
            
            # Normalizaing the advantagess  
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(s, a)
            
            # match state_values tensor dimensions with rewards tensor           
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, torch.squeeze(td_target.detach())) - 0.01*dist_entropy #critic은 경사하강, actor는 경사상승. 
            
            # take gradient step
            self.optimizer.zero_grad() #gradient값을 계산하기전 초기화 시키는것.
            loss.mean().backward()     #역전파로 gradient 계산.
            self.optimizer.step()      #weight & bias 계산 및 개선.
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict()) # weight & bias를 저장하고 복사.(모델 저장할때도 사용함.)
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))



def main(k,path,network_path):
    # path = "DRL(PPO)_result/"+"Cavity"+'/{}'.format(k) #learning_result폴더 만들어서 저장.           
    # if not os.path.exists(path):    # client 경로의 file 확인.
    #     os.makedirs(path)          
    if 'env_param.txt' in os.listdir():
        os.remove('env_param.txt')       
    commu = Commu()        
    commu.login(1)
    commu.server_cwd()
    while 'env_param.txt' not in commu.server_list(): #server file 확인.
        time.sleep(2) # 파일 생성될때까지 2초 대기.
        print("waiting for download environment param from server")
    commu.download('env_param.txt', 'env_param.txt')
    [max_episodes, action_num, state_dim, action_dim, update_epi_num, K_epochs, parr] = commu.read_env_param()
    
    state1, state2, state3, state4, state5 = [], [], [], [], []
    p_state1, p_state2, p_state3, p_state4, p_state5 = [], [], [], [], []
    update_action_num = update_epi_num*action_num      # update policy every n timesteps, update_epi_num : #회 epi당 업데이트

    # Read state normalization parameters : mean, std 
    input_mean, input_std = [], []
    input_mean, input_std = commu.read_state_normalized()
    
    #########################logging###############################   
    # log files for multiple runs are NOT overwritten    
    log_dir = "DRL(PPO)_logs" + '/' + "Cavity"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    # get number of log files in log directory
    for i in range(1,parr+1):        
        #### create new log file for each run 
        log_f_name = log_dir + '/PPO_' + "Cavity" + "_log_" + str(i) + ".csv"
        print("current logging run number for " + "Cavity" + " : ", i)
        print("logging at : " + log_f_name)            
        log_f = open(log_f_name,"a")
        log_f.write('Episode,Action_num,Reward\n')
        log_f.close()
        if i == parr:
            log_f_name_total = log_dir + '/PPO_Cavity_log_total.csv'
            print("current logging run number for " + "Cavity" + " : total")
            print("logging at : " + log_f_name_total)              
            log_f_total = open(log_f_name_total,"a")
            log_f_total.write('Episode,Reward\n')
            log_f_total.close() 
        
    log_running_episodes = 0
    log_freq = action_num * 1

    ##########################Save model############################
    save_dir = "DRL(PPO)_preTrained" + '/' + "Cavity"
    if not os.path.exists(save_dir):
          os.makedirs(save_dir)

    ####################### Hyperparametesrs #######################
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lmbda = 0.97                # gae lambda
    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic    
    
    
    memory1, memory2, memory3, memory4, memory5 = Memory(), Memory(), Memory(), Memory(), Memory() #Class Memory 사용.
    total_memory = Memory()
    
    ppo = PPO(state_dim, action_dim, action_std, lr_actor, lr_critic, gamma, lmbda, K_epochs, eps_clip)
    
    # if network_path != None:
    #     network = torch.load(network_path)
    #     ppo.policy_old.load_state_dict(network.state_dict())
    #     ppo.policy.load_state_dict(network.state_dict())
    # print(datetime.datetime.now(), '학습 시작')
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    update_count = 1
    action_step = update_action_num*update_count
    
    ##########################Load model############################    
    if network_path != None:
        ppo.load(save_dir+'/update_{}th.pth'.format(update_count))
        print('load model : '+save_dir+'/update_{}th.pth'.format(update_count))

    action1, action2, action3, action4, action5 = 0, 0, 0, 0, 0
    reward1, reward2, reward3, reward4, reward5 = 0, 0, 0, 0, 0
    done1, done2, done3, done4, done5 = 0, 0, 0, 0, 0
    reward = 0.0
    flag = 1
    result = []
    for i in range(1,parr+1):
        result.append(i)        
    #==========================================================================
    #                       Training loop(episode)
    #==========================================================================
    for i_episode in range(update_count*4+1, max_episodes+1): #episode loop
        current_ep_reward1, current_ep_reward2, current_ep_reward3, current_ep_reward4, current_ep_reward5 = 0, 0, 0, 0, 0         
        log_running_episodes += 1        
        
        print("")
        print(datetime.datetime.now(), i_episode,'/',max_episodes,'시작...')
        # if i_episode == 1: #1 episode에만 initializer.txt 읽음    
    #--------------initial file download (state) from server---------------
        for i in result:    #episode 시작할때마다 반복하니 if문으로 바꿔주자!
            while True:     #parr에 initializer.tex가 생길때까지 기다리고, 생기면 download하자.
                time.sleep(1)
                commu.login(i)
                commu.server_cwd()
                if 'initializer.txt' in commu.server_list(): #initializer 파일 다운로드 from server
                    commu.download('initializer.txt','initializer_'+str(i)+'.txt') #initializer_str(v)이름으로 다운로드
                    break                    
        #initial file read (state)                
        for v in range(1,parr+1):  #초기 데이터 read
            if v == 1:
                state1 = commu.read_init(v,state_dim,action_dim,flag)
            elif v == 2:
                state2 = commu.read_init(v,state_dim,action_dim,flag)
            elif v == 3:
                state3 = commu.read_init(v,state_dim,action_dim,flag)
            elif v == 4:
                state4 = commu.read_init(v,state_dim,action_dim,flag)
            elif v == 5:
                state5 = commu.read_init(v,state_dim,action_dim,flag)
            flag = 0
    #----------------------------------------------------------------------

        #======================================================================
        #                Action number per 1 episode   
        #======================================================================
        for t in range(1,action_num+1): #action 횟수로 알아보게 수정!!
            action_step +=1 #cummulative value
            
            # State normalization
            state1 = (state1 - input_mean) / (input_std + 1e-10)       
            state2 = (state2 - input_mean) / (input_std + 1e-10)
            state3 = (state3 - input_mean) / (input_std + 1e-10)
            state4 = (state4 - input_mean) / (input_std + 1e-10)
            state5 = (state5 - input_mean) / (input_std + 1e-10)
            
            # Running policy_old:
            for v in range(1,parr+1):
                commu.login(v)
                commu.server_cwd()
                if v == 1:
                    action1 = ppo.select_action(state1, memory1)   #memory에 state,action 저장
                    commu.write(action1)
                    commu.upload('made_in_client.txt', 'cl_to_sv.txt')  #action 범위지정할때 여기서 해야할거같은데?
                elif v == 2:
                    action2 = ppo.select_action(state2, memory2)
                    commu.write(action2)
                    commu.upload('made_in_client.txt', 'cl_to_sv.txt')
                elif v == 3:
                    action3 = ppo.select_action(state3, memory3)
                    commu.write(action3)
                    commu.upload('made_in_client.txt', 'cl_to_sv.txt')                    
                elif v == 4:
                    action4 = ppo.select_action(state4, memory4)
                    commu.write(action4)
                    commu.upload('made_in_client.txt', 'cl_to_sv.txt')
                elif v == 5:
                    action5 = ppo.select_action(state5, memory5)
                    commu.write(action5)
                    commu.upload('made_in_client.txt', 'cl_to_sv.txt')                    
                                  
            print("Action step :",action_step, ",", "Action value[1~5] :",action1, action2, action3, action4, action5) #action 값.
            
            #file download (state,reward) from server  
            for i in result:
                while True:
                    time.sleep(1)
                    commu.login(i)
                    commu.server_cwd()
                    if 'made_in_server.txt' in commu.server_list():
                        commu.download('made_in_server.txt','server_to_client_'+str(i)+'.txt')   
                        break
                    
            #file read (state, reward) and stack in each memory[i]     
            for v in range(1,parr+1):
                if v == 1:
                    state1, reward1, done1 = commu.read(v)      
                    p_state1 = state1.copy()
                    p_state1 = (p_state1 - input_mean) / (input_std + 1e-10)
                    
                    memory1.rewards.append([reward1])
                    memory1.is_terminals.append([done1])
                    memory1.states_p.append(p_state1)
                elif v == 2:
                    state2, reward2, done2 = commu.read(v)
                    p_state2 = state2.copy()
                    p_state2 = (p_state2 - input_mean) / (input_std + 1e-10)
                    
                    memory2.rewards.append([reward2])
                    memory2.is_terminals.append([done2])
                    memory2.states_p.append(p_state2)
                elif v == 3:
                    state3, reward3, done3 = commu.read(v)
                    p_state3 = state3.copy()
                    p_state3 = (p_state3 - input_mean) / (input_std + 1e-10)
                    
                    memory3.rewards.append([reward3])
                    memory3.is_terminals.append([done3])  
                    memory3.states_p.append(p_state3)
                elif v == 4:
                    state4, reward4, done4 = commu.read(v)
                    p_state4 = state4.copy()
                    p_state4 = (p_state4 - input_mean) / (input_std + 1e-10)
                    
                    memory4.rewards.append([reward4])
                    memory4.is_terminals.append([done4])   
                    memory4.states_p.append(p_state4)
                elif v == 5:
                    state5, reward5, done5 = commu.read(v)
                    p_state5 = state5.copy()
                    p_state5 = (p_state5 - input_mean) / (input_std + 1e-10)
                    
                    memory5.rewards.append([reward5])
                    memory5.is_terminals.append([done5])   
                    memory5.states_p.append(p_state5)
            print("done value[1~5] :",done1, done2, done3, done4, done5) #action 값.
            #==================================================================
            #logging : history file에 값들을 저장. 
            #==================================================================
            current_ep_reward1 += reward1
            current_ep_reward2 += reward2            
            current_ep_reward3 += reward3
            current_ep_reward4 += reward4  
            current_ep_reward5 += reward5  
            # every time step
            for v in range(1,parr+1):
                if v == 1:
                    commu.history_ep_ts(i_episode, t, v)
                    commu.history_state(p_state1)                #s(t+1)을 저장함.
                    commu.history_action(action1)
                    commu.history_reward_done(reward1, done1)
                elif v == 2:
                    commu.history_ep_ts(i_episode, t, v)
                    commu.history_state(p_state2)
                    commu.history_action(action2)
                    commu.history_reward_done(reward2, done2)
                elif v == 3:
                    commu.history_ep_ts(i_episode, t, v)
                    commu.history_state(p_state3)
                    commu.history_action(action3)
                    commu.history_reward_done(reward3, done3)
                elif v == 4:
                    commu.history_ep_ts(i_episode, t, v)
                    commu.history_state(p_state4)
                    commu.history_action(action4)
                    commu.history_reward_done(reward4, done4)
                elif v == 5:
                    commu.history_ep_ts(i_episode, t, v)
                    commu.history_state(p_state5)
                    commu.history_action(action5)
                    commu.history_reward_done(reward5, done5)
            #logging avg reward        
            if action_step % log_freq == 0:    
                for i in range(1,parr+1):
                    if i == 1:
                        log_avg_reward1 = current_ep_reward1 / log_running_episodes
                        log_avg_reward1 = round(log_avg_reward1, 4)
                        commu.logging(i,i_episode,action_step,log_avg_reward1)
                        commu.logging_total(parr*(i_episode-1)+1,log_avg_reward1)
                    elif i == 2:
                        log_avg_reward2 = current_ep_reward2 / log_running_episodes
                        log_avg_reward2 = round(log_avg_reward2, 4)
                        commu.logging(i,i_episode,action_step,log_avg_reward2)
                        commu.logging_total(parr*(i_episode-1)+2,log_avg_reward2)
                    elif i == 3:
                        log_avg_reward3 = current_ep_reward3 / log_running_episodes
                        log_avg_reward3 = round(log_avg_reward3, 4)
                        commu.logging(i,i_episode,action_step,log_avg_reward3)   
                        commu.logging_total(parr*(i_episode-1)+3,log_avg_reward3)                        
                    elif i == 4:
                        log_avg_reward4 = current_ep_reward4 / log_running_episodes
                        log_avg_reward4 = round(log_avg_reward4, 4)
                        commu.logging(i,i_episode,action_step,log_avg_reward4)    
                        commu.logging_total(parr*(i_episode-1)+4,log_avg_reward4)                        
                    elif i == 5:
                        log_avg_reward5 = current_ep_reward5 / log_running_episodes
                        log_avg_reward5 = round(log_avg_reward5, 4)
                        commu.logging(i,i_episode,action_step,log_avg_reward5)
                        commu.logging_total(parr*(i_episode-1)+5,log_avg_reward5)
                log_running_episodes = 0
                current_ep_reward1 = 0
                current_ep_reward2 = 0            
                current_ep_reward3 = 0
                current_ep_reward4 = 0
                current_ep_reward5 = 0
            #==================================================================        
            # Update DRL model
            #==================================================================
            if action_step % update_action_num == 0:
                update_count += 1
                start = datetime.datetime.now()
                print('\n',start, update_count, '번째 업데이트 시작...')
                for v in range(1,parr+1):
                    if v == 1:
                        total_memory.actions += memory1.actions
                        total_memory.states += memory1.states
                        total_memory.states_p += memory1.states_p                        
                        total_memory.logprobs += memory1.logprobs
                        total_memory.rewards += memory1.rewards
                        total_memory.is_terminals += memory1.is_terminals
                        memory1.clear_memory()
                    elif v == 2:
                        total_memory.actions += memory2.actions
                        total_memory.states += memory2.states
                        total_memory.states_p += memory2.states_p   
                        total_memory.logprobs += memory2.logprobs
                        total_memory.rewards += memory2.rewards
                        total_memory.is_terminals += memory2.is_terminals
                        memory2.clear_memory()
                    elif v == 3:
                        total_memory.actions += memory3.actions
                        total_memory.states += memory3.states
                        total_memory.states_p += memory3.states_p                           
                        total_memory.logprobs += memory3.logprobs
                        total_memory.rewards += memory3.rewards
                        total_memory.is_terminals += memory3.is_terminals
                        memory3.clear_memory()                       
                    elif v == 4:
                        total_memory.actions += memory4.actions
                        total_memory.states += memory4.states
                        total_memory.states_p += memory4.states_p                           
                        total_memory.logprobs += memory4.logprobs
                        total_memory.rewards += memory4.rewards
                        total_memory.is_terminals += memory4.is_terminals
                        memory4.clear_memory()    
                    elif v == 5:
                        total_memory.actions += memory5.actions
                        total_memory.states += memory5.states
                        total_memory.states_p += memory5.states_p                           
                        total_memory.logprobs += memory5.logprobs
                        total_memory.rewards += memory5.rewards
                        total_memory.is_terminals += memory5.is_terminals
                        memory5.clear_memory()  
                        
                        
                ppo.update(total_memory)
                total_memory.clear_memory()
                end = datetime.datetime.now()
                print(end, update_count, '번째 업데이트 끝 .. 소요시간:', end-start)
        #==================================================================
        # Save model after epi_freq 
        #==================================================================
        if i_episode % update_epi_num == 0:
            ppo.save(save_dir+'/update_{}th.pth'.format(update_count))
            # torch.save(ppo.policy, save_dir+'/update_{}th.pth'.format(update_count)) # state_dict() 해야하나??? 수정요망.                    
    
            # running_reward += reward
            # done = 1
            # for v in range(1,parr+1):
            #     if v == 1:
            #         done = done*done1
            #     elif v == 2:
            #         done = done*done2
            # if done == 1:
            #     break
        # avg_length += t
        
        # ------------------------------------------------------------------
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_forwardWoPos_solved_{}.pth'.format(env_name))
        #     break
        # ------------------------------------------------------------------
    
        # save every after updates
        # if i_episode % update_epi_num == 0:
        #     torch.save(ppo.policy, path+'/update_{}th.pth'.format(update_count)) # state_dict() 해야하나??? 수정요망.
        # # torch.load : 모델 불러오기
        # # ------------------------------------------------------------------
        # # logging
        # if i_episode % log_interval == 0:
        #     avg_length = int(avg_length/log_interval)
        #     running_reward = ((running_reward/log_interval))
        #     print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
        #     running_reward = 0
        #     avg_length = 0
        # # ------------------------------------------------------------------

if __name__ == '__main__':
    # print('direction150')
    # # training for 25 times
    # for k in range(1,25):        
    #     main(k)
    today = datetime.datetime.today()
    path = today.strftime("%Y-%m-%d %H%M")
    network_path = True
    
    # 이어서 학습하시려면 아래 줄에 파일 경로 입력 후 주석 해제 하시면 됩니다
    # network_path = 'C:\\Users\\PC\\Desktop\\졸업연구\\최종본 저장\\tttest.pth'
    
    main(1,path,network_path)

# wrap angles about a given center
# def angle_normalize(x,center = 0):
#     return (((x+np.pi-center) % (2*np.pi)) - np.pi+center)

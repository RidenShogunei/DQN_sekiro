import numpy as np
import find_blood_location
import cv2
import time
import directkeys
from getkeys import key_check
import torch
from restart import restart
from DQN_sekiro_pytroch import DQN
import grabscreen
import pickle
import random
from torch import nn
import torch.optim as optim
import sys
from torch.utils.tensorboard import SummaryWriter
import signal
# 保存训练信息
def handle_signal(signal, frame):
    print('Training interrupted. Saving training information...')
    save_training_info(current_net, episode)
    sys.exit(0)



# 保存训练信息
def save_training_info(net1, net2, episode):
    training_info = {
        'net1': net1.state_dict(),
        'net2': net2.state_dict(),
        'episode': episode
    }
    with open('training_info.pkl', 'wb') as f:
        pickle.dump(training_info, f)

# 加载训练信息
def load_training_info(net1, net2):
    try:
        with open('training_info.pkl', 'rb') as f:
            training_info = pickle.load(f)
            net1.load_state_dict(training_info['net1'])
            net2.load_state_dict(training_info['net2'])
            episode = training_info['episode']
            return net1, net2, episode
    except FileNotFoundError:
        return net1, net2, 0

def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused


def take_action(action):
    if action == 0:  # n_choose
        print("执行了0")
        pass
    elif action == 1:  # j
        print("执行了1")
        directkeys.attack()
    elif action == 2:  # k
        print("执行了2")
        directkeys.jump()
    elif action == 3:  # m
        print("执行了3")
        directkeys.defense()
    elif action == 4:  # r
        print("执行了4")
        directkeys.dodge()
    elif action == 5:  # r
        print("执行了5")
        directkeys.skill()
    elif action == 6:  # r
        print("执行了5")
        directkeys.skill()


# 定义一个函数，用于获取当前的敌人和自身的血量信息
def get_blood_data():
    self_screen_gray = cv2.cvtColor(grabscreen.grab_screen(self_blood_window), cv2.COLOR_BGR2GRAY)
    boss_screen_gray = cv2.cvtColor(grabscreen.grab_screen(boss_blood_window), cv2.COLOR_BGR2GRAY)
    boss_blood = find_blood_location.boss_blood_count(boss_screen_gray)
    self_blood = find_blood_location.self_blood_count(self_screen_gray)
    return self_blood, boss_blood


# 定义一个函数，用于将当前血量信息转换为模型输入的特征向量
def preprocess_blood_data(self_blood, boss_blood):
    # 在这个示例中，将敌人和自身的血量信息简单地串联成一个特征向量
    # 如果需要进行更复杂的特征处理，可以在这里进行修改
    feature_vector = torch.tensor([self_blood, boss_blood], dtype=torch.float32)
    return feature_vector


def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break):
    # get action reward
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    if next_self_blood < 3:  # self dead
        reward = -100
        done = 1
        return reward, done, stop, emergence_break
    elif boss_blood - next_boss_blood > 1:
        reward = 90
        done = 0
        return reward, done, stop, emergence_break
    elif self_blood - next_self_blood <= 10:
        reward = 10
        done = 0
        return reward, done, stop, emergence_break

    elif self_blood - next_self_blood > 10:
        reward = -80
        done = 0
        return reward, done, stop, emergence_break


    elif next_boss_blood < 3:
        reward = 100
        done = 0
        return reward, done, stop, emergence_break

    else:
        reward = 0
        done = 0
        return reward, done, stop, emergence_break


def preprocess_and_normalize_image():
    gray_image = cv2.cvtColor(grabscreen.grab_screen(window_size), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

    normalized_image = torch.from_numpy(edges).float() / 255.0
    return normalized_image


def soft_update(target_net, current_net, tau):
    for target_param, current_param in zip(target_net.parameters(), current_net.parameters()):
        target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_signal)
    Totle_reward = 0
    Death_time = 0
    image_channels = 24
    Inputsize = 2
    Outputsize = 6
    gamma = 0.6  # 折扣因子
    tau = 0.01  # 软更新的权重
    current_net = DQN(image_channels, Inputsize, Outputsize)
    target_net = DQN(image_channels, Inputsize, Outputsize)
    current_net, target_net, episode = load_training_info(current_net, target_net)
    print(episode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_net.to(device)
    target_net.to(device)
    action_size = 6
    num_episodes = 3000000
    save_interval = 400
    paused = True
    buffer_images = []
    buffer_features = []
    batch_size =24
    current_buffer_size = 0
    window_size = (700, 100, 1204, 1050)
    self_blood_window = (120, 1025, 700, 1040)
    boss_blood_window = (120, 135, 550, 155)
    # 定义经验回放缓冲区的容量
    buffer_capacity = 100000
    experience_buffer = []
    emergence_break = 0
    stop = 0
    optimizer = optim.Adam(current_net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()  # 使用均方误差损失
    boss_time = 0
    pause_game(paused)
    directkeys.lock_vision()
    writer = SummaryWriter('log')
    for episode in  range(episode, num_episodes):
        print("这是第",episode,"轮")
        save_training_info(current_net, target_net, episode)
        '''这里是正常实验'''
        self_blood, boss_blood = get_blood_data()
        if self_blood < 3:
            restart()
            directkeys.lock_vision()
            Death_time += 1
            print("死亡次数:", Death_time)
            continue

        if boss_blood < 3:
            boss_time = boss_time + 1
            continue

        if boss_time == 3:
            model_save_path = f'model_epoch_{episode}.pth'
            torch.save(current_net.state_dict(), model_save_path)
            print(f'Model weights saved at episode {episode} to {model_save_path}')
            sys.exit()

        while current_buffer_size < batch_size:
            image_data = preprocess_and_normalize_image()
            feature_vector = preprocess_blood_data(self_blood, boss_blood)
            buffer_images.append(image_data)
            buffer_features.append(feature_vector)
            current_buffer_size += 1

        batch_images = torch.stack(buffer_images)
        batch_features = torch.stack(buffer_features)
        batch_images = batch_images.to(device)
        batch_features = batch_features.to(device)
        # 将图像和特征向量添加到缓冲区
        # print("batch_images", batch_images.shape)
        # print("batch_features", batch_features.shape)
        q_values = current_net(batch_images, batch_features)
        buffer_images = []
        buffer_features = []
        current_buffer_size = 0
        # print("q_values.shape", q_values.shape)
        action = np.random.randint(action_size) if np.random.rand() < 0.1 else torch.argmax(q_values).item()
        # print("torch.argmax(q_values).item()", torch.argmax(q_values).item())
        take_action(action)
        # print("action", action)
        next_self_blood, next_boss_blood = get_blood_data()

        reward, done, stop, emergence_break = action_judge(boss_blood, next_self_blood, self_blood, next_boss_blood,
                                                           stop, emergence_break)
        experience = (batch_images, batch_features, action, reward, next_self_blood, next_boss_blood, done)
        experience_buffer.append(experience)
        if len(experience_buffer) > buffer_capacity:
            experience_buffer.pop(0)
        Totle_reward = Totle_reward + reward
        writer.add_scalar("Totle_reward", Totle_reward, episode)
        while current_buffer_size < batch_size:
            image_data = preprocess_and_normalize_image()
            feature_vector = preprocess_blood_data(self_blood, boss_blood)
            buffer_images.append(image_data)
            buffer_features.append(feature_vector)
            current_buffer_size += 1
        batch_images = torch.stack(buffer_images)
        batch_features = torch.stack(buffer_features)
        batch_images = batch_images.to(device)
        batch_features = batch_features.to(device)
        target_q_values = current_net(batch_images, batch_features)
        buffer_images = []
        buffer_features = []
        current_buffer_size = 0
        target_q_value = reward + gamma * torch.max(target_q_values)
        predicted_q_value = q_values[action]
        target_q_value = target_q_value.to(device)
        predicted_q_value = predicted_q_value.to(device)
        loss = loss_function(predicted_q_value, target_q_value)
        writer.add_scalar('Loss', loss, episode)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        soft_update(target_net, current_net, tau)
        '''下面是使用经验池的数据进行训练的'''
        print("经验池计算")
        # 从经验池中采样一批经验
        experiences = random.sample(experience_buffer, 1)
        # 解压缩经验
        batch_images, batch_features, actions, rewards, next_self_blood, next_boss_blood, dones = zip(*experiences)
        # 将采样的经验转换为张量
        batch_images = torch.stack(batch_images).to(device)
        batch_features = torch.stack(batch_features).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_self_blood = torch.tensor(next_self_blood).to(device)
        next_boss_blood = torch.tensor(next_boss_blood).to(device)
        dones = torch.tensor(dones).to(device)
        # 在当前网络上计算 Q 值
        q_values = current_net(batch_images, batch_features)

        # 根据采样的动作索引获取预测的 Q 值
        predicted_q_values = torch.max(q_values)
        # 在目标网络上计算下一个状态的 Q 值
        with torch.no_grad():
            next_q_values = target_net(batch_images, batch_features)
            next_q_max = torch.max(next_q_values)
            target_q_values = rewards + gamma * next_q_max

        # 计算损失和优化步骤
        #print("predicted_q_values",predicted_q_values.shape)
        #print("target_q_values",target_q_values.shape)
        target_q_value = target_q_value.to(device)
        predicted_q_values = torch.unsqueeze(predicted_q_values, 0)
        predicted_q_value = predicted_q_value.to(device)
        loss = loss_function(predicted_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''保存环节,每一千轮保存一次'''
        if episode % save_interval == 0:
            model_save_path = f'model_epoch_{episode}.pth'
            torch.save(current_net.state_dict(), model_save_path)
            print(f'Model weights saved at episode {episode} to {model_save_path}')

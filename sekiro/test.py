import find_blood_location
import cv2
import time
import directkeys
from getkeys import key_check
import torch
from restart import restart
from DQN_sekiro_pytroch import DQN
import grabscreen


# 定义模型和输入输出维度
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
        directkeys.drank()


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

def preprocess_and_normalize_image():
    gray_image = cv2.cvtColor(grabscreen.grab_screen(window_size), cv2.COLOR_BGR2GRAY)
    normalized_image = torch.from_numpy(gray_image).float() / 255.0
    return normalized_image
channal = 24
Inputsize = 2
Outputsize = 6
batch_size = 24
current_buffer_size = 0
window_size = (800, 300, 1204, 952)
self_blood_window = (120, 1025, 700, 1040)
boss_blood_window = (120, 135, 550, 155)
# 创建模型实例
model = DQN(channal,Inputsize, Outputsize)
paused = True
buffer_images = []
buffer_features = []
# 加载训练好的模型权重
model.load_state_dict(torch.load('model_epoch_2000.pth'))
model.eval()  # 将模型设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pause_game(paused)
directkeys.lock_vision()
while True:
    print("开始")
    self_blood, boss_blood = get_blood_data()
    # 示例输入数据（根据你的特征维度和预处理逻辑进行修改）
    while current_buffer_size < batch_size:
        image_data = preprocess_and_normalize_image()
        feature_vector = preprocess_blood_data(self_blood, boss_blood)
        buffer_images.append(image_data)
        buffer_features.append(feature_vector)
        current_buffer_size = current_buffer_size + 1
    batch_images = torch.stack(buffer_images)
    batch_features = torch.stack(buffer_features)
    batch_images = batch_images.to(device)
    batch_features = batch_features.to(device)
    # 使用模型进行预测
    with torch.no_grad():
        q_values = model(batch_images, batch_features)
        print("q_values",q_values,q_values.shape)
        # 获取预测的动作
        predicted_action = torch.argmax(q_values).item()

    # 执行预测的动作
    take_action(predicted_action)
    if self_blood < 3:
        restart()
        directkeys.lock_vision()
        directkeys.lock_vision()
        directkeys.lock_vision()


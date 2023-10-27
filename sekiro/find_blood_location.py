import cv2
import time
import grabscreen

def self_blood_count(self_gray):
    self_blood = 0
    for self_bd_num in self_gray[0]:
        # print(self_bd_num , ',')
        # self blood gray pixel 80~98
        # 血量灰度值70~80
        if 70 < self_bd_num < 80:
            self_blood += 1
    #print('self_blood', self_blood)
    return self_blood


def boss_blood_count(boss_gray):
    boss_blood = 0
    for boss_bd_num in boss_gray[3]:
        # print(boss_bd_num , ',')
        # boss blood gray pixel 65~75
        # 血量灰度值24~30
        # print(boss_bd_num)
        if 25 < boss_bd_num < 50:
            boss_blood += 1
    #print('boss_blood', boss_blood)
    return boss_blood



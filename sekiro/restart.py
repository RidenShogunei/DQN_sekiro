# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:31:36 2020

@author: pang
"""

import directkeys
import time

def restart():
    print("死,restart")
    time.sleep(4)
    directkeys.attack()
    time.sleep(0.2)
    directkeys.attack()
    print("开始新一轮")
  
if __name__ == "__main__":  
    restart()
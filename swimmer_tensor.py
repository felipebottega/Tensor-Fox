import numpy as np


def swimmer_initialize():
    swimmer_frame = np.ones((32,32))
    for i in range(10,22):
        swimmer_frame[i,15] = 0
    return swimmer_frame


def upper_left(swimmer_frame,n):
    if n == 0:
        for i in range(5,11):
            swimmer_frame[i,14] = 0
    if n == 1:
        for i in range(5,11):
            swimmer_frame[i,4+i] = 0
    if n == 2:
        for j in range(9,15):
            swimmer_frame[10,j] = 0
    if n == 3:
        for i in range(10,17):
            swimmer_frame[i,24-i] = 0
    return swimmer_frame


def upper_right(swimmer_frame,n):
    if n == 0:
        for i in range(5,11):
            swimmer_frame[i,16] = 0
    if n == 1:
        for i in range(5,11):
            swimmer_frame[i,26-i] = 0
    if n == 2:
        for j in range(16,22):
            swimmer_frame[10,j] = 0
    if n == 3:
        for i in range(10,16):
            swimmer_frame[i,6+i] = 0
    return swimmer_frame


def lower_left(swimmer_frame,n):
    if n == 0:
        for i in range(16,22):
            swimmer_frame[i,i-7] = 0
    if n == 1:
        for j in range(9,15):
            swimmer_frame[21,j] = 0
    if n == 2:
        for i in range(21,27):
            swimmer_frame[i,35-i] = 0
    if n == 3:
        for i in range(21,27):
            swimmer_frame[i,14] = 0
    return swimmer_frame


def lower_right(swimmer_frame,n):
    if n == 0:
        for i in range(16,22):
            swimmer_frame[i,37-i] = 0
    if n == 1:
        for j in range(16,22):
            swimmer_frame[21,j] = 0
    if n == 2:
        for i in range(21,27):
            swimmer_frame[i,i-5] = 0
    if n == 3:
        for i in range(21,27):
            swimmer_frame[i,16] = 0
    return swimmer_frame


def create():
    # Create tensor of swimmers.
    swimmer_tensor = np.ones((32,32,256))
    s = 0
    for i in range(0,4):
        for j in range(0,4):
            for k in range(0,4):
                for l in range(0,4):
                    swimmer_frame = swimmer_initialize()
                    swimmer_frame = upper_left(swimmer_frame,i)
                    swimmer_frame = upper_right(swimmer_frame,j)
                    swimmer_frame = lower_left(swimmer_frame,k)
                    swimmer_frame = lower_right(swimmer_frame,l)
                    swimmer_tensor[:,:,s] = swimmer_frame
                    s += 1
                
    return swimmer_tensor
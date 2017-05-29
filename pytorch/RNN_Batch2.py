import os
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for i in range(10):
    #os.system("python3 /home/lu/code/pytorch/EMS_Rnn2.py")
    output = os.popen("python3 /home/lu/code/pytorch/EMS_Rnn_gpu1.py")
    print(output.read())
    print(timeSince(start))
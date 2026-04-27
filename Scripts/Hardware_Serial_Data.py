import serial
import matplotlib.pyplot as plt
from collections import deque

# Serial setup
ser = serial.Serial('COM3', 115200, timeout=1)

# Data buffer
data = deque([0]*100, maxlen=100)

# Plot setup
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(data)

ax.set_ylim(0, 1023)  # adjust for your signal
ax.set_title("Live EMG Data")

while True:
    try:
        if ser.in_waiting:
            value = float(ser.readline().decode().strip())
            data.append(value)

            line.set_ydata(data)
            line.set_xdata(range(len(data)))

            print(data)

            plt.draw()
            plt.pause(0.01)

    except KeyboardInterrupt:
        break
    except:
        pass

ser.close()
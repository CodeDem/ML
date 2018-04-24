from scipy.io.wavfile import read
import numpy as np
# read audio samples
input_data = read("audio3.wav")
audio = input_data[1]

# Use numpy method for bigger audio file
# my_array = np.array(audio)
# zeroCrossingRate = ((my_array[:-1] * my_array[1:]) < 0).sum()
# print(zeroCrossingRate)

# Two line code to find ZCR
# data = sum(1 for i in range(1, len(audio)) if audio[i-1]*audio[i] < 0)
# print("data" + str(data))

# Easy to understand code to find ZCR


def change_sign(v1, v2):
    return v1 * v2 < 0


s = 0
for ind, _ in enumerate(audio):
    if ind+1 < len(audio):
        if change_sign(audio[ind], audio[ind+1]):
            s += 1
print("finalCheck " + str(s))

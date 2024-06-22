import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft

t = np.linspace(0, 3, 12 * 1024)

#Milestone 1 : Song Creation

#frequencies of piano 
c = 130.81
d = 146.83
e = 164.81
f = 174.61
g = 196
a = 220
b = 246.93
rest = 0   #indicates silence
bb = 233.082

beatsPerMin = 140
beatsInSecs = 60 / beatsPerMin

whole  = beatsInSecs
half = beatsInSecs / 2
quarter = beatsInSecs / 4

#generate sin wave of freq and freq*2
def getNote(f):
    return np.sin(2 * np.pi * f * t) + np.sin(2 * np.pi * (f*2) * t)

def makeSong(getNotes):
    song = 0
    nextNoteStart = 0
    for currNote in getNotes:
        freq = currNote[0]
        duration = currNote[1]
        pulse = (t >= nextNoteStart) & (t <= nextNoteStart + duration)
        nextNoteStart += duration + quarter / 4
        song += getNote(freq) * pulse
    return song
    
# We divide the note's freqency by 2 to get the same note in a lower octave
superMarioTheme= [
   [c, quarter], 
   [rest, quarter],
   [g/2, quarter],
   [rest, quarter],
   [e/2, quarter],

   [rest, quarter],
   [a/2, quarter],
   [b/2, quarter],
   [bb/2, quarter],
   [a/2, quarter],

   [g/2, quarter],
   [e, quarter],
   [g, quarter],
   [a, quarter],
   [f, quarter],
   [g, quarter],

   [rest, quarter],
   [e, quarter],
   [c, quarter],
   [d, quarter],
   [b/2, quarter]
]

song = makeSong(superMarioTheme)

#------------------------------------------------------------
#Milestone 2 : Noise Cancellation 

N = 3 * 1024
f = np.linspace(0, 512, int(N/2))

songF = fft(song)
songF = 2 / N * np.abs(songF[0:int(N/2)])

randNoiseFreq1, randNoiseFreq2 = np.random.randint(0, 512, 2)

noise = np.sin(2 * np.pi * randNoiseFreq1 * t) + np.sin(2 * np.pi * randNoiseFreq2 * t)

songNoiseT = song + noise

songNoiseF = fft(songNoiseT)
songNoiseF = 2 / N * np.abs(songNoiseF[0:int(N / 2)])

#max freq in original song
maxFreq = int(np.ceil(np.max(songF)))

(noiseFreqs,) = np.where(songNoiseF > maxFreq)

noiseCancellation = 0
for i in noiseFreqs:
    noiseFreq = f[i]
    noiseCancellation += np.sin(2 * np.pi * np.round(noiseFreq) * t)
    
  
songAfterNoiseRemovalT = songNoiseT - noiseCancellation

songAfterNoiseRemovalF = fft(songAfterNoiseRemovalT)
songAfterNoiseRemovalF = 2 / N * np.abs(songAfterNoiseRemovalF[0:int(N/2)])

# Signal Plotting
# First figure for time domain plots
plt.figure(1)

# Song without noise in time domain
plt.subplot(3, 1, 1)
plt.title("Time Domain")
plt.plot(t, song)
plt.ylabel("No Noise")

# Song with noise in time domain
plt.subplot(3, 1, 2)
plt.plot(t, songNoiseT)
plt.ylabel("With Noise")

# Song after removing noise in time domain
plt.subplot(3, 1, 3)
plt.plot(t, songAfterNoiseRemovalT)
plt.ylabel("After Removal")

# Second figure for freqency domain plots
plt.figure(2)

# Song without noise in freqency domain
plt.subplot(3, 1, 1)
plt.title("Frequency Domain")
plt.plot(f, songF)
plt.ylabel("No Noise")

# Song with noise in freqency domain
plt.subplot(3, 1, 2)
plt.plot(f, songNoiseF)
plt.ylabel("With Noise")

# Song after removing noise in freqency domain
plt.subplot(3, 1, 3)
plt.plot(f, songAfterNoiseRemovalF)
plt.ylabel("After Removal")


plt.show()
sd.play(songAfterNoiseRemovalT, N)

#End of Code

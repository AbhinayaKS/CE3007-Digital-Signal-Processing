import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import winsound
from scipy import signal
import time


def fnGenSampledSinusoid(A, Freq, Phi, Fs, sTime, eTime):
    n = np.arange(sTime, eTime, 1.0 / Fs)
    y = A * np.cos(2 * np.pi * Freq * n + Phi)
    return [n, y]


# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s * 32767) for s in yFloat]
    return np.array(y_16bit, dtype='int16')


# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s / 32767.0) for s in y_16bit]
    return np.array(yFloat, dtype='float')


def Lab2_1b():
    h = np.array([0.2, 0.3, -0.5])

    # Because x = cos(0.1 * pi * n), the number of samples per cycle = 2/0.1 = 20
    nCycles = 5
    noOfSamples = 20 * nCycles + 1
    n = np.arange(0, noOfSamples)
    x = 1 * np.cos(0.1 * np.pi * n)
    y = np.convolve(x, h)
    plt.figure()
    plt.title('Original Signal (G) and Convolved Signal (R)')
    plt.xlabel('n')
    plt.ylabel('Signal')
    plt.stem(y, linefmt='ro')
    plt.stem(x, linefmt='g-x')
    plt.show()


def convolve(x, h):
    result = np.zeros(len(x) + len(h) - 1)
    for i, xVal in enumerate(x):
        for j, hVal in enumerate(h):
            result[i + j] = result[i + j] + xVal * hVal
    return result


def Lab2_3():
    # Subsection (a)
    # Filter declaration
    impulseH = np.zeros(8000)
    impulseH[1] = 1
    impulseH[4000] = 0.5
    impulseH[7900] = 0.3

    plt.figure()
    plt.title('Impulse Response')
    plt.xlabel('n')
    plt.ylabel('Impulse Response')
    plt.stem(impulseH, linefmt='ro')
    plt.show()

    # Read the input file
    fileName = 'Lab 2/testIp_16bit.wav'
    winsound.PlaySound(fileName, winsound.SND_FILENAME)
    [Fs, sampleX_16bit] = wavfile.read(fileName)
    sampleX_float = fnNormalize16BitToFloat(sampleX_16bit)

    start = time.time()
    customConv = convolve(sampleX_float, impulseH)
    npConv = np.convolve(sampleX_float, impulseH)
    end = time.time()
    print(f"Runtime of the convolution is {(end - start)/2}")

    assert (customConv == npConv).all()
    print("Results are the same between the custom function and the numpy function.")

    # Subsection (b)
    y16Bit = fnNormalizeFloatTo16Bit(npConv)
    fileName = 'Lab 2/output/filteredSound.wav'
    wavfile.write(fileName, Fs, y16Bit)
    winsound.PlaySound(fileName, winsound.SND_FILENAME)

    '''
    Subsection (c)
    We can generate the convolution equation and substitute that directly
    This will save a lot of computation time as the convolution array is sparse
    '''


def Lab2_4ab(h1, h2):

    # Subsection (a)
    plt.figure(1)
    plt.xlabel('n')
    plt.ylabel('Impulse Response h1')
    plt.title('Impulse Response h1')
    plt.stem(h1, linefmt='ro')
    plt.figure(2)
    plt.xlabel('n')
    plt.ylabel('Impulse Response h2')
    plt.title('Impulse Response h2')
    plt.stem(h2, linefmt='gx')
    plt.show()

    # Subsection (b)
    # We choose 16 samples as the range is till delta of n - 15
    x = signal.unit_impulse(16) - 2 * signal.unit_impulse(16, 15)

    y1SciPy = signal.lfilter(h1, [1], x)
    y1NumPy = np.convolve(x, h1)[0: len(x)]
    y1Custom = convolve(x, h1)[0: len(x)]

    plt.figure(1)
    plt.title('Output of system with h1 - SciPy')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.stem(y1SciPy, linefmt='ro')

    plt.figure(2)
    plt.title('Output of system with h1 - NumPy')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.stem(y1NumPy, linefmt='ro')

    plt.figure(3)
    plt.title('Output of system with h1 - Custom')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.stem(y1Custom, linefmt='ro')

    plt.show()

    y2SciPy = signal.lfilter(h2, [1], x)
    y2NumPy = np.convolve(x, h2)[0: len(x)]
    y2Custom = convolve(x, h2)[0: len(x)]

    plt.figure(1)
    plt.title('Output of system with h2 - SciPy')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.stem(y2SciPy, linefmt='ro')

    plt.figure(2)
    plt.title('Output of system with h2 - NumPy')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.stem(y2NumPy, linefmt='ro')

    plt.figure(3)
    plt.title('Output of system with h2 - Custom')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.stem(y2Custom, linefmt='ro')

    plt.show()

    # The different implementations of convolutions provide the same output
    # The output is the impulse response convolved with the input


def Lab2_4c(h1, h2):
    Fs = 16000
    _, x1 = fnGenSampledSinusoid(0.1, 700, 0, Fs, 0, 1)
    _, x2 = fnGenSampledSinusoid(0.1, 3333, 0, Fs, 0, 1)
    x = x1 + x2

    # Subsection (a)
    [f, t, Sxx_clean] = signal.spectrogram(x, Fs, window='blackmanharris', nperseg=512,
                                           noverlap=int(0.9 * 512))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx_clean), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of signal')
    plt.show()

    # Subsection (b)
    y1 = np.convolve(x, h1)
    y2 = np.convolve(x, h2)

    # Subsection (c)
    # Spectrogram visualization
    plt.figure(1)
    plt.title('Output with h1')
    [f, t, Sxx_clean] = signal.spectrogram(y1, Fs, window='blackmanharris', nperseg=512,
                                           noverlap=int(0.9 * 512))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx_clean), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.figure(2)
    plt.title('Output with h2')
    [f, t, Sxx_clean] = signal.spectrogram(y2, Fs, window='blackmanharris', nperseg=512,
                                           noverlap=int(0.9 * 512))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx_clean), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()

    # Time Domain Visualization
    plt.figure(1)
    plt.title('Output y1')
    plt.stem(y1, linefmt='ro')

    plt.figure(2)
    plt.title('Output y2')
    plt.stem(y2, linefmt='gx')

    plt.show()

    # Save sound
    x_16Bit = fnNormalizeFloatTo16Bit(x)
    fileName = 'Lab 2/output/SinusoidInput.wav'
    wavfile.write(fileName, Fs, x_16Bit)
    print('Playing Input Sound..')
    winsound.PlaySound(fileName, winsound.SND_FILENAME)

    y1_16Bit = fnNormalizeFloatTo16Bit(y1)
    fileName = 'Lab 2/output/h1_Output.wav'
    wavfile.write(fileName, Fs, y1_16Bit)
    print('Playing Output Sound with H1..')
    winsound.PlaySound(fileName, winsound.SND_FILENAME)

    y2_16Bit = fnNormalizeFloatTo16Bit(y2)
    fileName = 'Lab 2/output/h2_Output.wav'
    wavfile.write(fileName, Fs, y2_16Bit)
    print('Playing Output Sound with H2..')
    winsound.PlaySound(fileName, winsound.SND_FILENAME)


def Lab2_5():
    fileName = 'Lab 2/helloworld_noisy_16bit.wav'
    winsound.PlaySound(fileName, winsound.SND_FILENAME)
    [Fs, sampleX_16bit] = wavfile.read(fileName)
    sampleX_float = fnNormalize16BitToFloat(sampleX_16bit)

    # Subsection (a)
    plt.figure(1)
    plt.title('Input Signal in time domain')
    plt.plot(sampleX_float, 'b-')

    plt.figure(2)
    plt.title('Input Signal Spectrogram')
    [f, t, Sxx_clean] = signal.spectrogram(sampleX_float, Fs, window='blackmanharris', nperseg=512, noverlap=int(0.9 * 512))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx_clean), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()

    # Subsection (b)
    # Draw the block diagram

    # Subsection (d)
    y = [0] * (len(sampleX_float) + 3)
    x = [0, 0, 0]
    x.extend(sampleX_float)
    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]
    for i in range(len(sampleX_float)):
        y[2 + i] = B[0] * x[2 + i] + B[1] * x[1 + i] + B[2] * x[i] - A[0] * y[1 + i] - A[1] * y[1 + i] - A[2] * y[i]

    plt.figure(1)
    plt.title('Output')
    plt.plot(y[3:])
    plt.show()


if __name__ == '__main__':
    H1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    H2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')
    # Lab2_1b()
    # Lab2_3()
    # Lab2_4ab(H1, H2)
    Lab2_4c(H1, H2)
    # Lab2_5()

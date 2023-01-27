import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import winsound


def fnGenSampledSinusoid(A, Freq, Phi, Fs, sTime, eTime):
    n = np.arange(sTime, eTime, 1.0 / Fs)
    y = A * np.cos(2 * np.pi * Freq * n + Phi)
    return [n, y]


def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s * 32767) for s in yFloat]
    return np.array(y_16bit, dtype='int16')


def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s / 32767.0) for s in y_16bit]
    return np.array(yFloat, dtype='float')


def continuousSinusoid(A, Freq, Phi, nCycles):
    t = np.arange(-1 * nCycles / (Freq * 2), nCycles / (Freq * 2), 0.1 / Freq)
    y = A * np.cos(2 * np.pi * Freq * t + Phi)
    return [t, y]


def Lab3_1(A, Freq, Phi, Fs, sTime, eTime, start, end, step, nCycles):
    # Subsection a)
    # Generate the different sounds sampled from signals of frequencies over a range
    for F in range(start, end, step):
        n, yFloat = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
        y16Bit = fnNormalizeFloatTo16Bit(yFloat)
        fileName = 'output/' + str(F // 1000) + 'K_16bit.wav'
        wavfile.write(fileName, Fs, y16Bit)

        # Play the sound
        print('Playing ' + str(F) + 'Hz...')
        winsound.PlaySound(fileName, winsound.SND_FILENAME)

    '''
    Based on the outputs, we can observe that we are unable to hear any sound at F = 16KHz and 32KHz. 
    We also observe that the output for F in the range of 2KHz to 16KHz is the same as that of 16KHz to 32KHz.
    This clearly indicates Aliasing as the sampling frequency is 16KHz. Therefore, at 16KHz and 32KHz we sample 0.
    '''

    # Subsection b)
    # Visualize the continuous time signal
    [t, yContinuous] = continuousSinusoid(A, Freq, Phi, nCycles)
    plt.figure(1)
    plt.stem(t, yContinuous, 'g-')
    plt.xlabel('time in seconds')
    plt.ylabel('y(t)')
    plt.grid()

    # Visualize the signal y[nT]
    n, yFloat = fnGenSampledSinusoid(A, Freq, Phi, Fs, - 1 * nCycles / (Freq * 2), nCycles / (Freq * 2))
    plt.figure(2)
    numSamples = Fs * nCycles // Freq
    plt.plot(n[0:numSamples], yFloat[0:numSamples], 'r--o')
    plt.xlabel('time in sec')
    plt.ylabel('y[nT]')
    plt.title('Sampled Signal y[nT]')
    plt.grid()

    # Visualize the signal y[n]
    plt.figure(3)
    nIdx = np.arange(0, numSamples)
    plt.plot(nIdx, yFloat[0:numSamples], 'r--o')
    plt.xlabel('sample index n')
    plt.ylabel('y[n]')
    plt.title('Sampled Signal y[n]')
    plt.grid()
    plt.show()

    # Plotting the same figures with F = 17000Hz
    Freq = 17000
    # Visualize the continuous time signal
    [t, yContinuous] = continuousSinusoid(A, Freq, Phi, nCycles)

    plt.figure(1)
    plt.stem(t, yContinuous, 'g-')
    plt.xlabel('time in seconds')
    plt.ylabel('y(t)')
    plt.grid()

    # Visualize the signal y[nT]
    n, yFloat = fnGenSampledSinusoid(A, Freq, Phi, Fs, -17 * nCycles / (Freq * 2), 17 * nCycles / (Freq * 2))

    plt.figure(2)
    numSamples = 17 * Fs * nCycles // Freq
    print(-17 * nCycles / (Freq * 2))
    plt.plot(n[0:numSamples], yFloat[0:numSamples], 'r--o')
    plt.xlabel('time in sec')
    plt.ylabel('y[nT]')
    plt.title('Sampled Signal y[nT]')
    plt.grid()

    # Visualize the signal y[n]
    plt.figure(3)
    nIdx = np.arange(0, numSamples)
    plt.plot(nIdx, yFloat[0:numSamples], 'r--o')
    plt.xlabel('sample index n')
    plt.ylabel('y[n]')
    plt.title('Sampled Signal y[n]')
    plt.grid()
    plt.show()

    '''
    i. We note that y(t) and y(nT) provide the same plots for the same start and end time where t = nT and T = 1 / Fs
    ii. Although the y values are the same, n is related to actual time with the relation t = nT and T = 1 / Fs
    iii. Yes, the signal is periodic. Number of samples generated = 16000 * 6 / 1000 = 96
    iv. Aliasing has occurred. Due to this we obtain the same sampled signal for a different signal frequency.
    '''

    # Subsection c)
    Freq = 1
    Fs = 16
    # Visualize the signal y[n]
    n, yFloat = fnGenSampledSinusoid(A, Freq, Phi, Fs, -17 * nCycles / (Freq * 2), 17 * nCycles / (Freq * 2))

    plt.figure(1)
    nIdx = np.arange(0, numSamples)
    plt.plot(nIdx, yFloat[0:numSamples], 'r--o')
    plt.xlabel('sample index n')
    plt.ylabel('y[n]')
    plt.title('Sampled Signal y[n]')
    plt.grid()
    plt.show()

    '''
    Yes, the values are the same as we have maintained the F / Fs ratio.
    '''


def Lab3_2(inputSeq, Fs, durTone):
    pass


def Lab3_4(A, w, Phi, N):
    n = np.arange(0, N, 1)
    y = np.multiply(np.power(A, n), np.exp(1j * (w * n + Phi)))

    # Subsection (1)
    # 2-D plot of real and imaginary values in the same figure
    plt.figure(1)
    plt.plot(n, y[0:N].real, 'r--o')
    plt.plot(n, y[0:N].imag, 'g--o')
    plt.xlabel('sample index n')
    plt.ylabel('y[n]')
    plt.title('Complex exponential (red = real) (green = imaginary)')
    plt.grid()

    # Subsection (2)
    # Polar plot of the sequence
    plt.figure(2)
    for x in y:
        plt.polar([0, np.angle(x)], [0, np.abs(x)], marker='o')
    plt.title('Polar plot showing phasors at n=0..N')

    # Subsection (3)
    # 3-D Plot showing the trajectory
    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    reVal = y[0:N].real
    imgVal = y[0:N].imag
    ax.plot(n, reVal, imgVal, label='complex exponential phasor')
    ax.scatter(n, reVal, imgVal, c='r', marker='o')
    ax.set_xlabel('sample n')
    ax.set_ylabel('real')
    ax.set_zlabel('imaginary')
    ax.legend()
    plt.show()

    # Subsection (4)
    # Redoing with w = 2 * pi / 18
    w = 2 * np.pi / 18
    n = np.arange(0, N, 1)
    y = np.multiply(np.power(A, n), np.exp(1j * (w * n + Phi)))

    # 2-D plot of real and imaginary values in the same figure
    plt.figure(1)
    plt.plot(n, y[0:N].real, 'r--o')
    plt.plot(n, y[0:N].imag, 'g--o')
    plt.xlabel('sample index n')
    plt.ylabel('y[n]')
    plt.title('Complex exponential (red = real) (green = imaginary)')
    plt.grid()

    # Polar plot of the sequence
    plt.figure(2)
    for x in y:
        plt.polar([0, np.angle(x)], [0, np.abs(x)], marker='o')
    plt.title('Polar plot showing phasors at n=0..N')

    # 3-D Plot showing the trajectory
    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    reVal = y[0:N].real
    imgVal = y[0:N].imag
    ax.plot(n, reVal, imgVal, label='complex exponential phasor')
    ax.scatter(n, reVal, imgVal, c='r', marker='o')
    ax.set_xlabel('sample n')
    ax.set_ylabel('real')
    ax.set_zlabel('imaginary')
    ax.legend()
    plt.show()

    '''
    Due to the increase in angular frequency, the magnitude of reduction in the amplitude is slower in the second case.
    Therefore, we observe a wider second cycle in the second case caused due to the higher angular frequency.
    '''


def Lab3_5(K, N):

    # Subsection (a)
    for k in range(K):
        n = np.arange(0, N, 1)
        Wk = 1 * np.exp(1j * 2 * np.pi * k * n / N)

        # 2-D plot of real and imaginary values in the same figure
        plt.figure(1)
        plt.plot(n, Wk[0:N].real, 'r--o')
        plt.plot(n, Wk[0:N].imag, 'g--o')
        plt.xlabel('sample index n')
        plt.ylabel('y[n]')
        plt.title('Complex exponential (red = real) (green = imaginary)')
        plt.grid()

        # 3-D Plot showing the trajectory
        plt.rcParams['legend.fontsize'] = 10
        fig = plt.figure(2)
        ax = fig.gca(projection='3d')
        reVal = Wk[0:N].real
        imgVal = Wk[0:N].imag
        ax.plot(n, reVal, imgVal, label='complex exponential phasor')
        ax.scatter(n, reVal, imgVal, c='r', marker='o')
        ax.set_xlabel('sample n')
        ax.set_ylabel('real')
        ax.set_zlabel('imaginary')
        ax.legend()
        plt.show()

    '''
    w = 2 * pi * k / N
    Therefore, the frequency increases as the value of k increases.
    When k = 0, we note that W0 = 1 (only has a constant real part)
    '''


if __name__ == '__main__':
    # Lab3_1(A=0.1, Freq=1000, Phi=0, Fs=16000, sTime=0, eTime=1, start=2000, end=32000, step=2000, nCycles=6)
    # Lab3_4(A=0.95, w=2*np.pi/36, Phi=0, N=200)
    Lab3_5(4, 16)

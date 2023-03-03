import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def convolveLab2(x, h):
    result = np.zeros(len(x) + len(h) - 1)
    for i, xVal in enumerate(x):
        for j, hVal in enumerate(h):
            result[i + j] = result[i + j] + xVal * hVal
    return result

def myDTFS(ipX, N):
    X = np.zeros(N, dtype=complex)
    for k in np.arange(0,N):
        tmpVal = 0.0
        omega = (2*np.pi/N)*k
        for n in range(N):
            tmpVal = tmpVal + ipX[n]*np.exp(-1j*omega*n)
        X[k] = tmpVal/N
    return X

def myIDTFS(X):
    x = np.zeros(len(X), dtype=float)
    N = len(x)
    for n in np.arange(0,len(x)):
        tmpVal = 0.0
        for k in np.arange(0,len(X)):
            tmpVal = tmpVal + X[k]*np.exp(+1j*(2*np.pi/N)*k*n)
        x[n] = np.absolute(tmpVal)
    return (x)

def myDFT(ipX, N):
    X = np.zeros(N, dtype=complex)
    for k in np.arange(0,N):
        tmpVal = 0.0
        omega = (2*np.pi/N)*k
        for n in range(N):
            tmpVal = tmpVal + ipX[n]*np.exp(-1j*omega*n)
        X[k] = tmpVal
    return X

def myIDFT(X):
    x = np.zeros(len(X), dtype=float)
    N = len(x)
    for n in np.arange(0,len(x)):
        tmpVal = 0.0
        for k in np.arange(0,len(X)):
            tmpVal = tmpVal + X[k]*np.exp(+1j*(2*np.pi/N)*k*n)
        x[n] = np.absolute(tmpVal)/N
    return (x)

def myDFTConvolve(ipX, impulseH):
    originalIpXLen = len(ipX)
    originalImpulseLen = len(impulseH)
    if (len(ipX) < len(impulseH)):
        ipX = np.pad(ipX, (0, len(impulseH) - len(ipX)), 'constant')
    elif (len(ipX) > len(impulseH)):
        impulseH = np.pad(impulseH, (0, len(ipX) - len(impulseH)), 'constant')
    X = myDFT(ipX, len(ipX))
    H = myDFT(impulseH, len(impulseH))
    Y = np.multiply(X,H)
    y = myIDFT(Y)
    y = np.pad(y, (0, originalIpXLen + originalImpulseLen - 1 - len(y)), 'constant')
    return y

def Q2B(ipX, N):
    X1 = myDTFS(ipX, N)
    X2 = myDFT(ipX, N)
    (X) = np.fft.fft(x)

    titleStr = 'x[n]'

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].stem(np.absolute(X1))
    axarr[0].set_title('DTFS '+titleStr)
    axarr[0].set_ylabel('mag value')


    axarr[1].stem(np.angle(X1))
    axarr[1].set_xlabel('k')
    axarr[1].set_ylabel('Phase (rad)')
    plt.show()

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].stem(np.absolute(X2))
    axarr[0].set_title('DFT '+titleStr)
    axarr[0].set_ylabel('mag value')


    axarr[1].stem(np.angle(X2))
    axarr[1].set_xlabel('k')
    axarr[1].set_ylabel('Phase (rad)')
    plt.show()

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].stem(np.absolute(X))
    axarr[0].set_title('FFT Package '+titleStr)
    axarr[0].set_ylabel('mag value')


    axarr[1].stem(np.angle(X))
    axarr[1].set_xlabel('k')
    axarr[1].set_ylabel('FFT Package Phase (rad)')
    plt.show()

    return (X, X1, X2)

def Q2C(valDTFS, valDFT, valFFT, ipX):
    iDTFSResult = myIDTFS(valDTFS)
    iDFTResult = myIDFT(valDFT)

    print("Verify inverse DTFS Result: " + str(np.allclose(iDTFSResult, ipX)))
    print("Verify inverse DFT Result: " + str(np.allclose(iDFTResult, ipX)))
    print("Verify FFT and DFT Result: " + str(np.allclose(valFFT, valDFT)))

def Q2D(x, N):
    newX = [x[len(x) - 1]] + x[:len(x) - 1]
    X1 = myDTFS(newX, N)

    titleStr = 'newX[n]'

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].stem(np.absolute(X1))
    axarr[0].set_title('DTFS Phase Shift'+titleStr)
    axarr[0].set_ylabel('mag value')


    axarr[1].stem(np.angle(X1))
    axarr[1].set_xlabel('k')
    axarr[1].set_ylabel('Phase (rad)')
    plt.show()

    newX = np.array(x) * 10
    X1 = myDTFS(newX, N)

    titleStr = 'newX[n]'

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].stem(np.absolute(X1))
    axarr[0].set_title('DTFS Amplitude Change'+titleStr)
    axarr[0].set_ylabel('mag value')


    axarr[1].stem(np.angle(X1))
    axarr[1].set_xlabel('k')
    axarr[1].set_ylabel('Phase (rad)')
    plt.show()

def Q3(N):
    W = np.zeros(shape=(N, N),dtype=complex)
    for n in np.arange(0,N):
        for k in np.arange(0, N):
            W[n, k] = np.exp(-1j*(2*np.pi/N)*k*n)
    W_angle = np.angle(W)

    plt.figure()
    plt.title('Each row shows the k-th harmonic, from n=0..N-1 index')
    Q = plt.quiver( np.cos(W_angle),np.sin(W_angle),  units='width')
    titleStr = 'Fourier complex vectors N='+str(N)
    plt.title(titleStr)
    plt.ylabel('k-values')
    plt.xlabel('n-values')
    plt.grid()
    plt.show()

def Q4B(ipX, truncateLengths):
    for N in truncateLengths:
        ipx = np.pad(ipX, (0, N - len(ipX)), 'constant')
        valDTFS = myDTFS(ipx, N)
        magnitudeDTFS = [abs(result) for result in valDTFS]
        phaseDTFS = [np.angle(result) for result in valDTFS]
        _, ax = plt.subplots(2, 1)
        plt.title('With N = ' + str(N))
        ax[0].set_title('DTFS Magnitude')
        ax[0].stem(magnitudeDTFS)
        ax[0].grid()
        ax[1].set_title('DTFS Phase')
        ax[1].stem(phaseDTFS)
        ax[1].grid()
        plt.show()

def Q5(x, h):
    y = myDFTConvolve(x, h)
    expectedResult = signal.fftconvolve(x, h)
    resultFromLab2 = convolveLab2(x, h)
    print("Result Comparisons: ", end = "")
    print(np.allclose(list(y), list(expectedResult)), end = ' ')   
    print(np.allclose(list(resultFromLab2), list(expectedResult)))  

if __name__ == '__main__':
    x = [1,1,0,0,0,0,0,0,0,0,0,0]
    N = 12
    truncateLengths = [12, 24,48,96]
    ipX = [1, 1, 1, 1, 1, 1, 1]
    convExampleX = [1, 2, 0, 0, 0, 0, 0, 0]
    convExampleH = [1, 2, 3, 0, 0, 0, 0, 0]
    # valFFT, valDTFS, valDFT = Q2B(x, N)
    # Q2C(valDTFS=valDTFS, valDFT=valDFT, valFFT=valFFT, ipX=x)
    # Q2D(x, N)
    # Q3(N=16)
    # Q4B(ipX, truncateLengths)
    Q5(convExampleX, convExampleH)
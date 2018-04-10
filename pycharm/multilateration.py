import numpy as np
import scipy.signal as sig


def highpass_filter(rate, data, cutoff, order=3, verbose=True):
    data = data.ravel()
    nyq = 0.5*rate
    high = cutoff/nyq
    b, a = sig.butter( 4, high, btype='highpass' )
    for o in xrange(order):
        fdata = sig.lfilter( b, a, data )
    return fdata


def lowpass_filter(rate, data, cutoff, order=3, verbose=True) :
    data = data.ravel()	
    nyq = 0.5*rate
    low = cutoff/nyq
    b, a = sig.butter( 4, low, btype='lowpass' )
    for o in xrange(order):
        fdata = sig.filtfilt(b, a, data)
    return fdata


def envelope(rate, data, window_size=0.0005, gauss=False):
    from scipy.signal import gaussian

    rstd_window_size = int(window_size * rate)
    if gauss:
        w = 1.0 * gaussian(rstd_window_size, std=rstd_window_size/7)
    else:
        w = 1.0 * np.ones(rstd_window_size)
    w /= np.sum(w)
    rstd = (np.sqrt((np.correlate(data ** 2, w, mode='same') -
                     np.correlate(data, w, mode='same') ** 2)).ravel())* np.sqrt(2.)
    return rstd


def filterMatrix(rate, data, cutoff,envelopeBool = False):
    containerMatrix = np.zeros(data.shape)
    for num,i in enumerate(data[:]):
        containerMatrix[num] = highpass_filter(rate, i, cutoff)
        containerMatrix[num] = lowpass_filter(rate, containerMatrix[num],10000)
        if(envelopeBool):
            containerMatrix[num] = envelope(rate,containerMatrix[num])
    return containerMatrix


def getFirstElements(inputMatrix):
    returnList = list()
    for i in inputMatrix:
        returnList.append(i[0])
    return returnList


def getOverlappingTransients(transientList, distance = 20):
    """
    @arg transientList: list of lists of transients
    @arg distance: int of sample distance considered to be an overlap
    iterates over one list of transients and checks the other lists for corresponding (determined by distance) events.
    @return: tuples of transients which start within a distance-amount of samples within each other
    """
    matchingTransients = list()
    for transient in transientList[0]:  # pick first list, iterate over
        matcher = [transient]
        hasMatch = False  # boolean to store whether we have matches so far
        rangeDown = transient.startTime-distance
        rangeUp = transient.startTime+distance
        # now: check all other transient lists for occurence within distance... get iterator for lists
        for otherLists in transientList[1:]:
            for entry in otherLists:  # entries of subsequent list
                if (rangeDown < entry.startTime < rangeUp):  # found occurence within range, continue with next list
                    matcher.append(entry)  # keep transients in temp list
                    hasMatch = True
                    '''
                    Problem: can shorten list of transients upon finding something because it is ordered 
                    BUT: iterator is already created, so does it help?
                    Is this even a bottleneck worth the trouble in the first place?'''
                    continue
                hasMatch = False  # above clause is false for the current list = no match found
        if(hasMatch):
            matchingTransients.append(matcher)
    return(matchingTransients)
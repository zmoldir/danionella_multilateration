import sys
sys.path.insert(0, '/home/maximilian/pycharm/')
sys.path.insert(0, '/home/maximilian/unistuff/paris_ens/cal_neuroim/pystuff/')
import multilateration as mult
import os
import soundfile as sf
import numpy as np
import itertools
from cal_neuroIm import thresholdEventDetect

wd = '/home/maximilian/jlab_fishies/jupyter_notebooks/megaSamples/'
fileList = os.listdir(wd)
print fileList
sampRate = 65000
rmse = False
highPassCutoff = 400
cutLength = 6500

testPulse = sf.read(wd+'/testPulse.wav')[0][cutLength:]

channel1 = mult.filterMatrix(sampRate,sf.read(wd+'channel1.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel1_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel1_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel1_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel1_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel1_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel1_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]*1.7

channel1mean = np.mean(channel1,axis=0)
channel1_5cmmean = np.mean(channel1_5cm, axis=0)
channel1_10cmmean = np.mean(channel1_10cm, axis=0)
channel1_15cmmean = np.mean(channel1_15cm, axis=0)

channel1_deviance = np.std(channel1,axis=0)
channel1_5cmdeviance = np.std(channel1_5cm, axis=0)
channel1_10cmdeviance = np.std(channel1_10cm, axis=0)
channel1_15cmdeviance = np.std(channel1_15cm, axis=0)

channel2 = mult.filterMatrix(sampRate,sf.read(wd+'channel2.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel2_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel2_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel2_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel2_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel2_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel2_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]

channel2mean = np.mean(channel2,axis=0)
channel2_5cmmean = np.mean(channel2_5cm, axis=0)
channel2_10cmmean = np.mean(channel2_10cm, axis=0)
channel2_15cmmean = np.mean(channel2_15cm, axis=0)

channel2_deviance = np.std(channel2,axis=0)
channel2_5cmdeviance = np.std(channel2_5cm, axis=0)
channel2_10cmdeviance = np.std(channel2_10cm, axis=0)
channel2_15cmdeviance = np.std(channel2_15cm, axis=0)

channel3 = mult.filterMatrix(sampRate,sf.read(wd+'channel3.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel3_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel3_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel3_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel3_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel3_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel3_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]

channel3mean = np.mean(channel3,axis=0)
channel3_5cmmean = np.mean(channel3_5cm, axis=0)
channel3_10cmmean = np.mean(channel3_10cm, axis=0)
channel3_15cmmean = np.mean(channel3_15cm, axis=0)

channel3_deviance = np.std(channel3,axis=0)
channel3_5cmdeviance = np.std(channel3_5cm, axis=0)
channel3_10cmdeviance = np.std(channel3_10cm, axis=0)
channel3_15cmdeviance = np.std(channel3_15cm, axis=0)

channel4 = mult.filterMatrix(sampRate,sf.read(wd+'channel4.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel4_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel4_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel4_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel4_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
channel4_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel4_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]

channel4mean = np.mean(channel4,axis=0)
channel4_5cmmean = np.mean(channel4_5cm, axis=0)
channel4_10cmmean = np.mean(channel4_10cm, axis=0)
channel4_15cmmean = np.mean(channel4_15cm, axis=0)

channel4_deviance = np.std(channel4,axis=0)
channel4_5cmdeviance = np.std(channel4_5cm, axis=0)
channel4_10cmdeviance = np.std(channel4_10cm, axis=0)
channel4_15cmdeviance = np.std(channel4_15cm, axis=0)


rmse= True

rmschannel1 = mult.filterMatrix(sampRate,sf.read(wd+'channel1.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel1_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel1_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel1_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel1_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel1_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel1_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]*1.7

rmschannel1mean = np.mean(rmschannel1,axis=0)
rmschannel1_5cmmean = np.mean(rmschannel1_5cm, axis=0)
rmschannel1_10cmmean = np.mean(rmschannel1_10cm, axis=0)
rmschannel1_15cmmean = np.mean(rmschannel1_15cm, axis=0)

rmschannel1_deviance = np.std(rmschannel1,axis=0)
rmschannel1_5cmdeviance = np.std(rmschannel1_5cm, axis=0)
rmschannel1_10cmdeviance = np.std(rmschannel1_10cm, axis=0)
rmschannel1_15cmdeviance = np.std(rmschannel1_15cm, axis=0)

rmschannel2 = mult.filterMatrix(sampRate,sf.read(wd+'channel2.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel2_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel2_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel2_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel2_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel2_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel2_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]

rmschannel2mean = np.mean(rmschannel2,axis=0)
rmschannel2_5cmmean = np.mean(rmschannel2_5cm, axis=0)
rmschannel2_10cmmean = np.mean(rmschannel2_10cm, axis=0)
rmschannel2_15cmmean = np.mean(rmschannel2_15cm, axis=0)

rmschannel2_deviance = np.std(rmschannel2,axis=0)
rmschannel2_5cmdeviance = np.std(rmschannel2_5cm, axis=0)
rmschannel2_10cmdeviance = np.std(rmschannel2_10cm, axis=0)
rmschannel2_15cmdeviance = np.std(rmschannel2_15cm, axis=0)

rmschannel3 = mult.filterMatrix(sampRate,sf.read(wd+'channel3.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel3_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel3_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel3_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel3_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel3_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel3_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]

rmschannel3mean = np.mean(rmschannel3,axis=0)
rmschannel3_5cmmean = np.mean(rmschannel3_5cm, axis=0)
rmschannel3_10cmmean = np.mean(rmschannel3_10cm, axis=0)
rmschannel3_15cmmean = np.mean(rmschannel3_15cm, axis=0)

rmschannel3_deviance = np.std(rmschannel3,axis=0)
rmschannel3_5cmdeviance = np.std(rmschannel3_5cm, axis=0)
rmschannel3_10cmdeviance = np.std(rmschannel3_10cm, axis=0)
rmschannel3_15cmdeviance = np.std(rmschannel3_15cm, axis=0)

rmschannel4 = mult.filterMatrix(sampRate,sf.read(wd+'channel4.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel4_5cm = mult.filterMatrix(sampRate,sf.read(wd+'channel4_5cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel4_10cm = mult.filterMatrix(sampRate,sf.read(wd+'channel4_10cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]
rmschannel4_15cm = mult.filterMatrix(sampRate,sf.read(wd+'channel4_15cm.wav')[0].T, highPassCutoff, envelopeBool=rmse)[:,cutLength:]

rmschannel4mean = np.mean(rmschannel4,axis=0)
rmschannel4_5cmmean = np.mean(rmschannel4_5cm, axis=0)
rmschannel4_10cmmean = np.mean(rmschannel4_10cm, axis=0)
rmschannel4_15cmmean = np.mean(rmschannel4_15cm, axis=0)

rmschannel4_deviance = np.std(rmschannel4,axis=0)
rmschannel4_5cmdeviance = np.std(rmschannel4_5cm, axis=0)
rmschannel4_10cmdeviance = np.std(rmschannel4_10cm, axis=0)
rmschannel4_15cmdeviance = np.std(rmschannel4_15cm, axis=0)

transLength =300
treshDev = 4
rmschan1Transients = mult.getFirstElements(thresholdEventDetect(rmschannel1.T, thresholdDeviance=treshDev, minEventLength= transLength))
rmschan2Transients = mult.getFirstElements(thresholdEventDetect(rmschannel2.T, thresholdDeviance=treshDev, minEventLength= transLength))
rmschan3Transients = mult.getFirstElements(thresholdEventDetect(rmschannel3.T, thresholdDeviance=treshDev, minEventLength= transLength))
rmschan4Transients = mult.getFirstElements(thresholdEventDetect(rmschannel4.T, thresholdDeviance=treshDev, minEventLength= transLength))

rmschan1_5cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel1_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan2_5cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel2_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan3_5cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel3_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan4_5cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel4_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))

rmschan1_10cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel1_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan2_10cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel2_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan3_10cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel3_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan4_10cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel4_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))

rmschan1_15cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel1_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan2_15cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel2_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan3_15cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel3_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
rmschan4_15cmTransients = mult.getFirstElements(thresholdEventDetect(rmschannel4_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))

transLength = 300
treshDev = 4
chan1Transients = mult.getFirstElements(thresholdEventDetect(channel1.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan2Transients = mult.getFirstElements(thresholdEventDetect(channel2.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan3Transients = mult.getFirstElements(thresholdEventDetect(channel3.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan4Transients = mult.getFirstElements(thresholdEventDetect(channel4.T, thresholdDeviance = treshDev, minEventLength= transLength))

chan1_5cmTransients = mult.getFirstElements(thresholdEventDetect(channel1_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan2_5cmTransients = mult.getFirstElements(thresholdEventDetect(channel2_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan3_5cmTransients = mult.getFirstElements(thresholdEventDetect(channel3_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan4_5cmTransients = mult.getFirstElements(thresholdEventDetect(channel4_5cm.T, thresholdDeviance = treshDev, minEventLength= transLength))

chan1_10cmTransients = mult.getFirstElements(thresholdEventDetect(channel1_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan2_10cmTransients = mult.getFirstElements(thresholdEventDetect(channel2_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan3_10cmTransients = mult.getFirstElements(thresholdEventDetect(channel3_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan4_10cmTransients = mult.getFirstElements(thresholdEventDetect(channel4_10cm.T, thresholdDeviance = treshDev, minEventLength= transLength))

chan1_15cmTransients = mult.getFirstElements(thresholdEventDetect(channel1_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan2_15cmTransients = mult.getFirstElements(thresholdEventDetect(channel2_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan3_15cmTransients = mult.getFirstElements(thresholdEventDetect(channel3_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))
chan4_15cmTransients = mult.getFirstElements(thresholdEventDetect(channel4_15cm.T, thresholdDeviance = treshDev, minEventLength= transLength))

rmscm15amp50 = list()
rmscm15amp50.append([i.amp50+i.startTime for i in rmschan1_15cmTransients])
rmscm15amp50.append([i.amp50+i.startTime for i in rmschan2_15cmTransients])
rmscm15amp50.append([i.amp50+i.startTime for i in rmschan3_15cmTransients])
rmscm15amp50.append([i.amp50+i.startTime for i in rmschan4_15cmTransients])
rmscm15amp50 = list(itertools.chain.from_iterable(rmscm15amp50))
np.savetxt(wd+"rmscm15amp50",rmscm15amp50)

rmscm10amp50 = list()
rmscm10amp50.append([i.amp50+i.startTime for i in rmschan1_10cmTransients])
rmscm10amp50.append([i.amp50+i.startTime for i in rmschan2_10cmTransients])
rmscm10amp50.append([i.amp50+i.startTime for i in rmschan3_10cmTransients])
rmscm10amp50.append([i.amp50+i.startTime for i in rmschan4_10cmTransients])
rmscm10amp50 = list(itertools.chain.from_iterable(rmscm10amp50))
np.savetxt(wd+"rmscm10amp50",rmscm10amp50)

rmscm5amp50 = list()
rmscm5amp50.append([i.amp50+i.startTime for i in rmschan1_5cmTransients])
rmscm5amp50.append([i.amp50+i.startTime for i in rmschan2_5cmTransients])
rmscm5amp50.append([i.amp50+i.startTime for i in rmschan3_5cmTransients])
rmscm5amp50.append([i.amp50+i.startTime for i in rmschan4_5cmTransients])
rmscm5amp50 = list(itertools.chain.from_iterable(rmscm5amp50))
np.savetxt(wd+"rmscm5amp50",rmscm5amp50)

rmsamp50 = list()
rmsamp50.append([i.amp50+i.startTime for i in rmschan1_5cmTransients])
rmsamp50.append([i.amp50+i.startTime for i in rmschan2_5cmTransients])
rmsamp50.append([i.amp50+i.startTime for i in rmschan3_5cmTransients])
rmsamp50.append([i.amp50+i.startTime for i in rmschan4_5cmTransients])
rmsamp50 = list(itertools.chain.from_iterable(rmsamp50))
np.savetxt(wd+"rmsamp50",rmsamp50)


cm15amp50 = list()
cm15amp50.append([i.amp50+i.startTime for i in chan1_15cmTransients])
cm15amp50.append([i.amp50+i.startTime for i in chan2_15cmTransients])
cm15amp50.append([i.amp50+i.startTime for i in chan3_15cmTransients])
cm15amp50.append([i.amp50+i.startTime for i in chan4_15cmTransients])
cm15amp50 = list(itertools.chain.from_iterable(cm15amp50))
np.savetxt(wd+"cm15amp50",cm15amp50)

cm10amp50 = list()
cm10amp50.append([i.amp50+i.startTime for i in chan1_10cmTransients])
cm10amp50.append([i.amp50+i.startTime for i in chan2_10cmTransients])
cm10amp50.append([i.amp50+i.startTime for i in chan3_10cmTransients])
cm10amp50.append([i.amp50+i.startTime for i in chan4_10cmTransients])
cm10amp50 = list(itertools.chain.from_iterable(cm10amp50))
np.savetxt(wd+"cm10amp50",cm10amp50)

cm5amp50 = list()
cm5amp50.append([i.amp50+i.startTime for i in chan1_5cmTransients])
cm5amp50.append([i.amp50+i.startTime for i in chan2_5cmTransients])
cm5amp50.append([i.amp50+i.startTime for i in chan3_5cmTransients])
cm5amp50.append([i.amp50+i.startTime for i in chan4_5cmTransients])
cm5amp50 = list(itertools.chain.from_iterable(cm5amp50))
np.savetxt(wd+"cm5amp50",cm5amp50)

amp50 = list()
amp50.append([i.amp50+i.startTime for i in chan1_5cmTransients])
amp50.append([i.amp50+i.startTime for i in chan2_5cmTransients])
amp50.append([i.amp50+i.startTime for i in chan3_5cmTransients])
amp50.append([i.amp50+i.startTime for i in chan4_5cmTransients])
amp50 = list(itertools.chain.from_iterable(amp50))
np.savetxt(wd+"amp50", amp50)

import itertools
pos1 = np.repeat(sampRate*2-cutLength,120)
starttime0cm = list()
starttime0cm.append([i.startTime for i in chan1Transients])
starttime0cm.append([i.startTime for i in chan2Transients])
starttime0cm.append([i.startTime for i in chan3Transients])
starttime0cm.append([i.startTime for i in chan4Transients])
starttime0cm = list(itertools.chain.from_iterable(starttime0cm))
np.savetxt(wd+'start0cm', starttime0cm)

starttime5cm = list()
starttime5cm.append([i.startTime for i in chan1_5cmTransients])
starttime5cm.append([i.startTime for i in chan2_5cmTransients])
starttime5cm.append([i.startTime for i in chan3_5cmTransients])
starttime5cm.append([i.startTime for i in chan4_5cmTransients])
starttime5cm = list(itertools.chain.from_iterable(starttime5cm))
np.savetxt(wd+'start5cm', starttime5cm)


starttime10cm = list()
starttime10cm.append([i.startTime for i in chan1_10cmTransients])
starttime10cm.append([i.startTime for i in chan2_10cmTransients])
starttime10cm.append([i.startTime for i in chan3_10cmTransients])
starttime10cm.append([i.startTime for i in chan4_10cmTransients])
starttime10cm = list(itertools.chain.from_iterable(starttime10cm))
np.savetxt(wd+'start10cm', starttime10cm)

starttime15cm = list()
starttime15cm.append([i.startTime for i in chan1_15cmTransients])
starttime15cm.append([i.startTime for i in chan2_15cmTransients])
starttime15cm.append([i.startTime for i in chan3_15cmTransients])
starttime15cm.append([i.startTime for i in chan4_15cmTransients])
starttime15cm = list(itertools.chain.from_iterable(starttime15cm))
np.savetxt(wd+'start15cm', starttime15cm)

rmsstarttime0cm = list()
rmsstarttime0cm.append([i.startTime for i in rmschan1Transients])
rmsstarttime0cm.append([i.startTime for i in rmschan2Transients])
rmsstarttime0cm.append([i.startTime for i in rmschan3Transients])
rmsstarttime0cm.append([i.startTime for i in rmschan4Transients])
rmsstarttime0cm = list(itertools.chain.from_iterable(rmsstarttime0cm))
np.savetxt(wd+'rmsstart0cm', rmsstarttime0cm)

rmsstarttime5cm = list()
rmsstarttime5cm.append([i.startTime for i in rmschan1_5cmTransients])
rmsstarttime5cm.append([i.startTime for i in rmschan2_5cmTransients])
rmsstarttime5cm.append([i.startTime for i in rmschan3_5cmTransients])
rmsstarttime5cm.append([i.startTime for i in rmschan4_5cmTransients])
rmsstarttime5cm = list(itertools.chain.from_iterable(rmsstarttime5cm))
np.savetxt(wd+'rmsstart5cm', rmsstarttime5cm)


rmsstarttime10cm = list()
rmsstarttime10cm.append([i.startTime for i in rmschan1_10cmTransients])
rmsstarttime10cm.append([i.startTime for i in rmschan2_10cmTransients])
rmsstarttime10cm.append([i.startTime for i in rmschan3_10cmTransients])
rmsstarttime10cm.append([i.startTime for i in rmschan4_10cmTransients])
rmsstarttime10cm = list(itertools.chain.from_iterable(rmsstarttime10cm))
np.savetxt(wd+'rmsstart10cm', rmsstarttime10cm)

rmsstarttime15cm = list()
rmsstarttime15cm.append([i.startTime for i in rmschan1_15cmTransients])
rmsstarttime15cm.append([i.startTime for i in rmschan2_15cmTransients])
rmsstarttime15cm.append([i.startTime for i in rmschan3_15cmTransients])
rmsstarttime15cm.append([i.startTime for i in rmschan4_15cmTransients])
rmsstarttime15cm = list(itertools.chain.from_iterable(rmsstarttime15cm))
np.savetxt(wd+'rmsstart15cm', rmsstarttime15cm)
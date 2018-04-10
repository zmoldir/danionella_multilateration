import matplotlib.pyplot as plt
import multilateration as mult
import soundfile as sf
import numpy as np
import sys
import seaborn as sb
import pandas as pdsys.path.insert(0, '/home/maximilian/pycharm/')
sys.path.insert(0, '/home/maximilian/unistuff/paris_ens/cal_neuroim/pystuff/')

wd = '/home/maximilian/jlab_fishies/jupyter_notebooks/megaSamples/'

''' Todo here: 
- read amp50 files
- read & normalize data
- plot correctly
then:
get transients (ugh thats what I'm trying to avoid ... )
- violin plots
'''

sampRate = 65000
highPassCutoff = 400
rmse = False
cutLength = 6500
'''
testPulse = sf.read(wd+'testPulse.wav')[0][cutLength:]

channel2 = np.mean(mult.filterMatrix(sampRate,sf.read(wd+'channel2.wav')[0].T,
                                     highPassCutoff,envelopeBool=rmse)[:,cutLength:],axis=0)
channel2_5cm = np.mean(mult.filterMatrix(sampRate,
                                         sf.read(wd+'channel2_5cm.wav')[0].T,highPassCutoff,
                                         envelopeBool=rmse)[:,cutLength:],axis=0)
channel2_10cm = np.mean(mult.filterMatrix(sampRate,
                                          sf.read(wd+'channel2_10cm.wav')[0].T,highPassCutoff,
                                          envelopeBool=rmse)[:,cutLength:],axis=0)
channel2_15cm = np.mean(mult.filterMatrix(sampRate,
                                          sf.read(wd+'channel2_15cm.wav')[0].T,highPassCutoff,
                                          envelopeBool=rmse)[:,cutLength:],axis=0)

rmse = True
rmschannel2 = np.mean(mult.filterMatrix(sampRate, sf.read(wd+'channel2.wav')[0].T,highPassCutoff,
                                        envelopeBool=rmse)[:,cutLength:],axis=0)
rmschannel2_5cm = np.mean(mult.filterMatrix(sampRate, sf.read(wd+'channel2_5cm.wav')[0].T,highPassCutoff,
                                            envelopeBool=rmse)[:,cutLength:],axis=0)
rmschannel2_10cm = np.mean(mult.filterMatrix(sampRate, sf.read(wd+'channel2_10cm.wav')[0].T,highPassCutoff,
                                             envelopeBool=rmse)[:,cutLength:],axis=0)
rmschannel2_15cm = np.mean(mult.filterMatrix(sampRate, sf.read(wd+'channel2_15cm.wav')[0].T,highPassCutoff,
                                             envelopeBool=rmse)[:,cutLength:],axis=0)
'''
rmsamp50 = np.loadtxt(wd+"rmsamp50")
rmscm5amp50 = np.loadtxt(wd+"rmscm5amp50")
rmscm10amp50 = np.loadtxt(wd+"rmscm10amp50")
rmscm15amp50 = np.loadtxt(wd+"rmscm15amp50")

amp50 = np.loadtxt(wd+'amp50')
cm5amp50 = np.loadtxt(wd+"cm5amp50")
cm10amp50 = np.loadtxt(wd+"cm10amp50")
cm15amp50 = np.loadtxt(wd+"cm15amp50")

start = np.loadtxt(wd+"start0cm")[30:59]
start5cm = np.loadtxt(wd+"start5cm")[30:59]
start10cm = np.loadtxt(wd+"start10cm")[30:59]
start15cm = np.loadtxt(wd+"start15cm")[30:59]

rmsstart = np.loadtxt(wd+"rmsstart0cm")[30:59]
rmsstart5cm = np.loadtxt(wd+"rmsstart5cm")[30:59]
rmsstart10cm = np.loadtxt(wd+"rmsstart10cm")[30:59]
rmsstart15cm = np.loadtxt(wd+"rmsstart15cm")[30:59]


sb.set_style("whitegrid")
test = sb.violinplot(data=[rmsamp50,rmscm5amp50,rmscm10amp50,rmscm15amp50], split=True)
plt.xlabel("Distance between emitter and receiver")
plt.ylabel("Sample index of 50% amplitude value, post-rms")
plt.xticks(range(4), ['2cm','7cm','12cm','17cm'])
plt.savefig('AmplitudeRMS_violinplot.png')
plt.close()
sb.set_style("whitegrid")
test = sb.violinplot(data=[amp50,cm5amp50,cm10amp50,cm15amp50],split=True)
plt.xlabel("Distance between emitter and receiver")
plt.ylabel("Sample index of 50% amplitude value")
plt.xticks(range(4), ['2cm','7cm','12cm','17cm'])
plt.savefig('Amplitude_violinplot.png')
plt.close()
sb.set_style("whitegrid")
test = sb.violinplot(data=[start, start5cm, start10cm, start15cm],split=True)
plt.xlabel("Distance between emitter and receiver")
plt.ylabel("Sample index of signal onset")
plt.xticks(range(4), ['2cm','7cm','12cm','17cm'])
plt.savefig('Onset_violinplot.png')
plt.close()
test = sb.violinplot(data=[rmsstart, rmsstart5cm, rmsstart10cm, rmsstart15cm],split=True)
plt.xlabel("Distance between emitter and receiver")
plt.ylabel("Sample index of signal onset, post-rms")
plt.xticks(range(4), ['2cm','7cm','12cm','17cm'])
plt.savefig('OnsetRMS_violinplot.png')
plt.close()
'''
pos1 = rmsstart
pos2 = pos1 + 200
print("positions are : {0} - {1}".format(pos1, pos2))

tvec = np.linspace(0,float(pos2-pos1)/sampRate*1000,pos2-pos1)
plt.plot(tvec,testPulse[pos1:pos2]-2,label='test signal')
plt.plot(tvec,rmschannel2[pos1:pos2],'grey',alpha=0.9)
plt.plot(tvec,rmschannel2_5cm[pos1:pos2]+1,'grey',alpha=0.9)
plt.plot(tvec,rmschannel2_10cm[pos1:pos2]+2,'grey',alpha=0.9)
plt.plot(tvec,rmschannel2_15cm[pos1:pos2]+3,'grey',alpha=0.9)


plt.xlabel('time in ms')
plt.ylabel('signal intensity + offset')

testPulse2 = mult.envelope(62500,testPulse)
plt.plot(tvec, testPulse2[pos1:pos2]-2,label='rms test signal',alpha=0.9,color='grey')
plt.plot(tvec, channel2[pos1:pos2],label="2cm")
plt.plot(tvec, channel2_5cm[pos1:pos2]+1,label="7cm")
plt.plot(tvec, channel2_10cm[pos1:pos2]+2,label="12cm")
plt.plot(tvec, channel2_15cm[pos1:pos2]+3,label="17cm")
plt.legend(loc=4)


# problem: amp50 is relative to start time, not plotting coordinate
# solution: add starttime instead of pos1
plt.plot([tvec[np.mean(rmsamp50[30:59], dtype=int)-pos1]],
         rmschannel2[np.mean(rmsamp50[30:59],dtype=int)], marker='o',markersize=3,color='black')
plt.plot([tvec[np.mean(rmscm5amp50[30:59]-pos1, dtype=int)]],
         rmschannel2_5cm[np.mean(rmscm5amp50[30:59],dtype=int)]+1, marker='o',markersize=3,color='black')
plt.plot([tvec[np.mean(rmscm10amp50[30:59], dtype=int)-pos1]],
         rmschannel2_10cm[np.mean(rmscm10amp50[30:59],dtype=int)]+2, marker='o',markersize=3,color='black')
plt.plot([tvec[np.mean(rmscm15amp50[30:59], dtype=int)-pos1]],
         rmschannel2_15cm[np.mean(rmscm15amp50[30:59],dtype=int)]+3,marker='o',markersize=3,color='black')

# non-rms positions
plt.plot([tvec[np.mean(amp50[30:59], dtype=int)-pos1-5]],
         channel2[np.mean(amp50[30:59],dtype=int)-5],marker='^',markersize=4,color='b')

plt.plot([tvec[np.mean(cm5amp50[30:59], dtype=int)-pos1-4]],
         channel2_5cm[np.mean(cm5amp50[30:59],dtype=int)-4]+1,marker='^',markersize=4,color='b')

plt.plot([tvec[np.mean(cm10amp50[30:59], dtype=int)-pos1]],
         channel2_10cm[np.mean(cm10amp50[30:59],dtype=int)]+2,marker='^',markersize=4,color='b')
plt.plot([tvec[np.mean(cm15amp50[30:59], dtype=int)-pos1]],
         channel2_15cm[np.mean(cm15amp50[30:59],dtype=int)]+3, marker='^',markersize=4,color='b')

plt.show()
plt.close()
'''

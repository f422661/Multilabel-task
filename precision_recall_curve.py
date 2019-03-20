import numpy as np
import argparse
import matplotlib.pyplot as plt

## dataset1
# precison = [0.7174823,0.78141095,0.81841213,0.84553563,0.87451518,0.90002543,0.92459673,0.94632916,0.96492429]
# recall = [0.88374249,0.8557132,0.83259048,0.81239254,0.77996905,0.75328408,0.71926329,0.68314313,0.63584742]

## dataset6

precision = [0.66673153,0.73965089,0.77930114,0.80828711,0.83614478,0.86472363,0.89516862,0.92814967,0.9553824 ]
recall = [0.86785554,0.82996005,0.79504904,0.76476115,0.7377286,0.70599019,0.66623692,0.61995572,0.56000462]


n = np.round(np.arange(0.1,1.0,0.1),decimals=1)


fig, ax = plt.subplots()


# plt.figure()
plt.title("Precision-Recall curve")
plt.plot(precision,recall,'o-',label = '$\eta_t = 5$')
# plt.plot(avg_fpr[1],avg_tpr[1],'v-',label = '$\eta_t = 10$')
# plt.plot(avg_fpr[2],avg_tpr[2],'+-',label = '$\eta_t = 15$')
# plt.plot(avg_fpr[3],avg_tpr[3],'x-',label = '$\eta_t = 20$')

for i, txt in enumerate(n):
  ax.annotate(txt, (precision[i], recall[i]))

plt.xlabel('Precision',rotation ='horizontal')
plt.ylabel('Recall',rotation ='vertical')

# plt.xticks([0.2,0.4,0.6,0.8,1,0])
# plt.yticks([0.2,0.4,0.6,0.8,1,0])
# plt.xlim(0.0,0.06)
# plt.ylim(0.95,1.0)
# plt.legend(loc = 'lower right')

# # plt.xlim(0,len(wave[0]))
# plt.savefig("p_roc_cv.eps")

plt.show()
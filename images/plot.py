
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
# Timing Data 
##########################################################################################
n_groups = 5

right = [5.464159e+01, 3.492398e+01, 2.681859e+01, 2.529457e+01, 6.645727e+01]

left = [1.009287e+03, 1.278701e+03, 8.069940e+02, 5.313168e+02, 6.562814e+00]

##########################################################################################
# Plot Speedup Compared to OMP 1 LayoutRight 
##########################################################################################

base = right[0]

right_speedup = [base/i for i in right]
left_speedup  = [base/i for i in left ]

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4

rects1 = ax.bar(index, right_speedup, bar_width,
                alpha=opacity, color='b',
                label='LayoutRight')

rects2 = ax.bar(index + bar_width, left_speedup, bar_width,
                alpha=opacity, color='r',
                label='LayoutLeft')

#ax.set_xlabel('')
ax.set_ylabel('Speedup compared to OMP 1 LayoutRight')
#ax.set_title('')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('OMP 1', 'OMP 2', 'OMP 4', 'OMP 8', 'CUDA'))
ax.legend()

fig.tight_layout()

plt.savefig("speedup.png")
plt.show()

##########################################################################################
# Plot Speedup From Optimal Memory Layout
##########################################################################################

left_over_right = [l/r for l, r in zip(left, right)]
right_over_left = [r/l for l, r in zip(left, right)]

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4

rects1 = ax.bar(index[:4],  left_over_right[:4], bar_width,
                alpha=opacity, color='b',
                label='LayoutRight')

rects2 = ax.bar(index[-1], right_over_left[-1], bar_width,
                alpha=opacity, color='r',
                label='LayoutLeft')

#ax.set_xlabel('')
ax.set_ylabel('Speedup From Optimal Layout')
#ax.set_title('')
ax.set_xticks(index)
ax.set_xticklabels(('OMP 1', 'OMP 2', 'OMP 4', 'OMP 8', 'CUDA'))
ax.legend()

fig.tight_layout()

plt.savefig("optimal_speedup.png")
plt.show()

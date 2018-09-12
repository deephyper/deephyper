import matplotlib.pyplot as plt

import vpg_mathfun as vpg
import ppo_mathfun as ppo
import rdm_mathfun as rdm

def max_list(l):
    rl = [l[0]]
    mx = l[0]
    for i in range(1, len(l)):
        mx = max(mx, l[i])
        rl.append(mx)
    return rl

def save_plot(name):
    plt.savefig(name+'.png', dpi=200)

vpg_rewards = vpg.go_learning()
ppo_rewards = ppo.go_learning()
rdm_rewards = rdm.go_learning()

x = [i for i in range(len(vpg_rewards))]
func_name = 'griewank'

plt.title(f'{func_name} 10 dim')
plt.figure(figsize=(16, 10), dpi=300, facecolor='w', edgecolor='k')
color = (63/255, 191/255, 63/255, 0.6)
plt.plot(x, vpg_rewards, color=color, label='vpg raw')
color = (3/255, 179/255, 3/255, 0.6)
plt.plot(x, max_list(vpg_rewards), color=color, label='vpg max')


color = (107/255, 158/255, 208/255, 0.8)
plt.plot(x, ppo_rewards, color=color, label='ppo raw')
color = (3/255, 91/255, 179/255, 0.8)
plt.plot(x, max_list(ppo_rewards), color=color, label='ppo max')

color = (191/255, 63/255, 191/255, 1.)
plt.plot(x, rdm_rewards, color=color, label='rdm raw')
color = (243/255, 108/255, 243/255, 1.)
plt.plot(x, max_list(rdm_rewards), color=color, label='rdm max')

plt.legend()
save_plot(f'{func_name}_10_dim')

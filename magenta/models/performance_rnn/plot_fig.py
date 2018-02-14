import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random

num_sections = 10
num_first_actions = 10
with open("/home/zha231/data/output_log_first_actions.csv", "rb") as f:
    spamreader = csv.reader(f)
    first_actions = []
    for row in spamreader:
        first_actions.append(row)

logs_100_act_org = []
log_action_smc_divid = []
log_action_brutal_divid = []
for j in range(num_sections):
    # 0(org): 1--num_sections, num_sections+1(smc):  num_sections+2 -- 2*num_sections+1
    # 2*num_sections+2(bru): 2*num_sections+3 -- 3*num_sections+2
    act_org = [float(first_actions[j+1][i]) for i in range(num_first_actions)]
    logs_100_act_org.append(act_org)
    act_smc = [float(first_actions[j+num_sections+2][i]) for i in range(num_first_actions)]
    log_action_smc_divid.append(act_smc)
    act_brutal = [float(first_actions[j+2*num_sections+3][i]) for i in range(num_first_actions)]
    log_action_brutal_divid.append(act_brutal)

log_plot_org = np.array(logs_100_act_org)
df = pd.DataFrame(log_plot_org)
pl_fig = plt.figure()
df.boxplot()
pl_fig.savefig('log_plot_org.eps')
plt.close()

log_plot_smc = np.array(log_action_smc_divid)
df = pd.DataFrame(log_plot_smc)
pl_fig = plt.figure()
df.boxplot()
pl_fig.savefig('log_plot_smc.eps')
plt.close()

log_plot_brutal = np.array(log_action_brutal_divid)
df = pd.DataFrame(log_plot_brutal)
pl_fig = plt.figure()
df.boxplot()
pl_fig.savefig('log_plot_brutal.eps')
plt.close()

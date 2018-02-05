import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random

num_selected_sections = 4
num_first_actions = 10
with open("/home/zha231/data/output_log_first_actions.csv", "rb") as f:
    spamreader = csv.reader(f)
    first_actions = []
    for row in spamreader:
        first_actions.append(row)

logs_100_act_org = []
log_action_smc_divid = []
log_action_brutal_divid = []
for j in range(num_selected_sections):
    # 0, 1-4, 5,6-9,10-13, 14, 15-18,19-22
    act_org = [float(first_actions[j+1][i]) for i in range(num_first_actions)]
    logs_100_act_org.append(act_org)
    act_smc = [float(first_actions[j+10][i]) for i in range(num_first_actions)]
    log_action_smc_divid.append(act_smc)
    act_brutal = [float(first_actions[j+19][i]) for i in range(num_first_actions)]
    log_action_brutal_divid.append(act_brutal)

log_plot_org = np.array(logs_100_act_org)
df = pd.DataFrame(log_plot_org)
pl_fig = plt.figure()
df.boxplot()
pl_fig.savefig('data/log_plot_org.eps')
log_plot_smc = np.array(log_action_smc_divid)
df = pd.DataFrame(log_plot_smc)
df.boxplot()
plt.savefig('data/log_plot_smc.eps')
log_plot_brutal = np.array(log_action_brutal_divid)
df = pd.DataFrame(log_plot_brutal)
df.boxplot()
plt.savefig('data/log_plot_brutal.eps')
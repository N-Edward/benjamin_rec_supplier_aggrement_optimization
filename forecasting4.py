# Renewable Energy Community — Supplier Agreement Optimization with Forecasting (Pyomo)
# ---------------------------------------------------------------
# This template models a REC that must choose ONE supplier contract and
# schedule its day-ahead import/export while managing flexibility (battery),
# under forecast uncertainty represented by scenarios. Objective: minimize
# expected cost with optional CVaR risk aversion.

from pyomo import environ as pyo
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1) Example input data (REPLACE with your data)
# ----------------------------
T = list(range(24))                   # 24 hourly periods (example day)
S = list(range(10))                   # 10 forecast scenarios
K = ["SupplierA", "SupplierB"]       # available supplier contracts

p_s = {s: 1/len(S) for s in S}        # scenario probabilities (uniform demo)
alpha = None                          # CVaR disabled for now

# Contract parameters (replace with real tariffs)
p_import = {(k,t): 0.20 + 0.05*(t in range(17,21)) for k in K for t in T}
p_export = {(k,t): 0.08 for k in K for t in T}
fixed_fee = {"SupplierA": 50.0, "SupplierB": 30.0}
cap_charge = {"SupplierA": 10.0, "SupplierB": 14.0}

imb_up  = {(k,t): 0.25 for k in K for t in T}
imb_down= {(k,t): 0.05 for k in K for t in T}

import math
NL = {(s,t): 5.0 + 2.0*math.sin(2*math.pi*(t/24)) + 0.5*math.sin(2*math.pi*(t/3 + s/7))
      for s in S for t in T}

E_max, P_max = 50.0, 15.0
eta_c, eta_d = 0.95, 0.95
soc0 = 0.5*E_max
soc_min, soc_max = 0.1*E_max, 0.9*E_max

BIGM = 1e4

# ----------------------------
# 2) Model
# ----------------------------
model = pyo.ConcreteModel()

model.T = pyo.Set(initialize=T, ordered=True)
model.S = pyo.Set(initialize=S)
model.K = pyo.Set(initialize=K)

model.y = pyo.Var(model.K, within=pyo.Binary)
model.q = pyo.Var(model.T, within=pyo.NonNegativeReals)

model.g_imp = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.g_exp = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

model.dev_pos = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.dev_neg = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

model.ch  = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.dis = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.soc = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

model.cap = pyo.Var(within=pyo.NonNegativeReals)

# ----------------------------
# 3) Constraints
# ----------------------------
model.one_supplier = pyo.Constraint(expr=sum(model.y[k] for k in model.K) == 1)

def balance_rule(m, s, t):
    return m.g_imp[s,t] + m.dis[s,t] - m.ch[s,t] - m.g_exp[s,t] == NL[(s,t)]
model.balance = pyo.Constraint(model.S, model.T, rule=balance_rule)

def dev_pos_rule(m, s, t):
    return m.dev_pos[s,t] >= m.g_imp[s,t] - m.q[t]
model.dev_pos_c = pyo.Constraint(model.S, model.T, rule=dev_pos_rule)

def dev_neg_rule(m, s, t):
    return m.dev_neg[s,t] >= m.q[t] - m.g_imp[s,t]
model.dev_neg_c = pyo.Constraint(model.S, model.T, rule=dev_neg_rule)

h = 1.0

def soc_rule(m, s, t):
    t_idx = list(m.T).index(t)
    if t_idx == 0:
        return m.soc[s,t] == soc0 + (eta_c*m.ch[s,t] - (1/eta_d)*m.dis[s,t])
    t_prev = list(m.T)[t_idx-1]
    return m.soc[s,t] == m.soc[s,t_prev] + (eta_c*m.ch[s,t] - (1/eta_d)*m.dis[s,t])
model.soc_dyn = pyo.Constraint(model.S, model.T, rule=soc_rule)

model.soc_bounds_lo = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.soc[s,t] >= soc_min)
model.soc_bounds_hi = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.soc[s,t] <= soc_max)

model.charge_limit = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.ch[s,t] <= P_max*h)
model.discharge_limit = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.dis[s,t] <= P_max*h)

def peak_rule(m, s, t, k):
    return m.g_imp[s,t] <= m.cap + BIGM*(1 - m.y[k])
model.peak_cap = pyo.Constraint(model.S, model.T, model.K, rule=peak_rule)

# ----------------------------
# 4) Objective with cost components
# ----------------------------

def cost_components(m):
    chosen_k = [k for k in m.K if pyo.value(m.y[k]) > 0.5][0]
    da = sum(p_import[(chosen_k,t)]*m.q[t] for t in m.T)
    imb = sum(p_s[s]*sum(imb_up[(chosen_k,t)]*m.dev_pos[s,t] - imb_down[(chosen_k,t)]*m.dev_neg[s,t] for t in m.T) for s in m.S)
    exp_rev = sum(p_s[s]*sum(p_export[(chosen_k,t)]*m.g_exp[s,t] for t in m.T) for s in m.S)
    fixed = fixed_fee[chosen_k]
    capc = cap_charge[chosen_k]*pyo.value(m.cap)
    return da, imb, exp_rev, fixed, capc


def energy_cost(m):
    da, imb, exp_rev, fixed, capc = cost_components(m)
    return da + imb - exp_rev + fixed + capc

model.total_cost = pyo.Objective(expr=energy_cost(model), sense=pyo.minimize)

# ----------------------------
# 5) Solve with Pyomo Solver
# ----------------------------
solver = pyo.SolverFactory("glpk")
result = solver.solve(model, tee=True)

print(result.solver.status, result.solver.termination_condition)

chosen = [k for k in K if pyo.value(model.y[k]) > 0.5][0]
print("Chosen supplier:", chosen)
print("Contracted capacity (kW):", round(pyo.value(model.cap),2))
DA = {t: round(pyo.value(model.q[t]),2) for t in T}
print("Day-ahead schedule:", DA)
print("Expected cost:", round(pyo.value(energy_cost(model)),2))

# ----------------------------
# 6) Visualization
# ----------------------------
plt.figure(figsize=(10,6))
for s in S:
    plt.plot(T, [NL[(s,t)] for t in T], color='gray', alpha=0.3)
plt.plot(T, [DA[t] for t in T], color='blue', linewidth=2, label='Day-ahead import schedule')
plt.xlabel('Hour of day')
plt.ylabel('Energy [kWh]')
plt.title('REC Supplier Optimization: Day-ahead Schedule vs. Forecast Scenarios')
plt.legend()
plt.grid(True)
plt.show()

sample_s = S[0]
ch = [pyo.value(model.ch[sample_s,t]) for t in T]
dis = [pyo.value(model.dis[sample_s,t]) for t in T]
soc = [pyo.value(model.soc[sample_s,t]) for t in T]

soc_all = np.array([[pyo.value(model.soc[s,t]) for t in T] for s in S])
soc_min_range = soc_all.min(axis=0)
soc_max_range = soc_all.max(axis=0)

plt.figure(figsize=(10,6))
plt.plot(T, ch, label='Charge [kWh]', color='green')
plt.plot(T, dis, label='Discharge [kWh]', color='red')
plt.plot(T, soc, label=f'SoC Scenario {sample_s} [kWh]', color='purple', linewidth=2)
plt.fill_between(T, soc_min_range, soc_max_range, color='purple', alpha=0.2, label='SoC range across scenarios')
plt.xlabel('Hour of day')
plt.ylabel('Energy [kWh]')
plt.title('Battery Operation and SoC Ranges')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 7) Cost breakdown bar chart
# ----------------------------
da, imb, exp_rev, fixed, capc = cost_components(model)
costs = [float(da), float(imb), -float(exp_rev), float(fixed), float(capc)]
labels = ["Day-ahead energy", "Imbalance", "Export revenue", "Fixed fee", "Capacity charge"]

plt.figure(figsize=(8,6))
plt.bar(labels, costs, color=['blue','orange','green','gray','red'])
plt.ylabel('Cost / Revenue [€]')
plt.title(f'Cost Breakdown for {chosen}')
plt.xticks(rotation=30, ha='right')
plt.grid(True, axis='y')
plt.show()


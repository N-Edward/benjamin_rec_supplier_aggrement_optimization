
# Renewable Energy Community — Supplier Agreement Optimization with Forecasting (Pyomo)
# Extended: Includes sensitivity analysis across different numbers of forecast scenarios
# ---------------------------------------------------------------

from pyomo import environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import math

# ----------------------------
# Function to build and solve model
# ----------------------------
def solve_rec_model(T, S, NL, p_import, p_export, fixed_fee, cap_charge, imb_up, imb_down,
                    E_max=50.0, P_max=15.0, eta_c=0.95, eta_d=0.95, soc0=None, soc_min=None, soc_max=None):
    if soc0 is None: soc0 = 0.5*E_max
    if soc_min is None: soc_min = 0.1*E_max
    if soc_max is None: soc_max = 0.9*E_max

    K = list(fixed_fee.keys())
    p_s = {s: 1/len(S) for s in S}
    BIGM = 1e4

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

    # Constraints
    model.one_supplier = pyo.Constraint(expr=sum(model.y[k] for k in model.K) == 1)

    def balance_rule(m, s, t):
        return m.g_imp[s,t] + m.dis[s,t] - m.ch[s,t] - m.g_exp[s,t] == NL[(s,t)]
    model.balance = pyo.Constraint(model.S, model.T, rule=balance_rule)

    model.dev_pos_c = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.dev_pos[s,t] >= m.g_imp[s,t] - m.q[t])
    model.dev_neg_c = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.dev_neg[s,t] >= m.q[t] - m.g_imp[s,t])

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

    # Objective
    def energy_cost(m):
        da = sum( sum( p_import[(k,t)]*m.q[t] for t in m.T ) * m.y[k] for k in m.K )
        imb = sum( p_s[s] * sum( sum( imb_up[(k,t)]*m.dev_pos[s,t] - imb_down[(k,t)]*m.dev_neg[s,t]
                                    for t in m.T ) * m.y[k]
                                for k in m.K ) for s in m.S )
        exp_rev = sum( p_s[s] * sum( sum( p_export[(k,t)]*m.g_exp[s,t] for t in m.T ) * m.y[k]
                               for k in m.K ) for s in m.S )
        fixed = sum( fixed_fee[k]*m.y[k] for k in m.K )
        cap   = sum( cap_charge[k]*m.cap*m.y[k] for k in m.K )
        return da + imb - exp_rev + fixed + cap

    model.total_cost = pyo.Objective(expr=energy_cost(model), sense=pyo.minimize)

    solver = pyo.SolverFactory("glpk")
    result = solver.solve(model, tee=False)

    chosen = [k for k in K if pyo.value(model.y[k]) > 0.5][0]
    return chosen, pyo.value(model.total_cost)

# ----------------------------
# 1) Base parameters
# ----------------------------
T = list(range(24))
K = ["SupplierA", "SupplierB"]

p_import = {(k,t): 0.20 + 0.05*(t in range(17,21)) for k in K for t in T}
p_export = {(k,t): 0.08 for k in K for t in T}
fixed_fee = {"SupplierA": 50.0, "SupplierB": 30.0}
cap_charge = {"SupplierA": 10.0, "SupplierB": 14.0}
imb_up  = {(k,t): 0.25 for k in K for t in T}
imb_down= {(k,t): 0.05 for k in K for t in T}

# ----------------------------
# 2) Sensitivity analysis: vary scenario numbers & volatility
# ----------------------------
scenario_counts = [5, 10, 20, 40]
volatilities = [0.2, 0.5, 1.0]  # scaling factors for uncertainty

results = []

for vol in volatilities:
    for nS in scenario_counts:
        S = list(range(nS))
        NL = {(s,t): 5.0 + 2.0*math.sin(2*math.pi*(t/24)) + vol*math.sin(2*math.pi*(t/3 + s/7))
              for s in S for t in T}
        chosen, cost = solve_rec_model(T, S, NL, p_import, p_export, fixed_fee, cap_charge, imb_up, imb_down)
        results.append({"scenarios": nS, "volatility": vol, "supplier": chosen, "cost": cost})

# ----------------------------
# 3) Plot sensitivity results
# ----------------------------
import pandas as pd
results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10,6))
for vol in volatilities:
    subset = results_df[results_df["volatility"]==vol]
    ax.plot(subset["scenarios"], subset["cost"], marker='o', label=f'Vol={vol}')
    for i,row in subset.iterrows():
        ax.text(row["scenarios"], row["cost"]+2, row["supplier"], ha='center', fontsize=8)

ax.set_xlabel("Number of forecast scenarios")
ax.set_ylabel("Expected cost [€]")
ax.set_title("Sensitivity of Supplier Choice to Scenario Number & Volatility")
ax.legend()
ax.grid(True)
plt.show()

print(results_df)

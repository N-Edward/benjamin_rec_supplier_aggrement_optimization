# Renewable Energy Community — Supplier Agreement Optimization with Forecasting (Pyomo)
# ---------------------------------------------------------------
# This template models a REC that must choose ONE supplier contract and
# schedule its day-ahead import/export while managing flexibility (battery),
# under forecast uncertainty represented by scenarios. Objective: minimize
# expected cost with optional CVaR risk aversion.
#
# Features:
# - Contract choice (binary), one active supplier k∈K
# - Day-ahead schedule q[t] and real-time imbalance deviations
# - Scenario-based net load from forecasts (demand - RES)
# - Battery (charge/discharge/SoC) with efficiency and power/energy limits
# - Capacity charge (peak import), fixed monthly fee, energy prices (TOU)
# - Import/export with prices by contract
# - CVaR(α) risk term to penalize expensive tails (optional)
#
# Notes for Austrian RECs (adjust to your tariff model):
# - Supplier bears BRP; imbalance settlement costs often passed via formulas
# - Network tariff specifics (e.g., reduced tariffs within REC) can be added
#   as additional terms/constraints if needed.

from pyomo import environ as pyo

# ----------------------------
# 1) Example input data (REPLACE with your data)
# ----------------------------
T = list(range(24))                   # 24 hourly periods (example day)
S = list(range(20))                   # 20 forecast scenarios
K = ["SupplierA", "SupplierB"]       # available supplier contracts

p_s = {s: 1/len(S) for s in S}        # scenario probabilities (uniform demo)
alpha = 0.9                           # CVaR confidence level (set None to disable)

# Contract parameters (replace with real tariffs)
# TOU import/export prices per hour and contract
p_import = {(k,t): 0.20 + 0.05*(t in range(17,21)) for k in K for t in T}  # €/kWh
p_export = {(k,t): 0.08 for k in K for t in T}                              # €/kWh feed-in
fixed_fee = {"SupplierA": 50.0, "SupplierB": 30.0}                        # €/month
cap_charge = {"SupplierA": 10.0, "SupplierB": 14.0}                       # €/kW-month

# Imbalance prices (up/down) per hour and contract (simplified)
# Positive deviation = buy extra in RT (up price); Negative deviation = sell back (down price)
imb_up  = {(k,t): 0.25 for k in K for t in T}   # €/kWh
imb_down= {(k,t): 0.05 for k in K for t in T}   # €/kWh

# Scenario net load NL[s,t] = demand - on-site generation (kWh in the hour)
# Positive => need import; Negative => surplus that can be exported or stored.
# Replace these with your forecast scenario matrix.
import math
NL = {(s,t): 5.0 + 2.0*math.sin(2*math.pi*(t/24)) + 0.5*math.sin(2*math.pi*(t/3 + s/7))
      for s in S for t in T}

# Battery parameters (set to zero to disable flexibility)
E_max = 50.0    # kWh usable capacity
P_max = 15.0    # kW charge/discharge limit per hour
eta_c = 0.95
eta_d = 0.95
soc0  = 0.5*E_max
soc_min, soc_max = 0.1*E_max, 0.9*E_max

# Big-M for contract-conditional constraints
BIGM = 1e4

# ----------------------------
# 2) Model
# ----------------------------
model = pyo.ConcreteModel()

model.T = pyo.Set(initialize=T, ordered=True)
model.S = pyo.Set(initialize=S)
model.K = pyo.Set(initialize=K)

# Contract choice: one supplier active
model.y = pyo.Var(model.K, within=pyo.Binary)

# Day-ahead import schedule (decided before scenarios realize)
model.q = pyo.Var(model.T, within=pyo.NonNegativeReals)

# Scenario-realized variables
model.g_imp = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)  # grid import (kWh)
model.g_exp = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)  # grid export (kWh)

# Imbalance decomposition: dev = g_imp - q
model.dev_pos = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.dev_neg = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

# Battery
model.ch  = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.dis = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.soc = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

# Contracted import capacity for capacity charge and peak constraint
model.cap = pyo.Var(within=pyo.NonNegativeReals)

# Optional CVaR variables
if alpha is not None:
    model.eta = pyo.Var()  # CVaR auxiliary
    model.z   = pyo.Var(model.S, within=pyo.NonNegativeReals)

# ----------------------------
# 3) Constraints
# ----------------------------

# One supplier must be chosen
model.one_supplier = pyo.Constraint(expr=sum(model.y[k] for k in model.K) == 1)

# Balance per scenario and time: imports + discharge - charge - exports = net load
def balance_rule(m, s, t):
    return m.g_imp[s,t] + m.dis[s,t] - m.ch[s,t] - m.g_exp[s,t] == NL[(s,t)]
model.balance = pyo.Constraint(model.S, model.T, rule=balance_rule)

# Deviation = g_imp - q split into positive/negative parts
def dev_pos_rule(m, s, t):
    return m.dev_pos[s,t] >= m.g_imp[s,t] - m.q[t]
model.dev_pos_c = pyo.Constraint(model.S, model.T, rule=dev_pos_rule)

def dev_neg_rule(m, s, t):
    return m.dev_neg[s,t] >= m.q[t] - m.g_imp[s,t]
model.dev_neg_c = pyo.Constraint(model.S, model.T, rule=dev_neg_rule)

# Battery dynamics
h = 1.0  # hour step

def soc_rule(m, s, t):
    t_idx = list(m.T).index(t)
    if t_idx == 0:
        return m.soc[s,t] == soc0 + (eta_c*m.ch[s,t] - (1/eta_d)*m.dis[s,t])
    t_prev = list(m.T)[t_idx-1]
    return m.soc[s,t] == m.soc[s,t_prev] + (eta_c*m.ch[s,t] - (1/eta_d)*m.dis[s,t])
model.soc_dyn = pyo.Constraint(model.S, model.T, rule=soc_rule)

model.soc_bounds_lo = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.soc[s,t] >= soc_min)
model.soc_bounds_hi = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.soc[s,t] <= soc_max)

# Power limits
model.charge_limit = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.ch[s,t]  <= P_max*h)
model.discharge_limit = pyo.Constraint(model.S, model.T, rule=lambda m,s,t: m.dis[s,t] <= P_max*h)

# Peak import cannot exceed contracted capacity (with contract activation)
# g_imp[s,t] <= cap + BIGM*(1 - y_k) for all k; ensures binding only for chosen k
# Equivalent: g_imp <= cap + BIGM*(1 - sum y_k) with sum y_k=1, but we keep explicit linking to show structure.

def peak_rule(m, s, t, k):
    return m.g_imp[s,t] <= m.cap + BIGM*(1 - m.y[k])
model.peak_cap = pyo.Constraint(model.S, model.T, model.K, rule=peak_rule)

# ----------------------------
# 4) Objective: Expected cost + CVaR (optional)
# ----------------------------

def energy_cost(m):
    # Day-ahead energy cost depends on chosen contract prices
    da = sum( sum( p_import[(k,t)]*m.q[t] for t in m.T ) * m.y[k] for k in m.K )

    # Expected imbalance costs by contract
    imb = sum( p_s[s] * sum( sum( imb_up[(k,t)]*m.dev_pos[s,t] +
                                  (-imb_down[(k,t)]*m.dev_neg[s,t])  # revenue (negative cost)
                                for t in m.T ) * m.y[k]
                            for k in m.K ) for s in m.S )

    # Import/export settlement at RT price can be embedded into imbalance;
    # here we keep DA for q and imbalance for deviations. If you prefer pure RT
    # settlement, shift terms accordingly.

    # Export revenue at contract feed-in price (apply to actual exports)
    exp_rev = sum( p_s[s] * sum( sum( p_export[(k,t)]*m.g_exp[s,t] for t in m.T ) * m.y[k]
                           for k in m.K ) for s in m.S )

    # Fixed fee and capacity charge (only for active contract)
    fixed = sum( fixed_fee[k]*m.y[k] for k in m.K )
    cap   = sum( cap_charge[k]*m.cap*m.y[k] for k in m.K )

    return da + imb - exp_rev + fixed + cap

# Scenario cost for CVaR (without eta/z). We reuse expected-decomposition but per scenario.
# For CVaR we need each scenario's realized cost.

def scen_cost(m, s):
    # Contract-weighted prices
    da = sum( sum( p_import[(k,t)]*m.q[t] for t in m.T ) * m.y[k] for k in m.K )
    imb = sum( sum( imb_up[(k,t)]*m.dev_pos[s,t] + (-imb_down[(k,t)]*m.dev_neg[s,t])
                    for t in m.T ) * m.y[k] for k in m.K )
    exp_rev = sum( sum( p_export[(k,t)]*m.g_exp[s,t] for t in m.T ) * m.y[k] for k in m.K )
    fixed = sum( fixed_fee[k]*m.y[k] for k in m.K )
    cap   = sum( cap_charge[k]*m.cap*m.y[k] for k in m.K )
    return da + imb - exp_rev + fixed + cap

if alpha is None:
    model.total_cost = pyo.Objective(expr=energy_cost(model), sense=pyo.minimize)
else:
    # CVaR augmentation: minimize E[cost] + λ * CVaR; here λ=1 by default
    lam = 1.0
    # Link scenario excess variables to scenario cost
    def z_link(m, s):
        return m.z[s] >= scen_cost(m, s) - m.eta
    model.cvar_link = pyo.Constraint(model.S, rule=z_link)

    cvar_term = model.eta + (1/(1-alpha))*sum(p_s[s]*model.z[s] for s in model.S)
    model.total_cost = pyo.Objective(expr=energy_cost(model) + lam*cvar_term, sense=pyo.minimize)

# ----------------------------
# 5) Solve
# ----------------------------
# Choose a solver installed on your system, e.g., glpk (LP/MIP), highs, cbc, gurobi, cplex
# Example with GLPK:
# solver = pyo.SolverFactory("glpk")
# result = solver.solve(model, tee=True)
# print(result.solver.status, result.solver.termination_condition)

# After solving, retrieve results, e.g.:
# chosen = [k for k in K if pyo.value(model.y[k]) > 0.5][0]
# print("Chosen supplier:", chosen)
# print("Contracted capacity (kW):", pyo.value(model.cap))
# DA = {t: pyo.value(model.q[t]) for t in T}
# Expected cost = pyo.value( energy_cost(model) )

# ----------------------------
# 6) Extensions you can add easily
# ----------------------------
# - Tiered/TOU prices that change by season/weekend: modify p_import/p_export
# - Separate grid fees and REC-internal transfers
# - Demand response variables for flexible loads
# - Binary export enable/disable per contract (use big-M linking)
# - Supplier-specific imbalance models
# - Additional network constraints (transformer limits), prosumer-level detail


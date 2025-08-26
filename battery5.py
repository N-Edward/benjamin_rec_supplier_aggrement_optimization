from pyomo.environ import *

# === Parameters ===
prosumers = ['A','B','C']
T = 24
batteries = {'A':['B1'], 'B':['B2'], 'C':['B3']}  # prosumer-level batteries

# PV & Load
E_pv = {('A',t):5 for t in range(T)}
E_pv.update({('B',t):3 for t in range(T)})
E_pv.update({('C',t):2 for t in range(T)})
E_load = {('A',t):6 for t in range(T)}
E_load.update({('B',t):4 for t in range(T)})
E_load.update({('C',t):3 for t in range(T)})

# Prices
C_buy = {t:0.2 for t in range(T)}
C_sell = {t:0.1 for t in range(T)}

# Battery parameters
E_max = {'B1':30, 'B2':20, 'B3':15}
P_max = {'B1':10, 'B2':5, 'B3':4}
eta_c = {'B1':0.95, 'B2':0.9, 'B3':0.9}
eta_d = {'B1':0.95, 'B2':0.9, 'B3':0.9}

# === Model ===
model = ConcreteModel()
model.T = RangeSet(0,T-1)
model.I = Set(initialize=prosumers)
model.B = Set(initialize=[b for blist in batteries.values() for b in blist])  # all batteries

# Variables
model.P_charge = Var(model.B, model.T, domain=NonNegativeReals, bounds=lambda model,b,t: (0,P_max[b]))
model.P_discharge = Var(model.B, model.T, domain=NonNegativeReals, bounds=lambda model,b,t: (0,P_max[b]))
model.SoC = Var(model.B, model.T, domain=NonNegativeReals, bounds=lambda model,b,t: (0,E_max[b]))
model.P_buy = Var(model.I, model.T, domain=NonNegativeReals)
model.P_sell = Var(model.I, model.T, domain=NonNegativeReals)

# Map battery to prosumer
battery_owner = {}
for i, blist in batteries.items():
    for b in blist:
        battery_owner[b] = i

# === Constraints ===
def soc_constraint(model, b, t):
    if t == 0:
        return model.SoC[b,t] == eta_c[b]*model.P_charge[b,t] - model.P_discharge[b,t]/eta_d[b]
    else:
        return model.SoC[b,t] == model.SoC[b,t-1] + eta_c[b]*model.P_charge[b,t] - model.P_discharge[b,t]/eta_d[b]
model.SOC_Constraint = Constraint(model.B, model.T, rule=soc_constraint)

def energy_balance(model, t):
    total_pv = sum(E_pv[i,t] for i in model.I)
    total_load = sum(E_load[i,t] for i in model.I)
    total_buy = sum(model.P_buy[i,t] for i in model.I)
    total_sell = sum(model.P_sell[i,t] for i in model.I)
    total_discharge = sum(model.P_discharge[b,t] for b in model.B)
    total_charge = sum(model.P_charge[b,t] for b in model.B)
    return total_pv + total_discharge + total_buy == total_load + total_charge + total_sell
model.EnergyBalance = Constraint(model.T, rule=energy_balance)

# === Objective: Total Community Cost ===
def total_cost_rule(model):
    return sum(C_buy[t]*sum(model.P_buy[i,t] for i in model.I) -
               C_sell[t]*sum(model.P_sell[i,t] for i in model.I) for t in model.T)
model.TotalCost = Objective(rule=total_cost_rule, sense=minimize)

# === Solve ===
solver = SolverFactory('glpk')
solver.solve(model)

# === Allocate Costs per Prosumer ===
print("Prosumer Billing with Optimized Battery Usage:")
for i in model.I:
    total_cost = 0
    for t in model.T:
        net_load = E_load[i,t] - E_pv[i,t]
        total_net = sum(E_load[j,t]-E_pv[j,t] for j in model.I)
        if total_net > 0:
            # Include battery usage of prosumer's own batteries
            battery_cost = sum((model.P_charge[b,t]() - model.P_discharge[b,t]())*C_buy[t] 
                               for b in batteries[i])
        else:
            battery_cost = 0
        cost = C_buy[t]*model.P_buy[i,t]() - C_sell[t]*model.P_sell[i,t]() + battery_cost
        total_cost += cost
    print(f"{i}: {round(total_cost,2)} €")


from pyomo.environ import *

# === Parameters ===
prosumers = ['A','B','C']
T = 24

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

# Battery
E_max = 50
P_max = 10
eta_c = 0.95
eta_d = 0.95

# === Model ===
model = ConcreteModel()
model.T = RangeSet(0,T-1)
model.I = Set(initialize=prosumers)

# Variables
model.P_charge = Var(model.T, domain=NonNegativeReals, bounds=(0,P_max))
model.P_discharge = Var(model.T, domain=NonNegativeReals, bounds=(0,P_max))
model.SoC = Var(model.T, domain=NonNegativeReals, bounds=(0,E_max))
model.P_buy = Var(model.I, model.T, domain=NonNegativeReals)
model.P_sell = Var(model.I, model.T, domain=NonNegativeReals)

# === Constraints ===
def soc_constraint(model, t):
    if t==0:
        return model.SoC[t] == eta_c*model.P_charge[t] - model.P_discharge[t]/eta_d
    else:
        return model.SoC[t] == model.SoC[t-1] + eta_c*model.P_charge[t] - model.P_discharge[t]/eta_d
model.SOC_Constraint = Constraint(model.T, rule=soc_constraint)

def energy_balance(model, t):
    total_pv = sum(E_pv[i,t] for i in model.I)
    total_load = sum(E_load[i,t] for i in model.I)
    total_buy = sum(model.P_buy[i,t] for i in model.I)
    total_sell = sum(model.P_sell[i,t] for i in model.I)
    return total_pv + model.P_discharge[t] + total_buy == total_load + model.P_charge[t] + total_sell
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
print("Prosumer Billing:")
for i in model.I:
    total_cost = 0
    for t in model.T:
        # Net load
        net_load = E_load[i,t] - E_pv[i,t]
        # Total net load
        total_net = sum(E_load[j,t]-E_pv[j,t] for j in model.I)
        if total_net > 0:
            # Battery share proportional
            battery_cost = ((model.P_charge[t]() - model.P_discharge[t]())*C_buy[t]) * (net_load/total_net)
        else:
            battery_cost = 0
        # Buy/Sell cost
        cost = C_buy[t]*model.P_buy[i,t]() - C_sell[t]*model.P_sell[i,t]() + battery_cost
        total_cost += cost
    print(f"{i}: {round(total_cost,2)} €")


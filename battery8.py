from pyomo.environ import *

# === Parameters ===
prosumers = ['A','B','C']
T = 24
batteries = {'A':['B1'], 'B':['B2'], 'C':['B3']}
scenarios = ['S1','S2','S3']
prob = {'S1':0.5, 'S2':0.3, 'S3':0.2}  # scenario probabilities

# Forecasted PV & Load per scenario
E_pv = {('A',t,'S1'):5, ('A',t,'S2'):4, ('A',t,'S3'):6 for t in range(T)}
E_pv.update({('B',t,'S1'):3, ('B',t,'S2'):2.5, ('B',t,'S3'):3.5 for t in range(T)})
E_pv.update({('C',t,'S1'):2, ('C',t,'S2'):1.5, ('C',t,'S3'):2.5 for t in range(T)})

E_load = {('A',t,'S1'):6, ('A',t,'S2'):5.5, ('A',t,'S3'):6.5 for t in range(T)}
E_load.update({('B',t,'S1'):4, ('B',t,'S2'):3.5, ('B',t,'S3'):4.5 for t in range(T)})
E_load.update({('C',t,'S1'):3, ('C',t,'S2'):2.5, ('C',t,'S3'):3.5 for t in range(T)})

# Contracted energy per prosumer
E_contract = {('A',t):5 for t in range(T)}
E_contract.update({('B',t):4 for t in range(T)})
E_contract.update({('C',t):3 for t in range(T)})

# TOU Prices
C_buy = {t:0.1 if 0<=t<8 else 0.25 if 8<=t<18 else 0.2 for t in range(T)}
C_sell = {t:0.08 if 0<=t<8 else 0.15 if 8<=t<18 else 0.12 for t in range(T)}
C_penalty = {t:0.3 for t in range(T)}  # €/kWh deviation

# Battery parameters
E_max = {'B1':30, 'B2':20, 'B3':15}
P_max = {'B1':10, 'B2':5, 'B3':4}
eta_c = {'B1':0.95, 'B2':0.9, 'B3':0.9}
eta_d = {'B1':0.95, 'B2':0.9, 'B3':0.9}
c_deg = {'B1':0.01, 'B2':0.015, 'B3':0.02}

# === Model ===
model = ConcreteModel()
model.T = RangeSet(0,T-1)
model.I = Set(initialize=prosumers)
model.B = Set(initialize=[b for blist in batteries.values() for b in blist])
model.S = Set(initialize=scenarios)

# Variables: scenario-dependent (wait-and-see)
model.P_charge = Var(model.B, model.T, model.S, domain=NonNegativeReals, bounds=lambda m,b,t,s:(0,P_max[b]))
model.P_discharge = Var(model.B, model.T, model.S, domain=NonNegativeReals, bounds=lambda m,b,t,s:(0,P_max[b]))
model.SoC = Var(model.B, model.T, model.S, domain=NonNegativeReals, bounds=lambda m,b,t,s:(0,E_max[b]))
model.P_buy = Var(model.I, model.T, model.S, domain=NonNegativeReals)
model.P_sell = Var(model.I, model.T, model.S, domain=NonNegativeReals)
model.penalty = Var(model.I, model.T, model.S, domain=NonNegativeReals)

# Map battery to prosumer
battery_owner = {}
for i, blist in batteries.items():
    for b in blist:
        battery_owner[b] = i

# === Constraints ===
def soc_constraint(model, b, t, s):
    if t==0:
        return model.SoC[b,t,s] == eta_c[b]*model.P_charge[b,t,s] - model.P_discharge[b,t,s]/eta_d[b]
    else:
        return model.SoC[b,t,s] == model.SoC[b,t-1,s] + eta_c[b]*model.P_charge[b,t,s] - model.P_discharge[b,t,s]/eta_d[b]
model.SOC_Constraint = Constraint(model.B, model.T, model.S, rule=soc_constraint)

def energy_balance(model, t, s):
    total_pv = sum(E_pv[i,t,s] for i in model.I)
    total_load = sum(E_load[i,t,s] for i in model.I)
    total_buy = sum(model.P_buy[i,t,s] for i in model.I)
    total_sell = sum(model.P_sell[i,t,s] for i in model.I)
    total_discharge = sum(model.P_discharge[b,t,s] for b in model.B)
    total_charge = sum(model.P_charge[b,t,s] for b in model.B)
    return total_pv + total_discharge + total_buy == total_load + total_charge + total_sell
model.EnergyBalance = Constraint(model.T, model.S, rule=energy_balance)

# Penalty calculation
def penalty_rule(model, i, t, s):
    return model.penalty[i,t,s] >= C_penalty[t] * (model.P_buy[i,t,s] - E_contract[i,t])
model.PenaltyConstraint = Constraint(model.I, model.T, model.S, rule=penalty_rule)

# === Objective: Expected total cost ===
def expected_cost(model):
    return sum(prob[s] * (
        sum(C_buy[t]*sum(model.P_buy[i,t,s] for i in model.I)
            - C_sell[t]*sum(model.P_sell[i,t,s] for i in model.I)
            + sum(c_deg[b]*(model.P_charge[b,t,s]+model.P_discharge[b,t,s]) for b in model.B)
            + sum(model.penalty[i,t,s] for i in model.I)
        for t in model.T)
    ) for s in model.S)
model.ExpectedCost = Objective(rule=expected_cost, sense=minimize)

# === Solve ===
solver = SolverFactory('glpk')
solver.solve(model)

# === Prosumer Billing: Expected Values ===
print("Prosumer Billing under Stochastic Scenarios:")
for i in model.I:
    expected_total = 0
    for s in model.S:
        prob_s = prob[s]
        total_s = 0
        for t in model.T:
            battery_cost = sum(c_deg[b]*(model.P_charge[b,t,s]() + model.P_discharge[b,t,s]()) for b in batteries[i])
            cost = C_buy[t]*model.P_buy[i,t,s]() - C_sell[t]*model.P_sell[i,t,s]() + battery_cost + model.penalty[i,t,s]()
            total_s += cost
        expected_total += prob_s * total_s
    print(f"{i}: {round(expected_total,2)} €")


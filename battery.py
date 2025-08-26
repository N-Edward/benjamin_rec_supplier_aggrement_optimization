from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, SolverFactory, RangeSet

# Time horizon
T = 24

# Parameters (example)
E_load = [10]*T       # kWh
E_pv = [5]*T          # kWh
C_buy = [0.2]*T       # €/kWh
C_sell = [0.1]*T      # €/kWh
E_max = 50
P_max = 10
eta_c = 0.95
eta_d = 0.95

model = ConcreteModel()

# Sets
model.T = RangeSet(0, T-1)

# Variables
model.P_charge = Var(model.T, domain=NonNegativeReals, bounds=(0,P_max))
model.P_discharge = Var(model.T, domain=NonNegativeReals, bounds=(0,P_max))
model.SoC = Var(model.T, domain=NonNegativeReals, bounds=(0,E_max))
model.P_buy = Var(model.T, domain=NonNegativeReals)
model.P_sell = Var(model.T, domain=NonNegativeReals)

# Constraints
def soc_constraint(model, t):
    if t == 0:
        return model.SoC[t] == 0 + eta_c*model.P_charge[t] - model.P_discharge[t]/eta_d
    else:
        return model.SoC[t] == model.SoC[t-1] + eta_c*model.P_charge[t] - model.P_discharge[t]/eta_d
model.SOC_Constraint = Constraint(model.T, rule=soc_constraint)

def energy_balance(model, t):
    return E_pv[t] + model.P_discharge[t] + model.P_buy[t] == E_load[t] + model.P_charge[t] + model.P_sell[t]
model.EnergyBalance = Constraint(model.T, rule=energy_balance)

# Objective
def cost_rule(model):
    return sum(C_buy[t]*model.P_buy[t] - C_sell[t]*model.P_sell[t] for t in model.T)
model.Cost = Objective(rule=cost_rule, sense=1)  # sense=1 -> minimize

# Solve
solver = SolverFactory('glpk')
solver.solve(model)

# Display results
for t in model.T:
    print(t, model.P_charge[t](), model.P_discharge[t](), model.SoC[t](), model.P_buy[t](), model.P_sell[t]())

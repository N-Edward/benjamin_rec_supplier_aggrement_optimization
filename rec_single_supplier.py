from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary, Objective, Constraint, SolverFactory, value, Reals, summation
)
from pyomo.environ import minimize

# -----------------------
# Toggle: choose supplier or use fixed one
SUPPLIER_SELECTION = False  # set to False if supplier is predetermined
FIXED_SUPPLIER = 'supA'    # only used if SUPPLIER_SELECTION = False
# -----------------------

# ---------- Toy data ----------
T = list(range(1, 5))  # 4 time steps
I = ['m1', 'm2']
S = ['supA', 'supB']

# Member loads (kWh)
L = {
    ('m1', 1): 4, ('m1', 2): 5, ('m1', 3): 6, ('m1', 4): 4,
    ('m2', 1): 3, ('m2', 2): 3, ('m2', 3): 2, ('m2', 4): 3,
}

# Member generation (kWh)
G = {
    ('m1', 1): 2, ('m1', 2): 3, ('m1', 3): 5, ('m1', 4): 1,
    ('m2', 1): 1, ('m2', 2): 2, ('m2', 3): 1, ('m2', 4): 0,
}

# Prices per supplier (€/kWh), Time-of-Use
p_buy = {
    ('supA', 1): 0.22, ('supA', 2): 0.22, ('supA', 3): 0.30, ('supA', 4): 0.18,
    ('supB', 1): 0.20, ('supB', 2): 0.24, ('supB', 3): 0.28, ('supB', 4): 0.20,
}
p_sell = {
    ('supA', 1): 0.06, ('supA', 2): 0.05, ('supA', 3): 0.04, ('supA', 4): 0.05,
    ('supB', 1): 0.07, ('supB', 2): 0.06, ('supB', 3): 0.05, ('supB', 4): 0.05,
}

# Fixed fees per horizon (e.g., monthly) — optional
F = {'supA': 20.0, 'supB': 18.0}

# Internal-priority parameter (alpha=1 means allocate all internal generation first)
alpha = 1.0

# ---------- Model ----------
m = ConcreteModel()

m.T = Set(initialize=T, ordered=True)
m.I = Set(initialize=I)
m.S = Set(initialize=S)

m.L = Param(m.I, m.T, initialize=L, within=NonNegativeReals)
m.G = Param(m.I, m.T, initialize=G, within=NonNegativeReals)
m.p_buy = Param(m.S, m.T, initialize=p_buy, within=NonNegativeReals)
m.p_sell = Param(m.S, m.T, initialize=p_sell, within=NonNegativeReals)
m.F = Param(m.S, initialize=F, within=NonNegativeReals)
m.alpha = Param(initialize=alpha)

# Allocation from community pool to each member
m.a = Var(m.I, m.T, within=NonNegativeReals)

# Residual import/export with the single chosen supplier
m.q = Var(m.T, within=NonNegativeReals)  # import
m.e = Var(m.T, within=NonNegativeReals)  # export

if SUPPLIER_SELECTION:
    m.y = Var(m.S, within=Binary)

    # exactly one supplier
    def one_supplier_rule(m):
        return sum(m.y[s] for s in m.S) == 1
    m.one_supplier = Constraint(rule=one_supplier_rule)

    # Effective prices via convex combination of exactly one supplier:
    # q_cost_t = sum_s y_s * p_buy[s,t] * q_t  ; similarly for e_t
    # This is linear because q_t is common across s and y_s sum to 1
    # (No need for big-M in this specific structure)
else:
    chosen = FIXED_SUPPLIER
    # For fixed supplier, we can emulate selection with a constant param
    m.y = Param(m.S, initialize={s: 1 if s == chosen else 0 for s in S}, within=NonNegativeReals, mutable=False)

# Community pool cannot allocate more than total generation
def pool_limit_rule(m, t):
    return sum(m.a[i, t] for i in m.I) <= sum(m.G[i, t] for i in m.I)
m.pool_limit = Constraint(m.T, rule=pool_limit_rule)

# Optional: prioritize internal allocation
def internal_priority_rule(m, t):
    return sum(m.a[i, t] for i in m.I) >= m.alpha * sum(m.G[i, t] for i in m.I)
m.internal_priority = Constraint(m.T, rule=internal_priority_rule)

# Member cannot receive more allocation than their load
def member_cap_rule(m, i, t):
    return m.a[i, t] <= m.L[i, t]
m.member_cap = Constraint(m.I, m.T, rule=member_cap_rule)

# System net balance: q - e = total_load - total_gen
def net_balance_rule(m, t):
    total_load = sum(m.L[i, t] for i in m.I)
    total_gen = sum(m.G[i, t] for i in m.I)
    return m.q[t] - m.e[t] == total_load - total_gen
m.net_balance = Constraint(m.T, rule=net_balance_rule)

# Objective: energy cost + fixed fee of chosen supplier
def obj_rule(m):
    energy_term = sum(
        sum(m.y[s] * m.p_buy[s, t] for s in m.S) * m.q[t]
        - sum(m.y[s] * m.p_sell[s, t] for s in m.S) * m.e[t]
        for t in m.T
    )
    fee_term = sum(m.y[s] * m.F[s] for s in m.S)
    return energy_term + fee_term

m.OBJ = Objective(rule=obj_rule, sense=minimize)

# ---------- Solve ----------
# You can use any LP/MIP solver available (cbc, glpk, highs, gurobi, cplex)
solver = SolverFactory('gurobi')  # change if you have a different solver
results = solver.solve(m, tee=False)

# ---------- Report ----------
total_cost = value(m.OBJ)
supplier_choice = None
if SUPPLIER_SELECTION:
    supplier_choice = [s for s in m.S if value(m.y[s]) > 0.5][0]
else:
    supplier_choice = FIXED_SUPPLIER

print(f"Chosen supplier: {supplier_choice}")
print(f"Total cost: €{total_cost:.2f}")
print("time  import(q)  export(e)  pool_alloc  load  gen")
for t in m.T:
    q = value(m.q[t]); e = value(m.e[t])
    pool = sum(value(m.a[i, t]) for i in m.I)
    load = sum(value(m.L[i, t]) for i in m.I)
    gen = sum(value(m.G[i, t]) for i in m.I)
    print(f"{t:>4}  {q:10.3f}  {e:10.3f}  {pool:10.3f}  {load:5.3f}  {gen:5.3f}")


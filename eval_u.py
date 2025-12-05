# Constant drug dose
def constant_input(u_val):
    return lambda t: u_val

# Actual drug input: dose at fixed intervals  
def actual_drug_input(dose=200.0, interval=21.0):
    def r(t):
        return dose if (t % interval == 0.0) else 0.0
    return r
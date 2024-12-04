from z3 import *
import numpy as np

def maxsat(customers_R, customers_A, depot, dmax, delta):
    """
    Optimizes a route given initial sub-tour R and additional customers A.
    """
    R = len(customers_R)  # Number of customers in R
    A = len(customers_A)  # Number of additional customers in A
    N = R + A  # Total number of customers
    
    # Combine R and A into a single list for easier indexing
    customers = customers_R + customers_A
    
    # Variables
    O = [Int(f"O_{i}") for i in range(N)]  # Ordering variables
    D = [Real(f"D_{i}") for i in range(N)]  # Distances
    D_return = Real("D_return")            # Return distance
    
    s = Solver()  # Create a Z3 solver instance
    
    # Constraints for customers in R and A
    for i in range(N):
        # Ordering: O_i is either skipped (-1) or in [1, R + A]
        s.add(Or(O[i] == -1, And(O[i] >= 1, O[i] <= N)))
        
        # Distance penalties
        s.add(If(O[i] == -1, D[i] == dmax, D[i] >= 0))
    
    # Successive customer constraints
    for i in range(N):
        for j in range(N):
            if i != j:
                # If O_j + 1 == O_i, then D[i] is the distance between customers
                s.add(If(And(O[i] > 1, O[i] == O[j] + 1), 
                         D[i] == np.linalg.norm(customers[i] - customers[j]),
                         True))  # No constraint otherwise
    
    # First customer constraint: Distance from current location
    for i in range(N):
        s.add(If(O[i] == 1, D[i] == np.linalg.norm(customers[i] - depot), True))
    
    # Return distance
    for i in range(N):
        s.add(If(O[i] == N, D_return == np.linalg.norm(customers[i] - depot), True))
    
    
    # Constraints specific to R
    for i in range(R):
        # CPS constraint: Constrained position shifting (Constraint 11)
        s.add(And(O[i] >= i - delta, O[i] <= i + delta))
        
        # Force all customers in R to be served (Constraint 12)
        s.add(O[i] >= 1)

    # all ordering indeces must be unique (except for -1)
    s.add(Distinct([If(O[i] >= 1, O[i], 0) for i in range(N)]))
    
    # Objective: Minimize total distance
    total_distance = Sum(D) + D_return
    s.add(D_return >= 0)  # Ensure valid return distance
    opt = Optimize()
    opt.add(s.assertions())
    opt.minimize(total_distance)
    
    # Solve the problem
    if opt.check() == sat:
        m = opt.model()
        ordering = [m.evaluate(O[i]) for i in range(N)]
        distances = [m.evaluate(D[i]).as_decimal(2) for i in range(N)]
        return ordering, distances, m.evaluate(D_return).as_decimal(2)
    else:
        return None

# Example usage
# customers_R = [np.array([2, 3]), np.array([8, 6])]  # Example coordinates for customers in R
# customers_A = [np.array([5, 7])]          # Example coordinates for additional customers in A
# depot = np.array([0, 0])
# current_loc = (1, 1)
# dmax = 100
# delta = 1

# result = optimize_route(customers_R, customers_A, depot, dmax, delta)
# if result:
#     ordering, distances, d_return = result
#     print("Ordering:", ordering)
#     print("Distances:", distances)
#     print("Return Distance:", d_return)
# else:
#     print("No solution found.")

# 2,3 8,6 = 6.708  1 2
# 8,6 5,7 = 3.162  2 3
# 2,3 5,7 = 5      1 3

# 1 2 3 9.86
# 1 3 2 8.16
# 2 1 3 11

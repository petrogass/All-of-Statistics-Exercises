###Chapter1###

import numpy as np
import random
import matplotlib.pyplot as plt

#Exercise 1
def simulate_coins(p, n):
    flips = np.random.rand(n) < p 
    proportion = [sum(flips[:i+1])/float(i+1) for i in range(n)]  
    plt.figure(figsize=(10,6))
    plt.plot(range(1, n+1), proportion, label=f'p = {p}')    
    plt.xlabel("Number of Flips")
    plt.ylabel("Proportion of heads")
    plt.title("Coin Toss Simulation")
    plt.legend()
    plt.grid(True)
    plt.show()
p_values = [0.3, 0.03]
n = 2000
for p in p_values:
    simulate_coins(p, n)
    
#Exercise 2
def simulate_coins_2(p, n, trials):
    total = 0
    for _ in range(trials):
        heads=np.random.rand(n) < p
        total+=sum(heads)
    average = total/trials
    return average
    
p = 0.3
n_values = [10, 100, 1000]
trials = 1000

for n in n_values:
    expected = n*p
    average = simulate_coins_2(p, n, trials)
    print(f"For n = {n}, Expected number of heads (np) = {expected}, Average number of heads = {average}")
    
#Exercise 3
def ex_3():
    sample_space = [1,2,3,4,5,6]
    event_A = [2,4,6]
    event_B = [1,2,3,4]
    
    trials = 10000
    count_A = 0
    count_B = 0
    count_AB = 0
    
    for _ in range(trials):
        outcome = random.choice(sample_space)
        if outcome in event_A:
            count_A += 1
        if outcome in event_B:
            count_B += 1
        if outcome in event_A and outcome in event_B:
            count_AB += 1
    
    P_A = count_A / trials
    P_B = count_B / trials
    P_AB = count_AB / trials
    
    print("Theoretical value of P(AB):", 1/3)
    print("Simulated value of P(AB):", P_AB)
    print("Simulated value of P(A) * P(B):", P_A * P_B)
    
    C = [1, 3, 5] 
    D = [2, 4, 6]
    count_C = 0
    count_D = 0
    count_CD = 0
    
    for _ in range(trials):
        outcome = random.choice(sample_space)
        if outcome in C:
            count_C += 1
        if outcome in D:
            count_D += 1
        if outcome in C and outcome in D:
            count_CD += 1

    
    P_C = count_C / trials
    P_D = count_D / trials
    P_CD = count_CD / trials

    
    print("\nTheoretical value of P(C):", 1/2)
    print("Simulated value of P(C):", P_C)
    print("\nTheoretical value of P(D):", 1/2)
    print("Simulated value of P(D):", P_D)
    print("\nTheoretical value of P(CD):", 0)
    print("Simulated value of P(CD):", P_CD)
    print("Simulated value of P(C) * P(D):", P_C * P_D)

ex_3()




























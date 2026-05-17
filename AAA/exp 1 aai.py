# =========================================================
# Hidden Markov Model (HMM) for Weather Prediction
# =========================================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# Define States and Observations
# =========================================================

states = ["Sunny", "Rainy", "Cloudy"]
observations = ["Happy", "Sad"]

# Initial Probabilities
start_prob = np.array([0.4, 0.3, 0.3])

# Transition Probability Matrix
transition_prob = np.array([
    [0.6, 0.2, 0.2],
    [0.3, 0.5, 0.2],
    [0.4, 0.3, 0.3]
])

# Emission Probability Matrix
emission_prob = np.array([
    [0.2, 0.8],
    [0.9, 0.1],
    [0.6, 0.4]
])

# Observation Sequence
# Happy = 0, Sad = 1
obs_sequence = [0, 0, 1]


def forward_algorithm(obs_seq):

    N = len(states)       # Number of states
    T = len(obs_seq)      # Length of observation sequence

    alpha = np.zeros((N, T))

    # Initialization
    for i in range(N):
        alpha[i, 0] = start_prob[i] * emission_prob[i, obs_seq[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = np.sum(
                alpha[:, t-1] * transition_prob[:, j]
            ) * emission_prob[j, obs_seq[t]]

    # Termination
    probability = np.sum(alpha[:, T-1])

    return alpha, probability


# Run Forward Algorithm
alpha_matrix, sequence_probability = forward_algorithm(obs_sequence)

print("Forward Probability Matrix (alpha):")
print(alpha_matrix)

print("\nProbability of the observation sequence:")
print(sequence_probability)


def viterbi(obs_seq):

    n_states = len(states)
    T = len(obs_seq)

    dp = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=int)

    # Initialization
    dp[:, 0] = start_prob * emission_prob[:, obs_seq[0]]

    # Recursion
    for t in range(1, T):
        for s in range(n_states):

            prob = dp[:, t-1] * transition_prob[:, s] * emission_prob[s, obs_seq[t]]

            dp[s, t] = np.max(prob)
            backpointer[s, t] = np.argmax(prob)

    # Backtracking
    best_path = [np.argmax(dp[:, T-1])]

    for t in range(T-1, 0, -1):
        best_path.insert(0, backpointer[best_path[0], t])

    return [states[i] for i in best_path]


# Predict Hidden States
predicted_states = viterbi(obs_sequence)

print("\nPredicted Hidden States:")
print(predicted_states)

actual_states = ["Rainy", "Sunny", "Sunny"]

# Convert States to Numeric
state_to_index = {state: idx for idx, state in enumerate(states)}

actual_numeric = [state_to_index[s] for s in actual_states]
predicted_numeric = [state_to_index[s] for s in predicted_states]

# Accuracy
accuracy = accuracy_score(actual_numeric, predicted_numeric)

print("\nAccuracy Score:")
print(accuracy)

cm = confusion_matrix(actual_numeric, predicted_numeric)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=states,
    yticklabels=states
)

plt.xlabel("Predicted States")
plt.ylabel("Actual States")
plt.title("Confusion Matrix for Weather Prediction")

plt.show()


plt.figure(figsize=(8, 4))

plt.plot(actual_numeric, marker='o', label="Actual States")
plt.plot(predicted_numeric, marker='s', label="Predicted States")

plt.yticks(range(len(states)), states)

plt.xlabel("Time Step")
plt.ylabel("Weather State")
plt.title("Time Series Plot: Actual vs Predicted Weather States")

plt.legend()
plt.grid(True)

plt.show()
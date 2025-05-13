import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

import matplotlib.pyplot as plt
import networkx as nx

def plot_bn_structure(model):
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)  # layout with consistent positioning
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', arrowsize=20, font_size=10)
    plt.title("Bayesian Network Structure")
    plt.show()

def plot_cpt(model, var, given):
    cpd = model.get_cpds(var)
    if cpd is None:
        print(f"No CPD found for {var}")
        return

    # Find the row indexes where parent is fixed to a specific value
    parent_values = list(cpd.get_evidence())
    if given not in parent_values:
        print(f"{given} is not a parent of {var}")
        return

    idx = parent_values.index(given)
    parent_card = cpd.cardinality[idx]
    labels = cpd.state_names[given]

    # Outcome values
    outcome_labels = cpd.state_names[var]

    # Collect P(Outcome | Glucose=each_value)
    probs = {label: [] for label in labels}
    for i, state in enumerate(labels):
        for j, outcome in enumerate(outcome_labels):
            row = i * len(outcome_labels) + j
            probs[state].append(cpd.values.flatten()[row])

    # Plot
    fig, ax = plt.subplots()
    x = range(len(outcome_labels))
    for label, prob in probs.items():
        ax.bar(x, prob, label=f'{given}={label}', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(outcome_labels)
    ax.set_ylabel(f'P({var})')
    ax.set_title(f'P({var} | {given})')
    ax.legend()
    plt.show()

def plot_inference_result(query_result, title):
    labels = query_result.state_names['Outcome']
    values = query_result.values

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color='skyblue')
    plt.ylabel("Probability")
    plt.title(title)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.ylim(0, 1)
    plt.show()

# Step 1: Load and preprocess dataset
df = pd.read_csv("diabetes.csv")

# Replace invalid zero values with NaN
cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)
df.dropna(inplace=True)

# Discretization: convert continuous features into categories
df['Pregnancies'] = pd.cut(df['Pregnancies'], bins=[-1, 2, 6, 20], labels=['low', 'medium', 'high'])
df['Glucose'] = pd.cut(df['Glucose'], bins=[0, 110, 140, 200], labels=['low', 'medium', 'high'])
df['BloodPressure'] = pd.cut(df['BloodPressure'], bins=[0, 70, 90, 122], labels=['low', 'medium', 'high'])
df['SkinThickness'] = pd.cut(df['SkinThickness'], bins=[0, 20, 35, 100], labels=['low', 'medium', 'high'])
df['Insulin'] = pd.cut(df['Insulin'], bins=[0, 100, 200, 850], labels=['low', 'medium', 'high'])
df['BMI'] = pd.cut(df['BMI'], bins=[0, 25, 35, 70], labels=['low', 'medium', 'high'])
df['DiabetesPedigreeFunction'] = pd.cut(df['DiabetesPedigreeFunction'], bins=[0, 0.4, 0.8, 2.5], labels=['low', 'medium', 'high'])
df['Age'] = pd.cut(df['Age'], bins=[20, 35, 50, 100], labels=['young', 'middle', 'old'])

# Convert Outcome to string (pgmpy requires categorical vars)
df['Outcome'] = df['Outcome'].astype(str)

# Keep only discretized columns
df_cleaned = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]

# Convert all to string for categorical handling
df_cleaned = df_cleaned.astype(str)

# Step 2: Split into train/test
train_data, test_data = train_test_split(df_cleaned, test_size=0.2, random_state=42)

# Step 2: Force all 8 features -> Outcome (Naive Bayes structure)
features = list(train_data.columns)
features.remove("Outcome")
edges = [(feature, "Outcome") for feature in features]

model = BayesianNetwork(edges)
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Step 5: Inference
infer = VariableElimination(model)

plot_bn_structure(model)
plot_cpt(model, var="Outcome", given="Glucose")


print("\nSample inference: P(Outcome | Glucose=high, Age=old)")
q = infer.query(variables=["Outcome"], evidence={"Glucose": "high", "Age": "old"})
plot_inference_result(q, "P(Outcome | Glucose=high, Age=old)")
print(q)

y_true = []
y_pred = []
skipped = 0

for i, row in test_data.iterrows():
    evidence = row.drop("Outcome").to_dict()
    try:
        q = infer.query(variables=["Outcome"], evidence=evidence)
        pred = str(q.values.argmax())  # '0' or '1'
        y_pred.append(pred)
        y_true.append(row["Outcome"])
    except Exception as e:
        print(f"Skipping sample {i} due to error: {e}")
        skipped += 1
        continue

print(f"\nSkipped {skipped} samples due to unsupported evidence.")

# Final check to avoid computing accuracy on empty lists
if len(y_pred) > 0:
    acc = accuracy_score(y_true, y_pred)
    print(f"Prediction accuracy on test set: {acc:.4f}")
else:
    print("No valid predictions were made. Consider simplifying the model or reducing evidence.")

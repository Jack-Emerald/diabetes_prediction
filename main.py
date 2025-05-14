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


def plot_feature_distribution(df, feature, original_series = None, method = 'qcut'):
    """
    Plot the count of patients in each category of a given discretized feature,
    with bin ranges shown in x-axis labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing the discretized feature.
        feature (str): Column name of the feature.
        original_series (pd.Series): Original continuous series before discretization.
        method (str): 'qcut' or 'cut' to indicate how the discretization was done.
    """
    counts = df[feature].value_counts().sort_index()
    labels = counts.index.tolist()

    # Prepare bin edge labels if original data is provided
    if original_series is not None:
        if method == 'qcut':
            bins = \
            pd.qcut(original_series.astype(float), q = 3, retbins = True, duplicates = 'drop')[1]
        elif method == 'cut':
            bins = pd.cut(original_series.astype(float), bins = 3, retbins = True)[1]
        else:
            raise ValueError("Method must be 'qcut' or 'cut'")

        # Format labels with bin ranges
        bin_labels = [
            f"{label}\n[{bins[i]:.1f}, {bins[i + 1]:.1f})"
            for i, label in enumerate(labels)
        ]
    else:
        bin_labels = labels

    # Plot manually with formatted x-axis
    plt.figure(figsize = (8, 5))
    x = range(len(counts))
    plt.bar(x, counts.values, color = 'lightgreen', edgecolor = 'black')
    plt.xticks(ticks = x, labels = bin_labels, rotation = 0)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(f"{feature} Category")
    plt.ylabel("Number of Patients")
    for i, v in enumerate(counts):
        plt.text(i, v + 2, str(v), ha = 'center', fontweight = 'bold')
    plt.tight_layout()
    plt.show()


# Step 1: Load and preprocess dataset
df = pd.read_csv("diabetes.csv")

# Replace invalid zero values with NaN
cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)
df.dropna(inplace=True)

# Discretization: convert continuous features into categories
df['Pregnancies'] = pd.qcut(df['Pregnancies'].astype(float),  q=3,  labels=['low', 'medium', 'high'])
df['Glucose'] = pd.qcut(df['Glucose'].astype(float),  q=3,  labels=['low', 'medium', 'high'])
df['BloodPressure'] = pd.qcut(df['BloodPressure'].astype(float),  q=3,  labels=['low', 'medium', 'high'])
df['SkinThickness'] = pd.qcut(df['SkinThickness'].astype(float),  q=3,  labels=['low', 'medium', 'high'])
df['Insulin'] = pd.qcut(df['Insulin'].astype(float),  q=3,  labels=['low', 'medium', 'high'])
df['BMI'] = pd.qcut(df['BMI'].astype(float), q=3, labels=['low', 'medium', 'high'])
df['DiabetesPedigreeFunction'] = pd.qcut(df['DiabetesPedigreeFunction'].astype(float),  q=3,  labels=['low', 'medium', 'high'])
df['Age'] = pd.qcut(df['Age'].astype(float),  q=3,  labels=['young', 'middle', 'old'])

# Convert Outcome to string (pgmpy requires categorical vars)
df['Outcome'] = df['Outcome'].astype(str)

# Keep only discretized columns
df_cleaned = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]

# Convert all to string for categorical handling
df_cleaned = df_cleaned.astype(str)

plot_feature_distribution(df_cleaned, 'Glucose')
#plot_feature_distribution(df_cleaned, 'BMI')
#plot_feature_distribution(df_cleaned, 'Age')

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



# Define comprehensive test cases
test_cases = [
    {"evidence": {"Glucose": "low"}, "label": "Glucose=low"},
    {"evidence": {"Glucose": "high"}, "label": "Glucose=high"},
    {"evidence": {"Glucose": "high", "BMI": "high"}, "label": "Glucose=high, BMI=high"},
    {"evidence": {"Glucose": "high", "BMI": "high", "Age": "old"}, "label": "Glucose=high, BMI=high, Age=old"},
    {"evidence": {"Glucose": "high", "BMI": "high", "Age": "old", "Insulin": "high"}, "label": "Glucose=high, BMI=high, Age=old, Insulin=high"},
    {"evidence": {"Glucose": "high", "BMI": "high", "Age": "old", "Insulin": "high", "DiabetesPedigreeFunction": "high"}, "label": "Glucose=high, BMI=high, Age=old, Insulin=high, DiabetesPedigreeFunction=high"},
]


# Collect and plot inference results
labels = []
probs = []

for case in test_cases:
    try:
        q = infer.query(variables=["Outcome"], evidence=case["evidence"])
        labels.append(case["label"])
        probs.append(q.values[1])  # Probability of Outcome = 1
    except Exception as e:
        print(f"Skipped {case['label']} due to error: {e}")

# Plot
plt.figure(figsize=(10, 6))
plt.barh(labels, probs, color='skyblue')
plt.xlabel("P(Outcome = 1)")
plt.title("Posterior Probabilities for Different Evidence Scenarios")
for i, v in enumerate(probs):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center')
plt.xlim(0.3, 0.7)
plt.tight_layout()
plt.show()






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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import friedmanchisquare, rankdata
import matplotlib.pyplot as plt

# 1) Carrega o CSV e descarta a primeira coluna
df = pd.read_csv('D:\\Collective_cog_prediction_MLP\\data\\pre_processado\\cog_coletivo_full_data.csv')
df = df.iloc[:, 1:]

# 2) Separa X e y (supondo que o alvo seja 'resultado')
y = df['resultado'].values
X = df.drop(columns=['resultado', 'PCT_1', 'PCT_2'], errors='ignore')
colunas_originais = X.columns.tolist()

# 3) Transformer para recalcular PCT1 e PCT2 após cada normalização de fold
def add_pct(X_arr):
    X_df = pd.DataFrame(X_arr, columns=colunas_originais)
    X_df['PCT1'] = X_df.filter(regex='_1$').mean(axis=1)
    X_df['PCT2'] = X_df.filter(regex='_2$').mean(axis=1)
    return X_df.values

pct_transformer = FunctionTransformer(add_pct, validate=False)

# 4) Grade de hiperparâmetros
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'alpha': [1e-4, 1e-3],
    'learning_rate_init': [1e-3, 1e-2]
}
configs = [
    {'hidden_layer_sizes': hl, 'alpha': a, 'learning_rate_init': lr}
    for hl in param_grid['hidden_layer_sizes']
    for a in param_grid['alpha']
    for lr in param_grid['learning_rate_init']
]

# 5) Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
long_names = [
    f"HL{cfg['hidden_layer_sizes']}_A{cfg['alpha']}_L{cfg['learning_rate_init']}"
    for cfg in configs
]

# 6) Abreviações para os gráficos
short_names = []
for name in long_names:
    s = (name
         .replace("HL(50,)","50")
         .replace("HL(100,)","100")
         .replace("HL(50, 50)","50-50")
         .replace("_A"," A")
         .replace("_L"," L"))
    short_names.append(s)

# 7) DataFrame para resultados
results = pd.DataFrame(index=range(1, 6), columns=long_names)

# 8) Loop de avaliação sem vazamento
for cfg, long_name in zip(configs, long_names):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pct',    pct_transformer),
        ('mlp',    MLPClassifier(random_state=42, max_iter=500, **cfg))
    ])
    scores = cross_val_score(pipe, X.values, y, cv=cv, scoring='accuracy')
    results[long_name] = scores

# 9) Friedman test
stat, p_value = friedmanchisquare(*[results[col] for col in results.columns])
print(f"Friedman χ² = {stat:.3f}, p-value = {p_value:.3f}")

# 10) Nemenyi post-hoc (aprox.)
def nemenyi_posthoc(data, q_crit=3.0):
    n_folds, k = data.shape
    ranks = np.apply_along_axis(lambda row: rankdata(-row), 1, data)
    avg_ranks = ranks.mean(axis=0)
    se = np.sqrt(k * (k + 1) / (6 * n_folds))
    cd = q_crit * se
    return avg_ranks, cd

avg_ranks, cd = nemenyi_posthoc(results.values)

# 11) Gráficos
# Usa cópia para plot com nomes curtos
plot_df = results.copy()
plot_df.columns = short_names

plt.figure(figsize=(10, 6))
plot_df.boxplot()
plt.title('MLP: 5-fold CV (Accuracy)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.figure(figsize=(10, 6))
y_pos = np.arange(len(avg_ranks))
plt.hlines(y_pos, [0], avg_ranks)
plt.plot(avg_ranks, y_pos, 'o')
plt.yticks(y_pos, short_names)
plt.xlabel('Average Rank (lower = better)')
plt.title('Critical Difference Diagram (approx.)')
max_r = avg_ranks.max()
plt.hlines(-1, max_r - cd, max_r, linewidth=4)
plt.text(max_r - cd/2, -1.3, f'CD = {cd:.2f}', ha='center')
plt.ylim(-2, len(avg_ranks))
plt.tight_layout()

# 12) Matriz de Confusão para a melhor configuração
best_name = results.mean().idxmax()
best_cfg = configs[long_names.index(best_name)]

best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pct',    pct_transformer),
    ('mlp',    MLPClassifier(random_state=42, max_iter=500, **best_cfg))
])
y_pred = cross_val_predict(best_pipe, X.values, y, cv=cv)
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix ({best_name}) – Acc: {acc:.3f}')
plt.colorbar()
ticks = np.unique(y)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.tight_layout()

plt.show()

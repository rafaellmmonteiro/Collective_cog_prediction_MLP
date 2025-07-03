import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical as GeneticCategorical, Continuous, Integer as GeneticInteger
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from sklearn.metrics import confusion_matrix, accuracy_score

# ─── 0) Wrapper que converte strings em tuplas ────────────────────────────────
class MLPWrapper(MLPClassifier):
    def set_params(self, **params):
        # Se vier string para hidden_layer_sizes, avalia para tupla
        hls_key = 'hidden_layer_sizes'
        if hls_key in params and isinstance(params[hls_key], str):
            params[hls_key] = eval(params[hls_key])
        return super().set_params(**params)

# ─── 1) Processamento dos dados ─────────────────────────────────────────────
df = pd.read_csv(
    '/home/rafael-savino/Documents/Mestrado_Bioinfo/Disciplinas/Comp_Bioinsp/projeto_2_MLP/cog_coletivo_full_data.csv'
).iloc[:, 1:]
y = df['resultado'].values
X = df.drop(columns=['resultado', 'PCT_1', 'PCT_2'], errors='ignore')
feature_cols = X.columns.tolist()

# ─── 2) Transformer para recalcular PCT1/PCT2 ────────────────────────
def add_pct(X_arr):
    X_df = pd.DataFrame(X_arr, columns=feature_cols)
    X_df['PCT1'] = X_df.filter(regex='_1$').mean(axis=1)
    X_df['PCT2'] = X_df.filter(regex='_2$').mean(axis=1)
    return X_df.values

pct_transformer = FunctionTransformer(add_pct, validate=False)

# ─── 3) Define base pipeline usando o wrapper ────────────────────────────────
base_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pct', pct_transformer),
    ('mlp', MLPWrapper(random_state=42, max_iter=500))
])

# ─── 4) Define search spaces ─────────────────────────────────────────────────
param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'mlp__alpha': [1e-4, 1e-3],
    'mlp__learning_rate_init': [5e-4, 1e-3],
    'mlp__activation': ['relu', 'tanh', 'logistic'],
    'mlp__max_iter': [500, 1000]
}
param_dist = param_grid

# Para BayesSearchCV, usamos strings nas categorias
bayes_space = {
    'mlp__hidden_layer_sizes': Categorical(["(50,)", "(100,)", "(50, 50)", "(100, 50)"]),
    'mlp__alpha': Real(1e-4, 1e-3, prior='log-uniform'),
    'mlp__learning_rate_init': Real(5e-4, 1e-2, prior='log-uniform'),
    'mlp__activation': Categorical(['relu', 'tanh', 'logistic']),
    'mlp__max_iter': Integer(500, 1000)
}

# GeneticSearchCV aceita instâncias diretas
genetic_params = {
    'mlp__hidden_layer_sizes': GeneticCategorical([(50,), (100,), (50, 50), (100, 50)]),
    'mlp__alpha': Continuous(1e-4, 1e-3),
    'mlp__learning_rate_init': Continuous(5e-4, 1e-2),
    'mlp__activation': GeneticCategorical(['relu', 'tanh', 'logistic']),
    'mlp__max_iter': GeneticInteger(500, 1000)
}

# ─── 5) CV splitter ───────────────────────────────────────────────────────────
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── 6) Setup das buscas ─────────────────────────────────────────────────
searches = {
    'GridSearch': GridSearchCV(
        clone(base_pipe), param_grid, scoring='accuracy',
        cv=cv_splitter, n_jobs=-1, verbose=1
    ),
    'RandomizedSearch': RandomizedSearchCV(
        clone(base_pipe), param_dist, n_iter=10, scoring='accuracy',
        cv=cv_splitter, random_state=42, n_jobs=-1, verbose=1
    ),
    'BayesSearch': BayesSearchCV(
        clone(base_pipe), bayes_space, n_iter=20, scoring='accuracy',
        cv=cv_splitter, random_state=42, n_jobs=-1, verbose=1
    ),
    'GeneticSearch': GASearchCV(
        estimator=clone(base_pipe),
        param_grid=genetic_params,
        scoring='accuracy',
        cv=cv_splitter,
        population_size=20,
        generations=10,
        n_jobs=-1,
        verbose=True
    )
}

# ─── 7) Fit e coleta de resultados ─────────────────────────────────────
results = []
cv_scores = {}

for name, search in searches.items():
    print(f"\nRunning {name}...")

    start_time = time.time()
    search.fit(X.values, y)
    elapsed_time = time.time() - start_time

    # 7)a. Ajuste do best_params: Bayes vem em string, wrapper já converte
    best_params = search.best_params_.copy()

    # 7)b. Clone e configura o pipeline com os best_params
    model = clone(base_pipe).set_params(**best_params)

    # 7)c. Avalia com cross_val_score
    scores = cross_val_score(model, X.values, y, scoring='accuracy', cv=cv_splitter)
    cv_scores[name] = scores

    results.append({
        'Strategy': name,
        'Best Score': np.mean(scores),
        'Std Dev': np.std(scores),
        'Best Params': best_params,
        'Time(s)': elapsed_time
    })

# ─── 8) Summary DataFrame ────────────────────────────────────────────────────
summary_df = pd.DataFrame(results).sort_values('Best Score', ascending=False)
summary_df.to_csv('search_comparison_summary.csv', index=False)

# ─── 9) Salvar melhores hiperparâmetros ──────────────────────────────────────
# Criar uma lista de dicionários planos para montar o DataFrame
param_records = []
for r in results:
    flat = {'Strategy': r['Strategy']}
    flat.update(r['Best Params'])  # adiciona os hiperparâmetros como colunas
    param_records.append(flat)

params_df = pd.DataFrame(param_records)
params_df.to_csv('search_best_params.csv', index=False)

# ─── 10) Teste de Friedman + post hoc ─────────────────────────────────────────────
scores_df = pd.DataFrame(cv_scores)
friedman_stat, friedman_p = friedmanchisquare(*[scores_df[col] for col in scores_df.columns])
posthoc = sp.posthoc_nemenyi_friedman(scores_df.T.values)

# ─── 11) Save stats ──────────────────────────────────────────────────────────
with open('friedman_test_results.txt', 'w') as f:
    f.write(f"Friedman χ² = {friedman_stat:.3f}, p = {friedman_p:.4f}\n\n")
    f.write("Post hoc de Nemenyi:\n")
    f.write(posthoc.to_string())

# ─── 12) Print para terminal ────────────────────────────────────────────────────
print("\n=== Friedman Test ===")
print(f"χ² = {friedman_stat:.3f}, p = {friedman_p:.4f}\n")
print("=== Post-hoc de Nemenyi (p-values) ===")
print(posthoc, "\n")
print("=== Resumo dos Métodos ===")
print(summary_df[['Strategy', 'Best Score', 'Std Dev', 'Time(s)']], "\n")

# Print melhores hiperparâmetros
print("=== Melhores Hiperparâmetros por Estratégia ===")
for r in sorted(results, key=lambda x: -x['Best Score']):
    print(f"\n🔍 {r['Strategy']}")
    for param, value in r['Best Params'].items():
        print(f"    {param}: {value}")

# ─── 13) Plot boxplot de CV acurácias ────────────────────────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(data=scores_df)
plt.ylabel('CV Accuracy')
plt.title('Comparison of CV Accuracy by Search Strategy')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('search_strategy_accuracy_boxplot.png')
plt.show()

# ─── 14) Plot heatmap of post-hoc p-values ────────────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(posthoc, annot=True, fmt=".3f")
plt.title("Nemenyi post-hoc p-values")
plt.tight_layout()
plt.savefig('posthoc_nemenyi_heatmap.png')
plt.show()

# ─── 13b) Plot do tempo de execução de cada estratégia ───────────────────────
plt.figure(figsize=(8, 5))
sns.barplot(
    data=summary_df.sort_values('Time(s)'),
    x='Strategy',
    y='Time(s)',
    palette='Blues_d'
)
plt.title('Tempo de Execução por Estratégia de Busca')
plt.ylabel('Tempo (segundos)')
plt.xlabel('Estratégia')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('search_strategy_time_barplot.png')
plt.show()

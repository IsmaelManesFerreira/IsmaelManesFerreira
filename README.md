Parte 1 | Pré-processamento dos Dados



import pandas as pd

# 1. Carregar a base de dados
df = pd.read_csv('Titanic-Dataset.csv')

# 2. Limpeza de Dados
# Preencher valores ausentes nas colunas 'Age' e 'Fare' com a mediana
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Remover linhas com valores nulos restantes
df_clean = df.dropna()

# 3. Seleção de variáveis
# Selecionar colunas relevantes
df_selected = df_clean[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

# Converter a coluna 'Sex' para variáveis numéricas
df_selected['Sex'] = df_selected['Sex'].map({'male': 1, 'female': 0})

# Exibir as primeiras linhas do DataFrame processado
print(df_selected.head())

# 4. Divisão dos dados
X = df_selected[['Pclass', 'Sex', 'Age', 'Fare']]
y = df_selected['Survived']    

Parte 2 | Implementação do Algoritmo de Classificação (k-NN)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Dividir os dados em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar e treinar o classificador k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

Parte 3 | Avaliação de Desempenho

from sklearn.metrics import accuracy_score, confusion_matrix

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Gerar a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)

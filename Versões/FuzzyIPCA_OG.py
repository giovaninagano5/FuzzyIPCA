import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Carregar dados pré-processados (exemplo)
df = pd.read_excel('dados_economicos.xlsx')
df.columns = ['Data', 'cambio', 'vari_cambio', 'selic',
              'ibc_br','variacao_ibc', 'desemprego', 'commodities', 'ipca_esp', 'IPCA']

# Identify numeric columns (exclude datetime columns)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Scale only numeric data
scaler = MinMaxScaler()
df_scaled = df.copy()  # Create a copy to preserve original data
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

cambio_range = np.arange(1.5, 6.1, 0.1)        # R$/US$: min=1.5, max=6.0
selic_range = np.arange(2, 20, 0.5)         # Taxa Selic (%)
expectativas_range = np.arange(0, 1, 0.01)  # Expectativas de IPCA
ibc_range = np.arange(70, 115, 1)           # IBC-Br
ipca_range = np.arange(-0.2, 1.4, 0.01)          # IPCA-alvo

# Variáveis fuzzy de entrada
cambio = ctrl.Antecedent(cambio_range, 'cambio')
selic = ctrl.Antecedent(selic_range, 'selic')
expectativas = ctrl.Antecedent(expectativas_range, 'expectativas')
ibc_br = ctrl.Antecedent(ibc_range, 'ibc_br')

# Variável de saída
ipca = ctrl.Consequent(ipca_range, 'ipca')

# Funções de pertinência para o Câmbio 
cambio['baixo'] = fuzz.trapmf(cambio_range, [1.5, 1.5, 3, 4])
cambio['medio'] = fuzz.trimf(cambio_range, [3, 4, 5])
cambio['alto'] = fuzz.trapmf(cambio_range, [4, 5, 6, 6])        

# Funções de pertinência para a Selic
selic['baixo'] = fuzz.trapmf(selic_range, [2, 2, 3, 8])
selic['medio'] = fuzz.trimf(selic_range, [4, 8, 12])
selic['alto'] = fuzz.trapmf(selic_range, [9, 15, 20, 20])

# Funções de pertinência para as Expectativas do Mercado
expectativas['baixo'] = fuzz.trapmf(expectativas_range, [0, 0, 0.1, 0.3])
expectativas['medio'] = fuzz.trimf(expectativas_range, [0.1, 0.3, 0.6])
expectativas['alto'] = fuzz.trapmf(expectativas_range, [0.3, 0.6, 1, 1])

# Funções de pertinência para as Variações da Atividade Econômica
ibc_br['baixo'] = fuzz.trapmf(ibc_range, [70, 70, 85, 100])
ibc_br['medio'] = fuzz.trimf(ibc_range, [85, 95, 105])
ibc_br['alto'] = fuzz.trapmf(ibc_range, [95, 105, 110, 115])

# Funções de pertinência personalizadas para o IPCA
ipca['baixo'] = fuzz.trapmf(ipca_range, [0, 0, 0.15, 0.3])
ipca['medio'] = fuzz.trimf(ipca_range, [0, 0.4, 0.7])
ipca['alto'] = fuzz.trapmf(ipca_range, [0.4, 0.7, 1.4, 1.4])

# Lista de regras
regras = []

# Regras para inflação ALTA (combinações onde IPCA tende a subir)
regras.append(ctrl.Rule(cambio['alto'] & expectativas['alto'], ipca['alto']))     # R1
regras.append(ctrl.Rule(cambio['alto'] & ibc_br['baixo'], ipca['alto']))          # R2
regras.append(ctrl.Rule(selic['baixo'] & ibc_br['baixo'], ipca['alto']))          # R3

# Regras para inflação MÉDIA (cenários intermediários)
regras.append(ctrl.Rule(selic['medio'] & ibc_br['medio'], ipca['medio']))         # R5
regras.append(ctrl.Rule(cambio['medio'] & ibc_br['baixo'], ipca['medio']))        # R6
regras.append(ctrl.Rule(expectativas['alto'] & selic['alto'], ipca['medio']))     # R7

# Regras para inflação BAIXA (combinações que reduzem IPCA)
regras.append(ctrl.Rule(cambio['baixo'] & selic['alto'], ipca['baixo']))          # R9
regras.append(ctrl.Rule(expectativas['baixo'] & cambio['baixo'], ipca['baixo']))  # R10
regras.append(ctrl.Rule(ibc_br['baixo'] & cambio['baixo'], ipca['baixo']))        # R11

# Sistema de controle
sistema_controle = ctrl.ControlSystem(regras)
sistema = ctrl.ControlSystemSimulation(sistema_controle)
ipca.defuzzify_method = 'centroid'

# Criar listas para armazenar os resultados
datas = df['Data']
ipca_real = df['IPCA']
ipca_esperado = df['ipca_esp']
ipca_fuzzy = []

# Calcular as previsões fuzzy para todos os períodos
for i in range(len(df)):
    sistema.input['cambio'] = df['cambio'].iloc[i]
    sistema.input['selic'] = df['selic'].iloc[i]
    sistema.input['expectativas'] = df['ipca_esp'].iloc[i]
    sistema.input['ibc_br'] = df['ibc_br'].iloc[i]
    
    sistema.compute()
    ipca_fuzzy.append(sistema.output['ipca'])

# Configurar o gráfico
plt.figure(figsize=(14, 7))
plt.plot(datas, ipca_real, 'b-', linewidth=2, label='IPCA Real')
plt.plot(datas, ipca_esperado, 'g--', linewidth=2, label='IPCA Esperado')
plt.plot(datas, ipca_fuzzy, 'r-.', linewidth=2, label='Modelo Fuzzy')

# Configurações do gráfico
plt.title('Comparação: Modelo Fuzzy vs IPCA Esperado vs IPCA Real', fontsize=14)
plt.xlabel('Data', fontsize=12)
plt.ylabel('IPCA', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Rotacionar as datas para melhor visualização
plt.xticks(rotation=45)

# Ajustar layout para evitar cortes
plt.tight_layout()

mae_fuzzy = mean_absolute_error(ipca_real, ipca_fuzzy)
mae_esperado = mean_absolute_error(ipca_real, ipca_esperado)

plt.figtext(0.15, 0.82, f'MAE Fuzzy: {mae_fuzzy:.4f}', fontsize=10)
plt.figtext(0.15, 0.78, f'MAE Esperado: {mae_esperado:.4f}', fontsize=10)

# Mostrar o gráfico
plt.show()
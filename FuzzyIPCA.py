import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Carrega o dataset
df = pd.read_excel('dados_economicos.xlsx')
df.columns = ['Data', 'cambio', 'vari_cambio', 'selic',
              'ibc_br','variacao_ibc', 'desemprego', 'commodities', 'ipca_esp', 'IPCA']

# Define universos de discurso
vari_cambio_range = np.arange(-0.45, 0.55, 0.01)        # R$/US$: min=1.5, max=6.0
selic_range = np.arange(1.5, 20, 0.5)         # Taxa Selic (%)
expectativas_range = np.arange(0, 0.8, 0.01)  # Expectativas de IPCA
commodities_range = np.arange(65, 220, 1)     # Índice Commodities
ipca_range = np.arange(-0.2, 1.4, 0.01)          # IPCA-alvo

# Variáveis fuzzy de entrada
cambio = ctrl.Antecedent(vari_cambio_range, 'cambio')
selic = ctrl.Antecedent(selic_range, 'selic')
expectativas = ctrl.Antecedent(expectativas_range, 'ipca_esp')
commodities = ctrl.Antecedent(commodities_range, 'commodities')

# Variável de saída
ipca = ctrl.Consequent(ipca_range, 'ipca')

# Funções de pertinência para o Câmbio 
cambio['baixo'] = fuzz.trapmf(vari_cambio_range, [-0.45, -0.45, -0.2, 0])
cambio['medio'] = fuzz.trimf(vari_cambio_range, [-0.2, 0, 0.2])
cambio['alto'] = fuzz.trapmf(vari_cambio_range, [0, 0.2, 0.55, 0.55])        

# Funções de pertinência para a Selic
selic['baixo'] = fuzz.trapmf(selic_range, [1.5, 1.5, 3, 8])
selic['medio'] = fuzz.trimf(selic_range, [4, 8, 12])
selic['alto'] = fuzz.trapmf(selic_range, [9, 15, 20, 20])

# Funções de pertinência para as Expectativas do Mercado
expectativas['baixo'] = fuzz.trapmf(expectativas_range, [0, 0, 0.1, 0.3])
expectativas['medio'] = fuzz.trimf(expectativas_range, [0.1, 0.3, 0.6])
expectativas['alto'] = fuzz.trapmf(expectativas_range, [0.3, 0.6, 0.8, 0.8])

# Funções de pertinência para os Índices de Preço dos Commodities
commodities['baixo'] = fuzz.trapmf(commodities_range, [65, 65, 95, 120])
commodities['medio'] = fuzz.trimf(commodities_range, [110, 135, 160])
commodities['alto'] = fuzz.trapmf(commodities_range, [135, 160, 220, 220])

# Funções de pertinência personalizadas para o IPCA
ipca['baixo'] = fuzz.trapmf(ipca_range, [-0.2, -0.2, 0.1, 0.3])
ipca['medio'] = fuzz.trimf(ipca_range, [0.2, 0.5, 0.7])
ipca['alto'] = fuzz.trapmf(ipca_range, [0.5, 0.8, 1.4, 1.4])

# Lista de regras
regras = []

# Regras para inflação ALTA
regras.append(ctrl.Rule(cambio['alto'] & expectativas['alto'], ipca['alto']))
regras.append(ctrl.Rule(cambio['alto'] & commodities['alto'], ipca['alto']))
regras.append(ctrl.Rule(cambio['baixo'] & expectativas['alto'], ipca['alto']))
regras.append(ctrl.Rule(selic['baixo'] & commodities['alto'], ipca['alto']))
regras.append(ctrl.Rule(cambio['baixo'] & commodities['alto'], ipca['alto']))

# Regras para inflação MÉDIA
regras.append(ctrl.Rule(selic['medio'] & commodities['medio'], ipca['medio']))
regras.append(ctrl.Rule(cambio['medio'] & commodities['alto'], ipca['medio']))
regras.append(ctrl.Rule(expectativas['alto'] & selic['alto'], ipca['medio']))
regras.append(ctrl.Rule(cambio['baixo'] & selic['medio'], ipca['medio']))
regras.append(ctrl.Rule(selic['baixo'] & expectativas['medio'], ipca['medio']))

# Regras para inflação BAIXA
regras.append(ctrl.Rule(cambio['medio'] & selic['alto'], ipca['baixo']))
regras.append(ctrl.Rule(expectativas['baixo'] & cambio['medio'], ipca['baixo']))
regras.append(ctrl.Rule(commodities['baixo'] & cambio['medio'], ipca['baixo']))

# Sistema de controle
sistema_controle = ctrl.ControlSystem(regras)
sistema = ctrl.ControlSystemSimulation(sistema_controle)
ipca.defuzzify_method = 'centroid'

# Criar listas para resultados
datas = df['Data']
ipca_real = df['IPCA']
ipca_esperado = df['ipca_esp']
ipca_fuzzy = []

# Calcular previsões fuzzy
for i in range(len(df)):
    sistema.input['cambio'] = df['vari_cambio'].iloc[i]
    sistema.input['selic'] = df['selic'].iloc[i]
    sistema.input['commodities'] = df['commodities'].iloc[i]
    sistema.input['ipca_esp'] = df['ipca_esp'].iloc[i]
        
    sistema.compute()
    ipca_fuzzy.append(sistema.output['ipca'])

# Configurar gráfico
plt.figure(figsize=(14, 7))
plt.plot(datas, ipca_real, 'b-', linewidth=2, label='IPCA Real')
plt.plot(datas, ipca_esperado, 'g--', linewidth=2, label='IPCA Esperado')
plt.plot(datas, ipca_fuzzy, 'r-.', linewidth=2, label='Modelo Fuzzy')

# Configurações do gráfico
plt.title('Comparação: Modelo Fuzzy vs Expectativas vs Real', fontsize=14)
plt.xlabel('Data', fontsize=12)
plt.ylabel('IPCA', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# Calcular métricas de erro
mae_fuzzy = mean_absolute_error(ipca_real, ipca_fuzzy)
mae_esperado = mean_absolute_error(ipca_real, ipca_esperado)

plt.text(0.015, 0.8, f'MAE Fuzzy: {mae_fuzzy:.4f}', transform=plt.gca().transAxes, 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.015, 0.75, f'MAE Expectativas: {mae_esperado:.4f}', transform=plt.gca().transAxes, 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.show()
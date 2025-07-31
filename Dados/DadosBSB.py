from bcb import sgs
import pandas as pd

codigos = {
    'IPCA': 433, 
    'Cambio': 3698, 
    'Selic': 4189,
    'IBC_br': 24363,
    'desemprego': 24369
}

dados = sgs.get(codigos, start='2005-01-01')
dados.index.name = 'Data'

dados.to_excel('dados_economicos.xlsx')

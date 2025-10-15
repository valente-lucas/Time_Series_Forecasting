####  Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from functools import partial
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Theta, SeasonalNaive, Naive, AutoETS
from statsforecast.core import StatsForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, mape, smape, mase
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, PatchTST
from neuralforecast.losses.pytorch import MAE

####  Base de dados
base=pd.read_excel(
    r'C:\Users\lucas\Documents\Trabalhos\Data_Science\Portfolio\Series_temporais\CONSUMO_MENSAL_DE_ENERGIA_ELÉTRICA_POR_CLASSE.xlsx',
    sheet_name=None
    )

#### Intervalo de tempo da série temporal
ultima_atualizacao=datetime.strptime(base['TOTAL'].iloc[18,0][41:48],'%Y-%m')
inicio_serie=datetime.strptime('2004-01','%Y-%m')
intervalo_serie = pd.date_range(start=inicio_serie, end=ultima_atualizacao, freq="M")
anos_serie=ultima_atualizacao.year-inicio_serie.year

### Trato as siglas de UF e separo os estados com e sem horário de verão  
estados_siglas = {"Acre": "AC", "Alagoas": "AL", "Amapa": "AP", "Amazonas": "AM", "Bahia": "BA",
                  "Ceara": "CE", "Distrito Federal": "DF", "Espirito Santo": "ES", "Goias": "GO",
                  "Maranhao": "MA", "Mato Grosso": "MT", "Mato Grosso do Sul": "MS", "Minas Gerais": "MG",
                  "Para": "PA", "Paraiba": "PB", "Parana": "PR", "Pernambuco": "PE", "Piaui": "PI",
                  "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN", "Rio Grande do Sul": "RS",
                  "Rondonia": "RO", "Roraima": "RR", "Santa Catarina": "SC", "Sao Paulo": "SP",
                  "Sergipe": "SE", "Tocantins": "TO"
                  }

estados_horario_verao = ['MS','MT','GO','RJ','SP','MG','ES','PR','SC','RS', 'DF']
estados_sem_horario_verao = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'MA', 'PA', 'PB', 'PE', 'PI', 'RN', 'RO', 'RR', 'SE', 'TO']


###Gero os dataframes a partir da base de dados
### CONSUMIDOR CATIVO
## Valores de consumo por UF
UFs_CATIVO=base['CONSUMO CATIVO POR UF'].iloc[6:33,0].map(estados_siglas)
VALORES_CATIVO_UF = base['CONSUMO CATIVO POR UF'].iloc[6:33, 1:len(intervalo_serie)+1]
df_cativo_UF = pd.DataFrame(
    VALORES_CATIVO_UF.T.to_numpy(),
    index=intervalo_serie,
    columns='CONSUMO_CATIVO_'+UFs_CATIVO
    ).fillna(0)

## Valores de consumo por Classe 
df_cativo_classes = pd.DataFrame(index=intervalo_serie)
for i in range(anos_serie+1):
    referencia = (anos_serie-i)*21 + 19
    for j in range(0, 4):
        classes = base['CATIVO'].iloc[referencia+j, 0]
        col_name = f"CONSUMO_CATIVO_{classes}"
        consumo = base['CATIVO'].iloc[referencia+j, 1:13].fillna(0).to_list()
        if col_name not in df_cativo_classes.columns:
            df_cativo_classes[col_name] = np.nan
        df_cativo_classes.iloc[12*i:12*(i+1), df_cativo_classes.columns.get_loc(col_name)] = consumo

## Número de Consumidores por UF 
consumidores_cativo_UF = base['CONSUMIDORES CATIVOS POR UF'].iloc[6:33, 1:len(intervalo_serie)+1]
df_consumidor_cativo_UF = pd.DataFrame(
    consumidores_cativo_UF.T.to_numpy(),
    index=intervalo_serie,
    columns='CONSUMIDORES_CATIVO_'+UFs_CATIVO
    ).fillna(0)

## Número de consumidores por Classe 
df_consumidor_cativo_classes = pd.DataFrame(index=intervalo_serie)
for i in range(anos_serie+1):
    referencia = (anos_serie-i)*21 + 19
    for j in range(0, 4):
        classes = base['CONSUMIDORES CATIVOS'].iloc[referencia+j, 0]
        col_name = f"CONSUMIDORES_CATIVO_{classes}"
        consumo = base['CONSUMIDORES CATIVOS'].iloc[referencia+j, 1:13].fillna(0).to_list()
        
        
        if col_name not in df_consumidor_cativo_classes.columns:
            df_consumidor_cativo_classes[col_name] = np.nan
        
        df_consumidor_cativo_classes.iloc[12*i:12*(i+1), df_consumidor_cativo_classes.columns.get_loc(col_name)] = consumo

### CONSUMIDOR LIVRE
## Valores de consumo por UF
UFs_livre=base['CONSUMO LIVRE POR UF'].iloc[6:33,0].map(estados_siglas)
valores_livre_UF = base['CONSUMO LIVRE POR UF'].iloc[6:33, 1:len(intervalo_serie)+1]
df_livre_UF = pd.DataFrame(
    valores_livre_UF.T.to_numpy(),
    index=intervalo_serie,
    columns='CONSUMO_LIVRE_'+UFs_livre
    ).fillna(0)

## Valores de consumo por Classe 
df_livre_classes = pd.DataFrame(index=intervalo_serie)
for i in range(anos_serie+1):
    referencia = (anos_serie-i)*21 + 19
    for j in range(0, 4):
        classes = base['LIVRE'].iloc[referencia+j, 0]
        col_name = f"CONSUMO_LIVRE_{classes}"
        consumo = base['LIVRE'].iloc[referencia+j, 1:13].fillna(0).to_list()


        if col_name not in df_livre_classes.columns:
            df_livre_classes[col_name] = np.nan

        df_livre_classes.iloc[12*i:12*(i+1), df_livre_classes.columns.get_loc(col_name)] = consumo

##  Número de consumidores por UF 
consumidores_livre_UF = base['CONSUMIDORES LIVRES POR UF'].iloc[6:33, 1:len(intervalo_serie)+1]
df_consumidor_livre_UF = pd.DataFrame(
    consumidores_livre_UF.T.to_numpy(),
    index=intervalo_serie,
    columns='CONSUMIDORES_LIVRE_'+UFs_livre
    ).fillna(0)

## Número de consumidores por Classe 
df_consumidor_livre_classes = pd.DataFrame(index=intervalo_serie)
for i in range(anos_serie+1):
    referencia = (anos_serie-i)*21 + 19
    for j in range(0, 4):
        classes = base['CONSUMIDORES LIVRES'].iloc[referencia+j, 0]
        col_name = f"CONSUMIDORES_LIVRE_{classes}"
        consumo = base['CONSUMIDORES LIVRES'].iloc[referencia+j, 1:13].fillna(0).to_list()
        
        
        if col_name not in df_consumidor_livre_classes.columns:
            df_consumidor_livre_classes[col_name] = np.nan
        
        df_consumidor_livre_classes.iloc[12*i:12*(i+1), df_consumidor_livre_classes.columns.get_loc(col_name)] = consumo


### Dataframe no formato largo (wide-format)
agrupado_uf_cativo = {
    "CONSUMO_CATIVO_CHV": df_cativo_UF[[c for c in df_cativo_UF.columns if any(estado in c for estado in estados_horario_verao) and 'CONSUMO_CATIVO' in c]].sum(axis=1),
    "CONSUMO_CATIVO_SHV": df_cativo_UF[[c for c in df_cativo_UF.columns if any(estado in c for estado in estados_sem_horario_verao) and 'CONSUMO_CATIVO' in c]].sum(axis=1),
    "CONSUMIDORES_CATIVO_CHV": df_consumidor_cativo_UF[[c for c in df_consumidor_cativo_UF.columns if any(estado in c for estado in estados_horario_verao) and 'CONSUMIDORES_CATIVO' in c]].sum(axis=1),
    "CONSUMIDORES_CATIVO_SHV": df_consumidor_cativo_UF[[c for c in df_consumidor_cativo_UF.columns if any(estado in c for estado in estados_sem_horario_verao) and 'CONSUMIDORES_CATIVO' in c]].sum(axis=1)
    }

agrupado_uf_livre = {
    "CONSUMO_LIVRE_CHV": df_livre_UF [[c for c in df_livre_UF.columns if any(estado in c for estado in estados_horario_verao) and 'CONSUMO_LIVRE' in c]].sum(axis=1),
    "CONSUMO_LIVRE_SHV": df_livre_UF[[c for c in df_livre_UF.columns if any(estado in c for estado in estados_sem_horario_verao) and 'CONSUMO_LIVRE' in c]].sum(axis=1),
    "CONSUMIDORES_LIVRE_CHV": df_consumidor_livre_UF[[c for c in df_consumidor_livre_UF.columns if any(estado in c for estado in estados_horario_verao) and 'CONSUMIDORES_LIVRE' in c]].sum(axis=1),
    "CONSUMIDORES_LIVRE_SHV": df_consumidor_livre_UF[[c for c in df_consumidor_livre_UF.columns if any(estado in c for estado in estados_sem_horario_verao) and 'CONSUMIDORES_LIVRE' in c]].sum(axis=1)
    }      

df_cativo_agrupado= pd.DataFrame(agrupado_uf_cativo , index=df_cativo_UF.index)
df_livre_agrupado= pd.DataFrame(agrupado_uf_livre , index=df_cativo_UF.index)
df_wide=pd.concat([df_cativo_agrupado,df_cativo_classes,df_consumidor_cativo_classes,df_livre_agrupado, df_livre_classes,df_consumidor_livre_classes],
                  axis=1
                  )
df_wide.to_csv("base_wide.csv", index=True, encoding="utf-8")

### Dataframe no formato compacto (compact-format)
linhas = []

for col in df_wide.columns:
    partes = col.split("_")
    
    if partes[0] == "CONSUMO":
        tipo = partes[1] 
        categoria = "_".join(partes[2:])
        consumo_array = df_wide[col].to_numpy()
        
        consumidores_col = col.replace("CONSUMO", "CONSUMIDORES")
        consumidores_array = df_wide[consumidores_col].to_numpy() if consumidores_col in df_wide else None
        
        linhas.append({
            "Tipo": tipo,
            "Categoria": categoria,
            "Consumo": consumo_array,
            "Consumidores": consumidores_array,
            "Inicio": inicio_serie,
            "Frequência": 'Mensal',
            "Comprimento_Serie": len(consumo_array)
        })

df_compact = pd.DataFrame(linhas)
df_compact.to_csv("base_compact.csv", index=False, encoding="utf-8")

### Dataframe no formato expandido (expanded-format)
df_wide_1 = df_wide.reset_index().rename(columns={'index': 'Data'})
df_expanded = pd.wide_to_long(df_wide_1,stubnames=['CONSUMO', 'CONSUMIDORES'],i='Data',j='Categoria',sep='_',suffix='.+').reset_index()
df_expanded = df_expanded.sort_values(['Data', 'Categoria']).reset_index(drop=True)
df_expanded[['Tipo_consumidor', 'Categoria']] = df_expanded['Categoria'].str.split('_', n=1, expand=True)
df_expanded.to_csv("base_expanded.csv", decimal=",",  index=False, encoding="utf-8")

### ANÁLISE DA SÉRIE TEMPORAL DE CONSUMO
### Consumo total (cativo + livre)
total_consumo_cativo = df_cativo_UF.sum(axis=1)
total_consumo_livre = df_livre_UF.sum(axis=1)
df_analise_consumo = pd.DataFrame({'CONSUMO_CATIVO_Total': total_consumo_cativo,'CONSUMO_LIVRE_Total': total_consumo_livre,'CONSUMO_TOTAL': total_consumo_cativo + total_consumo_livre})
total_consumo = df_analise_consumo['CONSUMO_TOTAL'].replace(0, np.nan).fillna(method='ffill').values

## Detrending via LOESS (ou LOWESS)
trend_loess = lowess(total_consumo, df_analise_consumo.index, frac=0.05, return_sorted=False)
detrended_loess = total_consumo - trend_loess
df_analise_consumo['detrended_loess'] = detrended_loess

## Detrending via médias móveis (decomposição aditiva)
serie_aditiva = df_analise_consumo['CONSUMO_TOTAL'].replace(0, np.nan).fillna(method='ffill')
decomp_aditiva = seasonal_decompose(serie_aditiva, model='additive', period=12)
detrended_aditiva = serie_aditiva - decomp_aditiva.trend
df_analise_consumo['detrended_aditiva'] = detrended_aditiva

## Detrending via médias móveis (decomposição multiplicativa)
serie_mult = df_analise_consumo['CONSUMO_TOTAL'].replace(0, np.nan).fillna(method='ffill')
decomp_mult = seasonal_decompose(serie_mult, model='multiplicative', period=12)
detrended_mult = serie_mult / decomp_mult.trend
df_analise_consumo['detrended_mult'] = detrended_mult

## Detrending via logaritmo + decomposição (aditiva) - Via comparação com método multiplicativo
df_analise_consumo['log_consumo'] = np.log(df_analise_consumo['CONSUMO_TOTAL'])
serie_log = df_analise_consumo['log_consumo'].replace(-np.inf, np.nan).fillna(method='ffill')
decomp_log = seasonal_decompose(serie_log, model='additive', period=12)
detrended_log= df_analise_consumo['log_consumo'] - decomp_log.trend
df_analise_consumo['detrended_log'] = detrended_log

## Avaliação dos métodos de detrending
resultados_analise_metodos = {
    'LOESS': {
        'Correlação': np.corrcoef(total_consumo, df_analise_consumo['detrended_loess'].fillna(method='ffill').fillna(method='bfill').values)[0,1],
        'p-ADF': adfuller(df_analise_consumo['detrended_loess'].fillna(method='ffill').fillna(method='bfill').values)[1],
        'p-KPSS': kpss(df_analise_consumo['detrended_loess'].fillna(method='ffill').fillna(method='bfill').values, regression='c')[1],
        'Variância relativa': np.nanvar(df_analise_consumo['detrended_loess'].values) / np.nanvar(total_consumo)
    },
    'Aditiva': {
        'Correlação': np.corrcoef(total_consumo, df_analise_consumo['detrended_aditiva'].fillna(method='ffill').fillna(method='bfill').values)[0,1],
        'p-ADF': adfuller(df_analise_consumo['detrended_aditiva'].fillna(method='ffill').fillna(method='bfill').values)[1],
        'p-KPSS': kpss(df_analise_consumo['detrended_aditiva'].fillna(method='ffill').fillna(method='bfill').values, regression='c')[1],
        'Variância relativa': np.nanvar(df_analise_consumo['detrended_aditiva'].values) / np.nanvar(total_consumo)
    },
    'Multiplicativa': {
        'Correlação': np.corrcoef(total_consumo, df_analise_consumo['detrended_mult'].fillna(method='ffill').fillna(method='bfill').values)[0,1],
        'p-ADF': adfuller(df_analise_consumo['detrended_mult'].fillna(method='ffill').fillna(method='bfill').values)[1],
        'p-KPSS': kpss(df_analise_consumo['detrended_mult'].fillna(method='ffill').fillna(method='bfill').values, regression='c')[1],
        'Variância relativa': np.nanvar(df_analise_consumo['detrended_mult'].values) / np.nanvar(total_consumo)
    },
    'Logarítmica': {
        'Correlação': np.corrcoef(total_consumo, df_analise_consumo['detrended_log'].fillna(method='ffill').fillna(method='bfill').values)[0,1],
        'p-ADF': adfuller(df_analise_consumo['detrended_log'].fillna(method='ffill').fillna(method='bfill').values)[1],
        'p-KPSS': kpss(df_analise_consumo['detrended_log'].fillna(method='ffill').fillna(method='bfill').values, regression='c')[1],
        'Variância relativa': np.nanvar(df_analise_consumo['detrended_log'].values) / np.nanvar(total_consumo)
    }
}
avaliacao_df = pd.DataFrame(resultados_analise_metodos).T
print(avaliacao_df)

## Decomposição de Fourier da série detrendida escolhida
serie_detrendida = df_analise_consumo['detrended_loess'].values
N = len(serie_detrendida)
espectro_fourier = np.fft.rfft(serie_detrendida)
variancia_total = (np.abs(espectro_fourier)**2).sum()
indices = np.arange(1, len(espectro_fourier)-1)
magnitude_indices = np.abs(espectro_fourier[indices])
indices_ordenados = indices[np.argsort(magnitude_indices)[::-1]]

## Avaliação do erro de reconstrução da série detrendida em função do número de harmônicos (k) selecionados
max_k = len(indices_ordenados)
ks = np.arange(1, max_k+1)
mse_list, rmse_list, mae_list, variancia_frac_list = [], [], [], []
for k in ks:
    Xf = np.zeros_like(espectro_fourier, dtype=complex)
    Xf[0] = espectro_fourier[0]                      
    corte = indices_ordenados[:k]
    Xf[corte] = espectro_fourier[corte]            
    serie_recourier = np.fft.irfft(Xf, n=N)
    
    resid = serie_detrendida - serie_recourier
    mse_k = np.mean(resid**2)
    rmse_k = np.sqrt(mse_k)
    mae_k = np.mean(np.abs(resid))
    mse_list.append(mse_k)
    rmse_list.append(rmse_k)
    mae_list.append(mae_k)
    variancia_frac_list.append(np.sum(np.abs(Xf)**2) / variancia_total)

mse_arr = np.array(mse_list)
rmse_arr = np.array(rmse_list)
mae_arr = np.array(mae_list)
variancia_arr = np.array(variancia_frac_list)

## Seleção de k para reconstrução e obtenção da série sem tendência e sem sazionalidade
k_sel = 53 
Xf = np.zeros_like(espectro_fourier, dtype=complex)
Xf[0] = espectro_fourier[0]
Xf[indices_ordenados[:k_sel]] = espectro_fourier[indices_ordenados[:k_sel]]
serie_recons_fourier = np.fft.irfft(Xf, n=N)

serie_comp_irregular = serie_detrendida - serie_recons_fourier
df_analise_consumo['serie_comp_irregular'] = serie_detrendida - serie_recons_fourier

#### PREVISÕES DA SÉRIE TEMPORAL DE CONSUMO
serie_previsao = pd.DataFrame({
    'unique_id': 'consumo',
    'ds': df_analise_consumo.index,
    'y': serie_comp_irregular,#'y': serie_detrendida, #'y': total_consumo,
})

serie_previsao_comp_irregular = pd.DataFrame({
    'unique_id': 'consumo',
    'ds': df_analise_consumo.index,
    'y': serie_comp_irregular
})

serie_previsao_consumo = pd.DataFrame({
    'unique_id': 'consumo',
    'ds': df_analise_consumo.index,
    'y': total_consumo
})

serie_previsao_detrendida = pd.DataFrame({
    'unique_id': 'consumo',
    'ds': df_analise_consumo.index,
    'y': serie_detrendida
})
## Divisão em treino e validação (V ultimos para validação)
h=12 
serie_treino = serie_previsao.iloc[:-h]
serie_validacao = serie_previsao.iloc[-h:]

## Previsão com modelos estatísticos (h para previsão)
sf = StatsForecast(
    models=[Naive(),SeasonalNaive(season_length=12), AutoETS(season_length=12),AutoARIMA(season_length=12),Theta(season_length=12)],
    freq='M',
    n_jobs=-1
)
sf.fit(serie_treino)
previsoes_stats = sf.predict(h=h)
previsoes_stats = previsoes_stats.merge(serie_validacao[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')

## Previsão com modelos neurais
nf = NeuralForecast(
    models=[NBEATS(h=h, input_size=60, max_steps=200, loss=MAE()),
            NHITS(h=h, input_size=60, max_steps=200, loss=MAE()),
            PatchTST(h=h, input_size=60, max_steps=200, loss=MAE())],
    freq='M'
)
nf.fit(serie_treino)
previsoes_neurais_1 = nf.predict(h=h)
previsoes_combinadas = previsoes_neurais_1.merge(
    previsoes_stats,
    on=['unique_id', 'ds'],
    how='inner' # Use 'inner' para garantir que você merge apenas onde as datas coincidem
)

#def calcular_metricas(y_true, y_pred, y_train):
#    return {
#        'RMSE': np.sqrt(mse(y_true, y_pred)),
#        'MAE': mae(y_true, y_pred),
#        'MAPE': mape(y_true, y_pred),
#        'sMAPE': smape(y_true, y_pred),
#        'MASE': mase(y_true, y_pred, y_train, seasonality=12)
#    }

cv_previsoes_stats = sf.cross_validation(df=serie_previsao,h=h,step_size=12, n_windows=6)
cv_previsoes_neurais = nf.cross_validation(df=serie_previsao, h=h, step_size=12, n_windows=6)
cv_previsoes = cv_previsoes_stats.merge(cv_previsoes_neurais,how='outer')

serie_treino= serie_treino.rename(columns={'unique_id':'id_col', 'ds':'time_col', 'y':'target_col'})
serie_validacao= serie_validacao.rename(columns={'unique_id':'id_col', 'ds':'time_col', 'y':'target_col'})
cv_previsoes= cv_previsoes.rename(columns={'unique_id':'id_col', 'ds':'time_col', 'y':'target_col'})
resultados_cv = []
for cutoff, grupo in cv_previsoes.groupby('cutoff'):
    grupo=grupo.drop(columns=['cutoff'])
    resultado =evaluate(
        df=grupo,
        metrics=[mae, mse, mape, smape, partial(mase, seasonality=12)],
        train_df=serie_treino,
        id_col='id_col',
        time_col='time_col',
        target_col='target_col')
    resultado['cutoff'] = cutoff
    resultados_cv.append(resultado)
resultados_cv = pd.concat(resultados_cv, ignore_index=True)
cv_metricas_medias = (resultados_cv
                      .drop(columns=['id_col', 'cutoff'], errors='ignore')
                      .groupby('metric')
                      .mean(numeric_only=True))

cv_metricas_medias = cv_metricas_medias[cv_metricas_medias.loc['mse'].sort_values().index]
cv_metricas_medias = cv_metricas_medias.drop(columns=['AutoETS'],axis=1)
melhores_modelos = cv_metricas_medias.columns[0:3]

#Faço as previsões futuras
previsoes_stats_future= sf.predict(h=12)
previsoes_neurais_future = nf.predict(h=12)
previsoes_future = previsoes_neurais_future.merge(previsoes_stats_future, on=['unique_id', 'ds'], how='outer')

#### Plotes 
## Comparação do consumo total, cativo e livre ao longo do tempo
plt.figure(figsize=(12, 6))
plt.plot(df_analise_consumo.index, total_consumo_cativo.replace(0, np.nan).fillna(method='ffill'), label='Cativo', color='b')
plt.plot(df_analise_consumo.index, total_consumo_livre.replace(0, np.nan).fillna(method='ffill'), label='Livre', color='r')
plt.plot(df_analise_consumo.index, total_consumo, label='Total', color='k')
plt.title('Consumo Cativo, Livre e Total', fontsize=14)
plt.xlabel('Tempo')
plt.ylabel('Consumo [MWh]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

## Comparação das três formas de detrending em subplots
fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
ax[0].plot(df_analise_consumo.index, df_analise_consumo['detrended_loess'], color='c')
ax[0].set_title('Série Detrendida (LOESS)')
ax[0].grid()
ax[1].plot(df_analise_consumo.index, df_analise_consumo['detrended_aditiva'], color='g')
ax[1].set_title('Série Detrendida (Decomposição Aditiva)')
ax[1].grid()
ax[2].plot(df_analise_consumo.index, df_analise_consumo['detrended_mult'], color='m')
ax[2].set_title('Série Detrendida (Decomposição Multiplicativa)')
ax[2].grid()
ax[3].plot(df_analise_consumo.index, df_analise_consumo['detrended_log'], color='hotpink')
ax[3].set_title('Série Detrendida (Decomposição Multiplicativa)')
ax[3].grid()
plt.tight_layout()
plt.show() 

## Plot das tendências extraídas pelos métodos
plt.figure(figsize=(12,6))
plt.plot(df_analise_consumo.index, trend_loess, label='Tendência LOESS', color='c', alpha=0.7)
plt.plot(df_analise_consumo.index, decomp_aditiva.trend, label='Tendência Aditiva', color='g', alpha=0.7)
plt.plot(df_analise_consumo.index, decomp_mult.trend, label='Tendência Multiplicativa', color='m', alpha=0.7)
plt.plot(df_analise_consumo.index, decomp_log.trend, label='Tendência log_multi', color='hotpink', alpha=0.7)
plt.title("Comparação das Tendências Extraídas")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

## Gráfico combinado com as três séries detrendidas
plt.figure(figsize=(12,6))
plt.plot(df_analise_consumo.index, df_analise_consumo['detrended_loess'], label='LOESS', color='c', alpha=0.7)
plt.plot(df_analise_consumo.index, df_analise_consumo['detrended_aditiva'], label='Aditiva', color='g', alpha=0.7)
plt.plot(df_analise_consumo.index, df_analise_consumo['detrended_mult'], label='Multiplicativa', color='m', alpha=0.7)
plt.plot(df_analise_consumo.index, df_analise_consumo['detrended_log'], label='log_multi', color='hotpink', alpha=0.7)
plt.title("Comparação das Séries Detrendidas")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

## Plot de Autocorrelação (ACF)
plot_acf(detrended_loess, lags=128, zero=False, alpha=0.05)
plt.title("Autocorrelação - Série Detrendida (LOESS)")
plt.tight_layout()
plt.show()

## Plot de Autocorrelação Parcial (PACF)
plot_pacf(detrended_loess, lags=128, zero=False, alpha=0.05, method='ywm') 
plt.title("Autocorrelação Parcial - Série Detrendida (LOESS)")
plt.show()

## Plot do espectro dos períodos da série de Fourier
plt.figure(figsize=(10,4))
plt.plot(1/np.fft.rfftfreq(N), np.abs(espectro_fourier))
plt.title('Espectro de Fourier da série detrendida')
plt.xlabel('Período')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

## Plot das métricas de erro e da fração da variância explicada por número de harmônicos selecionados
fig, ax1 = plt.subplots(figsize=(10,5))
l1, = ax1.plot(ks, rmse_arr, label='RMSE', color='mediumpurple')
l2, = ax1.plot(ks, mae_arr, label='MAE', color='coral')
l4, = ax1.plot(ks, mse_arr, label='MSE', color='teal')
ax1.set_xlabel('Número de harmônicos (k)')
ax1.set_ylabel('Erro')
ax1.set_yscale('log')
ax1.grid()
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.get_offset_text().set_fontsize(10)
ax2 = ax1.twinx()
l3, = ax2.plot(ks, variancia_arr, color='gold', label='Variância explicada')
ax2.set_ylabel('Fração da variância explicada')
ax2.set_ylim(0, 1.05)
lines = [l1, l2, l4, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=10)
plt.title('Erro e fração da variância explicada por número de harmônicos', pad=30)
plt.subplots_adjust(top=0.82)
plt.show()



## Plot da série detrendida com a reconstrução da série de Fourier (Definir k da reconstrução - k_sel)
plt.figure(figsize=(12,4))
plt.plot(df_analise_consumo.index, serie_detrendida, label='Original (detrended)')
plt.plot(df_analise_consumo.index, serie_recons_fourier, label=f'Reconstrução k={k_sel}', linestyle='--')
plt.title(f'Reconstrução com k={k_sel} (Variância explicada = {variancia_arr[k_sel-1]:.3f})')
plt.grid()
plt.legend()
plt.show()

## Plot da série sem a tendência e sem a sazionalidade
plt.figure(figsize=(12,4))
plt.plot(serie_comp_irregular, label='Série - Componente Irregular')
plt.plot(serie_detrendida, label='Série - Componente irregular + Sazionalidade')
plt.title('Série limpa, sem tendência e sem sazionalidade')
plt.grid()
plt.legend()
plt.show()

## Plot dos dados de treino, validação e previsões dos modelos estatísticos e neurais

plt.figure(figsize=(12,6))
plt.plot(serie_treino['time_col'], serie_treino['target_col'], label='Treino', color='gray')
plt.plot(previsoes_combinadas['ds'], previsoes_combinadas['y'], label='Validação', color='k', linewidth=2)
cores = ['r', 'b', 'g']
for i, modelo in enumerate(melhores_modelos):
    plt.plot(previsoes_combinadas['ds'], previsoes_combinadas[modelo], color=cores[i], linewidth=0.5, linestyle='--', label=modelo)
for i, modelo in enumerate(melhores_modelos):
    plt.plot(previsoes_future['ds'], previsoes_future[modelo],label=modelo, color=cores[i], linestyle='--')
plt.xlim(left=datetime(2020,1,1), right=datetime(2027,1,1))
plt.legend()
plt.title('Previsões de consumo de energia - Modelos Estatísticos e Neurais')
plt.xlabel('Data')
plt.ylabel('Consumo detrendido (LOESS)')
plt.grid()
plt.tight_layout()
plt.show()

#CODIGO QUE INCLUI NOVOS INDICADORES TECNICOS
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import yfinance as yfin

def calcula_rsi(df, window):
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = abs(avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcula_macd(df, short=15, long=30, signal=12):
    short_ema = df.ewm(span=short, adjust=False).mean()
    long_ema = df.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def predicao_modelo(baseTotal, baseX_teste, predictions, stop_loss_pct=0.05, take_profit_pct=0.1):
    capital = 100000
    quantidadeAcoes = 0
    precoCompra = 0
    dataUltimaVenda = None

    for i, row in baseTotal.iterrows():
        if i not in baseX_teste.index:
            continue

        predicao = predictions[baseX_teste.index.get_loc(i)]
        preco_fechamento = row['Close'] if isinstance(row['Close'], (int, float)) else row['Close'].iloc[0]

        if predicao == 1 and quantidadeAcoes == 0:
            if capital >= preco_fechamento:
                precoCompra = preco_fechamento
                quantidadeAcoes = capital // precoCompra
                capital = 0
                print(f"Comprado {quantidadeAcoes} ações por R$ {precoCompra} no dia {i}")

        if quantidadeAcoes > 0 and (preco_fechamento <= precoCompra * (1 - stop_loss_pct)):
            precoVenda = preco_fechamento
            capital = quantidadeAcoes * precoVenda
            quantidadeAcoes = 0
            print(f"Stop-Loss ativado. Vendido por R$ {precoVenda} no dia {i}, capital atual: {capital}")

        if quantidadeAcoes > 0 and (preco_fechamento >= precoCompra * (1 + take_profit_pct)):
            precoVenda = preco_fechamento
            capital = quantidadeAcoes * precoVenda
            quantidadeAcoes = 0
            print(f"Take-Profit ativado. Vendido por R$ {precoVenda} no dia {i}, capital atual: {capital}")

        if predicao == -1 and quantidadeAcoes > 0:
            precoVenda = preco_fechamento
            capital = quantidadeAcoes * precoVenda
            quantidadeAcoes = 0
            print(f"Vendido por R$ {precoVenda} no dia {i}, capital atual: {capital}")

        if i.strftime('%m-%d') == '01-02' and quantidadeAcoes > 0:
            precoVenda = preco_fechamento
            capital = quantidadeAcoes * precoVenda
            quantidadeAcoes = 0
            dataUltimaVenda = i
            print(f"Venda final no fim do ano por R$ {precoVenda} no dia {dataUltimaVenda}, capital atualizado: {capital}")

    return capital, dataUltimaVenda


df = yfin.download("VALE3.SA", start="2019-01-02", end="2024-11-01")
# retorno com 14 ----> aproximadamente 125K
df["RSI"] = calcula_rsi(df, 14) # alterando o periodo de 14 para 7 ---> retorno menor: aproximadamente 121K # alterando o periodo de 14 para 21 ---> retorno menor: aproximadamente 123K
df["MACD"], df["Signal_Line"] = calcula_macd(df["Close"]) #mantendo o rsi com 14 e alterando para (short=9, long=18, signal=6) retorno de 123K
#mantendo o rsi com 14 e alterando para (short=15, long=30, signal=12) ----> retorno de 124K
#mesmo alterando o MACD e o RSI para diferentes combinacoes, o RSI com 14 ainda retorna o melhor valor
df["ROC"] = ((df["Close"] - df["Close"].shift(7) / df["Close"].shift(7)) * 100)
df["SMA_short"] = df["Close"].rolling(window=10).mean()# short
df["SMA_medium"] = df["Close"].rolling(window=50).mean()# medium
df["SMA_long"] = df["Close"].rolling(window=200).mean() #long
sma = df["Close"].rolling(window=20).mean()
std = df["Close"].rolling(window=20).std()
df["Bollinger_Upper"] = sma + (std * 2)
df["Bollinger_Lower"] = sma - (std * 2)
short_vol_ma = df["Volume"].rolling(window=5).mean()
long_vol_ma = df["Volume"].rolling(window=10).mean()
df["Volume_Oscillator"] = ((short_vol_ma - long_vol_ma) / long_vol_ma) * 100
df["Variacao_Preco"] = df['Close'].pct_change() * 100
df["Variacao_Preco"] = df["Variacao_Preco"].fillna(0)
df["Operacao"] = np.where(df["Variacao_Preco"] >= 0.1, 1, np.where(df["Variacao_Preco"] <= -0.19, -1, 0))

X = df[["Open", "Close", "High", "Low", "Volume", "RSI", "MACD", "Signal_Line", "SMA_short", "SMA_medium", "SMA_long", "Bollinger_Upper", "Bollinger_Lower", "Volume_Oscillator", "ROC"]].dropna()
y = df["Operacao"][X.index]

dados_treinamento = df[str(2019):str(2022)]
X_treino = dados_treinamento[["Open", "Close", "High", "Low", "Volume", "RSI", "MACD", "Signal_Line", "Variacao_Preco", "SMA_short", "SMA_medium", "SMA_long", "Bollinger_Upper", "Bollinger_Lower", "Volume_Oscillator", "ROC"]].dropna()
Y_treino = dados_treinamento["Operacao"][X_treino.index]

dados_simulacao = df[str(2023):str(2024)]
X_simulacao = dados_simulacao[["Open", "Close", "High", "Low", "Volume", "RSI", "MACD", "Signal_Line", "Variacao_Preco", "SMA_short", "SMA_medium", "SMA_long", "Bollinger_Upper", "Bollinger_Lower", "Volume_Oscillator", "ROC"]].dropna()


param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": [0.01, 0.1, 1]
}
model = SVC()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
scaler = StandardScaler()

X_treino_scaled = scaler.fit_transform(X_treino)
grid_search.fit(X_treino_scaled, Y_treino)

melhorModelo = grid_search.best_estimator_
melhores_parametros = grid_search.best_params_
print(f"Melhores parâmetros encontrados pelo GridSearch: {melhores_parametros}")


k = 10
kf = KFold(n_splits=k)

accuracy_scores = []
precision_scores = []
recall_scores = []

for train_index, test_index in kf.split(X_treino):
    X_train, X_test = X_treino.iloc[train_index], X_treino.iloc[test_index]
    y_train, y_test = Y_treino.iloc[train_index], Y_treino.iloc[test_index]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(C=10, gamma=0.01, kernel='linear')
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy_scores.append(accuracy_score(y_test, predictions))
    precision_scores.append(precision_score(y_test, predictions, average="macro", zero_division=0))
    recall_scores.append(recall_score(y_test, predictions, average="macro", zero_division=0))

mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)

print(f"\nMétricas de validação cruzada (K-Fold):")
print(f"Acurácia média: {mean_accuracy:.2f}")
print(f"Precisão média: {mean_precision:.2f}")
print(f"Recall médio: {mean_recall:.2f}")


X_simulacao_scaled = scaler.transform(X_simulacao)
predictions = melhorModelo.predict(X_simulacao_scaled)

capital, dataVendaFinal = predicao_modelo(df, X_simulacao, predictions)

print(f"Capital final: R$ {capital}; Data da última transação: {dataVendaFinal}")

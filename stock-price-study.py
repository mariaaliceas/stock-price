#CODIGO ABAIXO COM AJUSTES: 

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import yfinance as yfin
from sklearn.metrics import accuracy_score, precision_score, recall_score


def calcula_rsi(df, window):
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = abs(avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calcula_macd(df, short=12, long=26, signal=9):
    short_ema = df.ewm(span=short, adjust=False).mean()
    long_ema = df.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def predicao_modelo(baseTotal, baseX_teste, predictions):

    capital = 100000
    quantidadeAcoes = 0
    fimDeAno = False
    dataUltimaVenda = None
    dataVendaFinal = None

    for i, row in baseTotal.iterrows():
        if i not in baseX_teste.index:
            continue

        if i == pd.Timestamp("2024-01-02 00:00:00+00:00"):
            fimDeAno = True

        preco_fechamento = (
            row["Close"]
            if isinstance(row["Close"], (int, float))
            else row["Close"].iloc[0]
        )

        predicao = predictions[baseX_teste.index.get_loc(i)]

        if fimDeAno:
            if predicao == -1 and quantidadeAcoes > 0:
                precoVenda = preco_fechamento
                capital += quantidadeAcoes * precoVenda
                quantidadeAcoes = 0
                dataVendaFinal = i
                break

            elif predicao == 1 and quantidadeAcoes == 0:
                dataVendaFinal = dataUltimaVenda
                break

        else:
            if predicao == 1 and quantidadeAcoes == 0:
                if capital >= preco_fechamento:
                    quantidadeAcoes += capital // preco_fechamento
                    capital %= preco_fechamento
                    print(
                        f"Compra: {quantidadeAcoes} ações a R$ {preco_fechamento} no dia {i} e predicao {predicao}"
                    )

            elif predicao == -1 and quantidadeAcoes > 0:
                capital += quantidadeAcoes * preco_fechamento
                print(
                    f"Venda: {quantidadeAcoes} ações a R$ {preco_fechamento} no dia {i} e {predicao}, VENDA: {capital}"
                )
                quantidadeAcoes = 0
                dataUltimaVenda = i

    return capital, dataVendaFinal


df = yfin.download("VALE3.SA", start="2019-01-02", end="2024-11-01")

df["RSI"] = calcula_rsi(df, 14)
df["MACD"], df["Signal_Line"] = calcula_macd(df["Close"])

df["Variacao_Preco"] = df["Close"] - df["Close"].shift(1)
df["Variacao_Preco"] = df["Variacao_Preco"].fillna(0)
df["Operacao"] = np.where(
    df["Variacao_Preco"] >= 0.40, 1, np.where(df["Variacao_Preco"] <= -0.40, -1, 0)
)

X = df[
    ["Open", "Close", "High", "Low", "Volume", "RSI", "MACD", "Signal_Line"]
].dropna()
y = df["Operacao"][X.index]

# selecionando dados para o treinamento
dados_treinamento = df[str(2019) : str(2022)]
X_treino = dados_treinamento[
    [
        "Open",
        "Close",
        "High",
        "Low",
        "Volume",
        "RSI",
        "MACD",
        "Signal_Line",
        "Variacao_Preco",
    ]
].dropna()
Y_treino = dados_treinamento["Operacao"][X_treino.index]

# selecionando dados para o teste
dados_simulacao = df[str(2023) : str(2024)]
X_simulacao = dados_simulacao[
    [
        "Open",
        "Close",
        "High",
        "Low",
        "Volume",
        "RSI",
        "MACD",
        "Signal_Line",
        "Variacao_Preco",
    ]
].dropna()

k = 10
kf = KFold(n_splits=k)

scaler = StandardScaler()

accuracy_scores = []
precision_scores = []
recall_scores = []

bestScaler = None
melhorModelo = None
melhorAcuracia = 0

for train_index, test_index in kf.split(X_treino):
    X_train, X_test = X_treino.iloc[train_index], X_treino.iloc[test_index]
    y_train, y_test = Y_treino.iloc[train_index], Y_treino.iloc[test_index]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="linear")
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracyAtual = accuracy_score(y_test, predictions)
    precisionAtual = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recallAtual = recall_score(y_test, predictions, average="macro", zero_division=0)

    accuracy_scores.append(accuracyAtual)
    precision_scores.append(precisionAtual)
    recall_scores.append(recallAtual)

    if accuracyAtual >= melhorAcuracia:
        melhorAcuracia = accuracyAtual
        melhorModelo = model
        bestScaler = scaler
    accuracy_scores.append(accuracyAtual)


mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)

print(f"\nMétricas de validação cruzada (K-Fold):")
print(f"Melhor acurácia: {melhorAcuracia:.2f}")
print(f"Acurácia média: {mean_accuracy:.2f}")
print(f"Precisão média: {mean_precision:.2f}")
print(f"Recall médio: {mean_recall:.2f}")

X_simulacao_scaled = scaler.transform(X_simulacao)
prediction = melhorModelo.predict(X_simulacao_scaled)

capital, dataVendaFinal = predicao_modelo(df, X_simulacao, prediction)

print(f"Capital final: R$ {capital}; Data da última transação: {dataVendaFinal}")

capital_df = pd.DataFrame(capital_history, index=[2019, 2020, 2021, 2022], columns=['Capital'])

plt.figure(figsize=(14, 7))
plt.plot(capital_df, marker='o', label='Capital Acumulado')
plt.xlabel('Ano')
plt.ylabel('Capital (R$)')
plt.title('Simulação de Capital Acumulado')
plt.xticks(capital_df.index)  # Marcar os anos no eixo x
plt.legend()
plt.grid()
plt.show()

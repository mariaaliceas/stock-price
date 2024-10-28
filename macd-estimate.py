def preve_macd(dados):

    capital = 100000
    quantidadeAcoes = 0

    for i, row in dados.iterrows():
        macd = row['MACD'] if isinstance(row['MACD'], (int, float)) else row['MACD'].iloc[0]
        signalLine = row['Signal_Line'] if isinstance(row['Signal_Line'], (int, float)) else row['Signal_Line'].iloc[0]

        if macd > signalLine and capital > 0:
            precoCompra = row['Close'] if isinstance(row['Close'], (int, float)) else row['Close'].iloc[0]
            quantidadeAcoes = capital // precoCompra
            capital = 0
            print(f"Comprado {quantidadeAcoes} ações por R$ {precoCompra} no dia {i}")

        elif macd < signalLine and capital == 0:
            precoVenda = row['Close'] if isinstance(row['Close'], (int, float)) else row['Close'].iloc[0]
            capital = quantidadeAcoes * precoVenda
            quantidadeAcoes = 0
            print(f"Vendido por R$ {precoVenda} no dia {i}, capital atual: {capital}")

        if i.strftime('%m-%d') == '01-02' and capital == 0:
            precoVenda = row['Close'] if isinstance(row['Close'], (int, float)) else row['Close'].iloc[0]
            capital = quantidadeAcoes * precoVenda
            quantidadeAcoes = 0
            print(f"Venda final no fim do ano por R$ {precoVenda} no dia {i}, capital atualizado: {capital}")

    return capital, i

df = pd.DataFrame()

start_date = '2019-01-01'
end_date = '2024-01-03'

tck_vale = ['VALE3.SA']

vale_data = yfin.download(tck_vale, start=start_date, end=end_date)
vale_data['RSI'] = calculate_rsi(vale_data, 14)

vale_data['EMA12'] = vale_data['Close'].ewm(span=12, adjust=False).mean()
vale_data['EMA26'] = vale_data['Close'].ewm(span=26, adjust=False).mean()

vale_data['MACD'] = vale_data['EMA12'].sub(vale_data['EMA26'])
vale_data['Signal_Line'] = vale_data['MACD'].ewm(span=9, adjust=False).mean()

capital, i = preve_macd(vale_data)

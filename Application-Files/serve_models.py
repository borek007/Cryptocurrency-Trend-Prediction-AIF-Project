import rpyc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras
from rpyc.utils.server import ThreadedServer
import threading
import time
import os

PERIODS_BACK = 20

class ServerService(rpyc.Service):
    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_get_prediction(self, crypto, number_of_days):
        if crypto == "btc":
            if(number_of_days in range(0,8)):
                #X_set = LoadBtc(df, PERIODS_BACK, 7)
                function_name = f"MakePrediction(model_7_btc, df_btc, X_7_btc, {PERIODS_BACK}, scaler)"
                result = eval(function_name)
                return result[0:number_of_days]
                #return result
            else:
                return "WRONG_NUMBER_OF_DAYS"
        if crypto == "doge":
            if(number_of_days in range(0,8)):
                #X_set = LoadBtc(df, PERIODS_BACK, 7)
                function_name = f"MakePrediction(model_7_doge, df_doge, X_7_doge, {PERIODS_BACK}, scaler)"
                result = eval(function_name)
                return result[0:number_of_days]
                #return result
            else:
                return "WRONG_NUMBER_OF_DAYS"
        if crypto == "eth":
            if(number_of_days in range(0,8)):
                #X_set = LoadBtc(df, PERIODS_BACK, 7)
                function_name = f"MakePrediction(model_7_eth, df_eth, X_7_eth, {PERIODS_BACK}, scaler)"
                result = eval(function_name)
                return result[0:number_of_days]
                #return result
            else:
                return "WRONG_NUMBER_OF_DAYS"
        else:
            return "WRONG_CRYPTO_CURRENCY_NAME"

    def exposed_test(self):
        return "Hello world!"

def StartServer():
    t = rpyc.ThreadedServer(ServerService, port = 12345)
    t.start()

def SplitSequence(seq, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out
        if out_end > len(seq):
            break
        seq_x, seq_y = seq[i:end], seq[end:out_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def LoadDataFromCsv(csv_file):
    return pd.read_csv(csv_file)

def LoadCryptoData(scaler, data_name):
    df = LoadDataFromCsv(data_name)
    df = df.set_index("Date")[['Close']].tail(1000)
    df = df.set_index(pd.to_datetime(df.index))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df

def LoadDogeData(scaler):
    return LoadCryptoData(scaler, "DOGE-USD.csv")

def LoadBtcData(scaler):
    # df = LoadDataFromCsv("BTC-USD.csv")
    # #df = pd.read_csv("BTC-USD.csv")
    # df = df.set_index("Date")[['Close']].tail(1000)
    # df = df.set_index(pd.to_datetime(df.index))
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    # return df
    return LoadCryptoData(scaler, "BTC-USD.csv")

def LoadEthData(scaler):
    return LoadCryptoData(scaler, "ETH-USD.csv")

def LoadBtc(df, periods_back, periods_future):
    X, y = SplitSequence(list(df.Close), periods_back, periods_future)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X

def LoadDoge(df, periods_back, periods_future):
    X, y = SplitSequence(list(df.Close), periods_back, periods_future)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X

def LoadEth(df, periods_back, periods_future):
    X, y = SplitSequence(list(df.Close), periods_back, periods_future)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X

def LoadBtc1Days(df):
    return LoadBtc(df, 1, PERIODS_BACK)

def LoadBtc3Days(df):
    return LoadBtc(df, 3, PERIODS_BACK)

def LoadBtc5Days(df):
    return LoadBtc(df, 5, PERIODS_BACK)

def LoadBtc7Days(df):
    return LoadBtc(df, 7, PERIODS_BACK)

def LoadDoge7Days(df):
    return LoadDoge(df, 7, PERIODS_BACK)

def LoadEth7Days(df):
    return LoadEth(df, 7, PERIODS_BACK)

def ImportModel(crypto, days_back):
    return tensorflow.keras.models.load_model(f'{crypto}-usd-lstm-{days_back}-days')

def ImportModel1Days():
    return ImportModel(1)

def ImportModel3Days():
    return ImportModel(3)

def ImportModel5Days():
    return ImportModel(5)

def ImportModel7Days(crypto):
    return ImportModel(crypto, 7)

def MakePrediction(model, btc_data_all, btc_data, periods_back, scaler):
    # yhat = model.predict(np.array(df.tail(periods_back)).reshape(1, periods_back, 1)).tolist()[0]
    yhat = model.predict(np.array(btc_data_all.tail(periods_back)).reshape(1, periods_back, 1)).tolist()[0]
    yhat = scaler.inverse_transform(np.array(yhat).reshape(-1,1)).tolist()
    preds = pd.DataFrame(yhat, index=pd.date_range(start=btc_data_all.index[-1], periods=len(yhat), freq="D"), columns=btc_data_all.columns)
    pers = 10
    actual = pd.DataFrame(scaler.inverse_transform(btc_data_all[["Close"]].tail(pers)), index=btc_data_all.Close.tail(pers).index, columns=btc_data_all.columns).append(preds.head(1))
    cnt = 0 
    preds_tend = []
    last_btc_price = GetLastBtcPrice(btc_data_all)
    #return preds
    for i in range(0, preds.shape[0]):
        if i == 0:
            if preds.iloc[0,0] < last_btc_price: preds_tend.append("down")
            else: preds_tend.append("up")
        else:
            if preds.iloc[i,0] < preds.iloc[i-1, 0]: preds_tend.append("down")
            else: preds_tend.append("up")
    #errors = model.evaluate()
    #return preds
    return preds_tend

def GetLastBtcPrice(df):
    return 33560.00
    #return df.tail(1)["Close"]

scaler = None
df_btc = None
df_doge = None
df_eth = None

X_7_btc = None
X_7_doge = None
X_7_eth = None

model_7_btc = None
model_7_doge = None
model_7_eth = None

if __name__ == "__main__":
    scaler = MinMaxScaler()
    df_btc = LoadBtcData(scaler)
    df_doge = LoadDogeData(scaler)
    df_eth = LoadEthData(scaler)

    X_7_btc = LoadBtc7Days(df_btc)
    X_7_doge = LoadDoge7Days(df_doge)
    X_7_eth = LoadEth7Days(df_eth)
    
    model_7_btc = ImportModel7Days("btc")
    print("BTC MODEL LOADED")
    model_7_doge = ImportModel7Days("doge")
    print("DOGE MODEL LOADED")
    model_7_eth = ImportModel7Days("eth")
    print("ETH MODEL LOADED")

    btc_f = False
    doge_f = False
    eth_f = False

    if model_7_btc != None and model_7_eth != None and model_7_doge != None:
        print("ALL MODELS LOADED")
        print("STARTING PREDICTION SERVER")
        thread = threading.Thread(target = StartServer, daemon = True)
        thread.start()
        print("PREDICTION SERVER STARTED")
        print("STARTING BACKEND SERVER")
        time.sleep(15)
        os.system('python app.py')
    else:
        print("ERROR LOADING MODELS!")

    #server = ThreadedServer(ServerService, port=12345)
    #server.start()

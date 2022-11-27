from flask import Flask, render_template, json, redirect
import requests
import datetime
import rpyc

app = Flask(__name__)
conn = rpyc.connect('127.0.0.1', 12345)  # connect

# routes


@app.route("/")
def home():
    return redirect("/bitcoin")


@app.route("/<currency>")
def currency(currency):
    now = datetime.datetime.now()
    then = now - datetime.timedelta(days=1)
    unix_then = int(then.timestamp())
    unix_now = int(now.timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{currency}/market_chart/range?vs_currency=usd&from={unix_then}&to={unix_now}"
    response = requests.request("GET", url)
    prices = json.loads(response.text)["prices"][::-1]
    data = []
    date = now.strftime("%d/%m/%y %H:%M")
    for p in prices:
        data.append({"value": p[1], "time": now.strftime("%H:%M")})
        now -= datetime.timedelta(minutes=5)
    price = round(data[0]["value"], 2)
    data = data[::-1]
    #pred = []
    #pred.append(conn.root.get_prediction(f'{currency}', 1))
    #pred.append(conn.root.get_prediction(f'{currency}', 3))
    #pred.append(conn.root.get_prediction(f'{currency}', 5))
    # pred.append(conn.root.get_prediction(7)[0])
    #pred.append("up")
    
    print(currency)
    
    currency_name = ''
    if currency == 'dogecoin':
        currency_name = 'doge'
    elif currency == 'ethereum':
        currency_name = 'eth'
    else:
        currency_name = 'btc'
        
    pred = conn.root.get_prediction(f'{currency_name}', 7)
    
    print('d')
    print(pred)
    
    return render_template("home.html", data=data, price=price, currency_name=currency_name, currency=currency.capitalize(), date=date, pred=pred)


# @app.route("/test_set")
# def test_set():
#     now = datetime.datetime.now()
#     for i in range(2):
#         now = now - datetime.timedelta(days=1)
#         then = now - datetime.timedelta(days=1) + datetime.timedelta(minutes=5)
#         print(then)
#         print(now)
#         unix_then = int(then.timestamp())
#         unix_now = int(now.timestamp())
#         url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={unix_then}&to={unix_now}"
#         response = requests.request("GET", url)
#         prices = json.loads(response.text)["prices"]
#         data = [p[1] for p in prices]
#         print(len(data))
#     return "1"


if __name__ == "__main__":
    app.run(debug=True)

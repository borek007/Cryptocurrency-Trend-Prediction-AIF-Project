import rpyc

conn = rpyc.connect('localhost', 12345) # connect
print(conn.root.get_prediction("doge", 5))
print(conn.root.get_prediction("btc", 7))
print(conn.root.get_prediction("eth", 1))

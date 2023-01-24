from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class Gateway(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
class Bot_logic:
    ib = None
    def __init__(self):
        ib = Gateway()
        ib.connect("127.0.0.1", 7496, 1)
        ib.run()

bot = Bot_logic
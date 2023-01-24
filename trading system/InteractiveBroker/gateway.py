from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class Gateway(EWrapper, EClient):
    def __init__(self, addr: str, port: int, client_id: int, user: str, password: str):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.user = user
        self.password = password
        self.connect(addr, port, client_id)
        
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.reqAccountUpdates(True, self.user)

gateway = Gateway("127.0.0.1", 7497, 0, "user_name", "password")
gateway.run()

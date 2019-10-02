class Instance():

    def __init__(self , history , reply):
        self.history = history
        self.reply = reply

    def set_history(self,history):
        self.history = history

    def get_history(self):
        return self.history

    def set_reply(self , question):
        self.reply= question

    def get_reply(self):
        return self.reply

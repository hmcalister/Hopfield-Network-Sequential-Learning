fns = [
    lambda x: x+1,
    lambda x: x+2,
    lambda x: x+3,
    lambda x: x+4,
]

def get_loss_function():
    def loss_function(x):
        loss = 0
        for f in fns:
            loss+=f(x)
        return loss
    return loss_function
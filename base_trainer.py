class BaseTrainer:
    def __init__(self, args, model, train_loader, val_loader, criterion, regularizer):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.regularizer = regularizer


    def train(self):
        pass


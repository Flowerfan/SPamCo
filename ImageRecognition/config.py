class Config(object):

    # logs dir
    logs_dir = 'logs'
    # model training parameters
    workers = 8
    dropout = 0.5
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    step_size = 20
    sampler = None
    print_freq = 40
    padding = 4

    def __init__(self,
                 model_name='resnet18',
                 loss_name='softmax',
                 num_classes=10,
                 height=32,
                 width=32,
                 batch_size=128,
                 epochs=50,
                 checkpoint=None,
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010],
                 early_transform=['rc', 'rf'],
                 later_transform=['re']):
        self.model_name = model_name
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.early_transform = early_transform
        self.later_transform = later_transform
        self.mean = mean
        self.std = std

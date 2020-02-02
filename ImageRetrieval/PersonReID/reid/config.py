from reid.utils.data.sampler import RandomIdentitySampler

class Config(object):

    # logs dir
    logs_dir = 'logs'
    # model training parameters
    workers = 4
    num_features = 512
    dropout = 0.5
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    sampler = None
    print_freq = 40

    # training flag
    training = True
    shuffle = True

    # distance metric
    dist_metric = 'euclidean'

    def __init__(self, model_name='resnet50', loss_name='softmax',
                 num_classes=751,height=256, width=128, batch_size=32,
                 epochs=50,num_features=512, checkpoint=None,
                 img_translation=None):
        self.model_name = model_name
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.epochs = epochs
        self.num_features = num_features
        self.checkpoint = checkpoint
        self.img_translation = img_translation


    def set_training(self, state):
        self.training = state
        self.shuffle = state




class TripletConfig(Config):



    loss_name = 'triplet'
    # quantity of each identity in one training batch
    num_instances = 4
    num_classes = 128
    num_features = 1024
    epochs = 150
    dropout = 0

    # margin of triplet loss
    margin = 0.5
    shuffle = False

    lr = 0.0002


    if Config.training:
        sampler = RandomIdentitySampler

    def set_training(self, state):
        self.training = state

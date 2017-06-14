import chainer
import chainer.functions as F
import chainer.links as L

class Q(chainer.Chain):
    def __init__(self, width=150, height=112, channel=3, action_size=100, latent_size=100):
        feature_width = width
        feature_height = height
        for i in range(4):
            feature_width = (feature_width + 1) // 2
            feature_height = (feature_height + 1) // 2
        feature_size = feature_width * feature_height * 64
        super(Q, self).__init__(
            conv1 = L.Convolution2D(channel, 16, 8, stride=4, pad=3),
            conv2 = L.Convolution2D(16, 32, 5, stride=2, pad=2),
            conv3 = L.Convolution2D(32, 64, 5, stride=2, pad=2),
            lstm  = L.LSTM(feature_size, latent_size),
            q     = L.Linear(latent_size, action_size),
        )
        self.width = width
        self.height = height
        self.latent_size = latent_size

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = self.lstm(h3)
        q = self.q(h4)
        return q

    def reset_state(self):
        self.lstm.reset_state()

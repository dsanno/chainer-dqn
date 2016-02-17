import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class Q(chainer.Chain):
    def __init__(self, width=150, height=112, channel=3, action_size=100, latent_size=100):
        feature_size = (((width / 2 + 1) / 2 + 1) / 2) * (((height / 2 + 1) / 2 + 1) / 2) * 32
        super(Q, self).__init__(
            conv1  = L.Convolution2D(channel, 16, 8, stride=2, pad=3, wscale=0.02 * math.sqrt(channel)),
            norm1  = L.BatchNormalization(16),
            conv2  = L.Convolution2D(16, 32, 5, pad=2, wscale=0.02 * math.sqrt(16)),
            norm2  = L.BatchNormalization(32),
            l1 = L.Linear(feature_size, latent_size, wscale=0.02 * math.sqrt(feature_size)),
            norml1 = L.BatchNormalization(latent_size),
            l2 = L.Linear(latent_size, latent_size, wscale=0.02 * math.sqrt(latent_size)),
            norml2 = L.BatchNormalization(latent_size),
            q      = L.Linear(latent_size, action_size, wscale=0.02 * math.sqrt(latent_size)),
        )
        self.width = width
        self.height = height
        self.latent_size = latent_size

    def __call__(self, (x, prev), train=True):
        h1 = F.max_pooling_2d(F.relu(self.norm1(self.conv1(x), test=not train)), 2)
        h2 = F.max_pooling_2d(F.relu(self.norm2(self.conv2(h1), test=not train)), 2)
        h3 = F.relu(self.norml1(self.l1(h2), test=not train) + prev)
        current = self.l2(h3)
        h4 = F.relu(self.norml2(current, test=not train))
        q = self.q(h4)
        return (q, current)

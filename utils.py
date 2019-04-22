import numpy as np

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def decimate_ts(ts, ds):
    win = np.ones(ds) / float(ds)
    return np.convolve(ts, win, mode='same')[::ds]

def decimate(y, ds):
    if ds > 1:
        y = np.array(y)
        if y.ndim == 1:
            return decimate_ts(list(y), ds)
        else:
            newy_transpose = []
            for i in range(y.ndim):
                newy_transpose.append(decimate_ts(list(y[:,i]), ds))
            return [list(x) for x in zip(*newy_transpose)]
    else:
        return y

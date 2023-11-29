import argparse
from collections import Counter, defaultdict
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch import log_softmax
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from models.vgg import VGG
from models.resnet import *
from models.mobilenet import *
from models.layers import Conv2dRecorded
from models.layers import ReLURecorded

parser = argparse.ArgumentParser()
parser.add_argument('--note', default="", type=str, help='Note of the experiment')
parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument('--tune', default=False, action="store_true")
parser.add_argument('--skip_plot', default=False, action="store_true")
parser.add_argument('--silent', default=False, action="store_true")
parser.add_argument('--device', default="cpu", type=str)
args = parser.parse_args()

config = {
  'dataset': 'cifar10',
  # 'dataset': 'mnist',
  # 'n_hiddens': [256, 256, 256],
  'n_hiddens': [500, 500, 500],
  'img_size': 28,
  'need_bias': False,
  'batch_size': 2048,
  'workers': 0,
  'epochs': 300,
  'top_k': 1,
  'one_hot_threshold': 0.5,
  'local': False,
  'mlp': True,
  'sim_all_layers': True,

  # Best mnist results
  # 'sim_w': 2,  # Need 2X for mnist
  # 'one_hot_w': [10000, 0, 10000],  # Need 20X

  'xent_w': 1,
  'sim_w': 0.03,  # Need 2X for mnist
  'one_hot_w': 1,  # Need 20X

  'predictive_w': 0,
  'hebbian_w': 0,
  'kernel_size': 3,
  'random_baseline': False,
  'output_dir': "image_conv_experiments/%s_%s" % (args.note, time.strftime("%Y_%m_%d_%H_%M_%S"))
}

config['n_layers'] = len(config['n_hiddens'])
if config['dataset'] == "mnist":
  if config['mlp']:
    config['n_hiddens'] =  [config['img_size'] ** 2] + config['n_hiddens']
  else:
    config['n_hiddens'] = [1] + config['n_hiddens']

elif config['dataset'] == "cifar10":
  config['n_hiddens'] = [3] + config['n_hiddens']
  config['img_size'] = 32
  config['batch_size'] = 128
  config['mlp'] = False
elif config['dataset'] == "mnist":
  config['epochs'] = 100
if args.debug:
  config['epochs'] = 20
os.makedirs(config['output_dir'], exist_ok=True)

class LocallyConnected2d(torch.nn.Module):
  def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=False):
    super(LocallyConnected2d, self).__init__()
    output_size = _pair(output_size)
    self.weight = torch.nn.Parameter(
      torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2) * 0.3
    )
    if bias:
      self.bias = torch.nn.Parameter(
        torch.randn(1, out_channels, output_size[0], output_size[1])
      )
    else:
      self.register_parameter('bias', None)
    # torch.nn.init.xavier_normal_(self.weight)
    # torch.nn.init.normal_(self.weight)
    # torch.nn.init.sparse_(self.weight, sparsity=0.3)
    self.kernel_size = _pair(kernel_size)
    self.stride = _pair(stride)

  def forward(self, x):
    _, c, h, w = x.size()
    kh, kw = self.kernel_size
    dh, dw = self.stride
    x = x.unfold(2, kh, dh).unfold(3, kw, dw)
    x = x.contiguous().view(*x.size()[:-2], -1)
    # Sum in in_channel and kernel_size dims
    out = (x.unsqueeze(1) * self.weight).sum([2, -1])
    if self.bias is not None:
      out += self.bias
    return out


transform = transforms.Compose([
  # transforms.RandomCrop(config['img_size']),
  # transforms.functional.crop(10, 10, config['img_size'], config['img_size']),
  # transforms.CenterCrop(config['img_size']),
  transforms.Resize(config['img_size']),
  transforms.ToTensor(),
])

cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
  transforms.Resize(config['img_size']),
  transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
  transforms.Resize(config['img_size']),
  transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_dir = '~/data'
if config['dataset'] == "mnist":
  dataset_train = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
  dataset_test = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
elif config['dataset'] == "cifar10":
  dataset_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=cifar_transform_train, download=True)
  dataset_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=cifar_transform_test, download=True)

train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'],
                          drop_last=True)
memory_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'],
                           drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False,
                         num_workers=config['workers'])

# one_hot_loss = torch.nn.CrossEntropyLoss()
one_hot_loss = torch.nn.MSELoss()
cosine_loss = torch.nn.CosineSimilarity(1)
xentropy_loss = torch.nn.CrossEntropyLoss()
n_classes = len(dataset_train.classes)

class Model(torch.nn.Module):
  def __init__(self, config):
    super(Model, self).__init__()
    self.config = config
    self.feed_forward = torch.nn.ModuleList()
    image_size = config['img_size']
    self.relus = torch.nn.ModuleList()
    for l in range(config['n_layers']):
      self.relus.append(ReLURecorded())
      if config['mlp']:
        self.feed_forward.append(
          torch.nn.Linear(config['n_hiddens'][l], config['n_hiddens'][l + 1], bias=config['need_bias']))
      else:
        if config['local']:
          self.feed_forward.append(LocallyConnected2d(config['n_hiddens'][l], config['n_hiddens'][l + 1],
                                                      output_size=image_size, stride=1,
                                                      kernel_size=config['kernel_size'], bias=config['need_bias']))
        else:
          self.feed_forward.append(torch.nn.Conv2d(config['n_hiddens'][l], config['n_hiddens'][l + 1], padding=0,
                                                   kernel_size=config['kernel_size'], bias=config['need_bias']))
      image_size //= 2

    self.padding = torch.nn.ZeroPad2d(1)
    self.pooling = torch.nn.MaxPool2d(2)
    if config['mlp']:
      self.readout = torch.nn.Linear(config['n_hiddens'][-1], n_classes, bias=config['need_bias'])
    else:
      self.readout = torch.nn.Linear(config['n_hiddens'][-1] * image_size ** 2, n_classes,
                                   bias=config['need_bias'])

  def forward(self, x):
    pre_pooled = []
    features = []
    if config['mlp']:
      x = x.view(x.shape[0], -1)
    for ff, relu in zip(self.feed_forward, self.relus):
      if config['mlp']:
        x = relu(ff(x))
        pre_pooled.append(x)
        features.append(x)
      else:
        x = self.padding(x)
        x = relu(ff(x))
        pre_pooled.append(x)
        x = self.pooling(x)
        features.append(x)
      # x.detach()
    readout = self.readout(x.view(len(x), -1))
    readout = log_softmax(readout, dim=-1)
    return readout, features, pre_pooled

  def measure_speed(self, x, sparse=True):
    inference_time = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for ff in  self.feed_forward:
      if sparse:
        x = x.to_sparse_csr()
      starter.record()
      x = x @ ff.weight.T
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      inference_time += curr_time
      x = torch.nn.functional.relu(x)
    return inference_time


if config['dataset'] == "mnist":
  model = Model(config)
else:
  # model = Model(config)
  # model = VGG("VGG19")
  model = ResNet152()
  # model = MobileNet()
model.to(args.device)

def get_features(model):
  features = []
  for layer in model.children():
    if len([l for l in layer.children()]):
      features += get_features(layer)
    if hasattr(layer, "activity") and layer.activity is not None:
      features.append(layer.activity)
  return features

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
annealing_tmax = int(config['epochs'] * 0.95)  # we want more time in the low lr region
print("annealing_tmax:", annealing_tmax)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
# scheduler = None
# scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], verbose=True)
scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True)

def hebbian_loss(a, stale_var=None, stale_mean=None, epsilon=1e-6):
  if stale_var is None:
    a_center = a.mean(dim=0).detach()
    variance = ((a - a_center) ** 2).sum(dim=0) / (a.shape[0] - 1)
    loss = -torch.log(variance + epsilon).mean()
  else:
    loss = -(1.0 / (stale_var + epsilon) * (a - stale_mean) ** 2).sum(dim=0).mean() / (a.shape[0] - 1)
  return loss


def vis_features(features, labels, name=""):
  """Visualize the image patches"""

  fig_size = 2
  nrows = 10
  ncols = 20
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * fig_size, nrows * fig_size))
  for c in range(n_classes):
    for i, im in enumerate(features[labels == c]):
      if i >= ncols:
        break
      axes[c, i].imshow(im[0], cmap='gray', vmin=0, vmax=1)
      axes[c, i].axis('off')

  plt.tight_layout()
  plt.suptitle("%s" % name)
  fig.subplots_adjust(top=0.8)
  plt.savefig(os.path.join(config["output_dir"], "%s.png" % name))


def get_sparsity():
  img, label = next(iter(memory_loader))
  img = img.to(args.device)
  readout, features, pre_pooled = model(img)
  pre_pooled = get_features(model)
  all_dead_channels = []
  for f, pre_pool in enumerate(pre_pooled):
    active = (pre_pool > 0).float()
    sparsity = 1 - active.mean()
    dead_pixels = (active.sum(0) == 0).float().mean()
    if len(active.shape)==4:
      dead_channels = (active.sum(dim=[0, 2, 3]) == 0).float()
    else:
      dead_channels = (active.sum(dim=0) == 0).float()
    all_dead_channels.append(dead_channels.detach())
    dead_channels = dead_channels.mean()
    print("Layer %i, sparsity: %.4f, dead pixels: %.4f, dead channels: %.4f" % (f, sparsity, dead_pixels, dead_channels))
  pre_pooled_flatten = torch.concat([p.flatten().detach().cpu() for p in pre_pooled], -1)
  overall_sparsity = (pre_pooled_flatten == 0).float().mean()
  all_dead_channels = torch.concat(all_dead_channels).mean()
  print("Overall sparsity: %.4f, overall dead channels: %.4f" % (overall_sparsity, all_dead_channels))


def inference_speed():
  model.eval()
  cpu_sparse = []
  cpu_dense = []
  gpu_sparse = []
  gpu_dense = []

  with torch.no_grad():
    for img, label in memory_loader:
      img = img.view(img.shape[0], -1)
      model.to("cpu")
      cpu_sparse.append(model.measure_speed(img, sparse=True))
      cpu_dense.append(model.measure_speed(img, sparse=False))
      model.to("cuda")
      img = img.to("cuda")
      gpu_sparse.append(model.measure_speed(img, sparse=True))
      gpu_dense.append(model.measure_speed(img, sparse=False))
  cpu_sparse = np.asarray(cpu_sparse).mean()
  cpu_dense = np.asarray(cpu_dense).mean()
  gpu_sparse = np.asarray(gpu_sparse).mean()
  gpu_dense = np.asarray(gpu_dense).mean()

  print("cpu_sparse: %.2f, cpu_dense: %.2f, gpu_sparse: %.2f, gpu_dense: %.2f" % (cpu_sparse, cpu_dense, gpu_sparse, gpu_dense))


def evaluate(name="", loss="l2"):
  if loss == "l2":
    sim_loss = torch.nn.PairwiseDistance()
  if loss == "cosine":
    sim_loss = torch.nn.CosineSimilarity()
  xentropy_loss = torch.nn.CrossEntropyLoss()

  all_class_vectors = []
  readout_layers = []
  readout_optimizers = []

  img, label = next(iter(memory_loader))
  img = img.to(args.device)
  readout, features, pre_pooled = model(img)
  if config['sim_all_layers']:
    features = get_features(model)
    pre_pooled = features
  else:
    pre_pooled = get_features(model)
  features = [img] + features
  pre_pooled = [img] + pre_pooled
  n_layers = len(features)
  evaluate_epochs = 10
  if args.debug:
    evaluate_epochs = 1
  for f, (feature, pre_pool) in enumerate(zip(features, pre_pooled)):
    feature_flatten = feature.view([len(feature), -1]).detach()
    readout_layers.append(torch.nn.Linear(feature_flatten.shape[-1], n_classes).to(args.device))
    readout_optimizers.append(torch.optim.Adam(readout_layers[-1].parameters(), lr=5e-2))
    all_class_vectors.append([])

  for e in range(evaluate_epochs):
    for img, label in memory_loader:
      img = img.to(args.device)
      label = label.to(args.device)
      readout, features, pre_pooled = model(img)
      if config['sim_all_layers']:
        features = get_features(model)
        pre_pooled = features
      else:
        pre_pooled = get_features(model)
      features = [img] + features
      n_layers = len(features)

      for f, feature in enumerate(features):
        feature_flatten = feature.view([len(feature), -1]).detach()
        readout = log_softmax(readout_layers[f](feature_flatten), dim=1)
        loss = xentropy_loss(readout, label)
        readout_optimizers[f].zero_grad()
        loss.backward()
        readout_optimizers[f].step()

        if e == 0:
          class_vectors = torch.zeros((n_classes, feature_flatten.shape[1])).to(args.device)
          for c in range(n_classes):
            class_vectors[c] = feature_flatten[label == c].mean(0).detach()
          all_class_vectors[f].append(class_vectors.detach().cpu())

  similarity = []
  readout_predicts = []
  all_labels = []
  for f in range(n_layers):
    similarity.append([])
    readout_predicts.append([])
    all_class_vectors[f] = torch.stack(all_class_vectors[f], dim=-1).mean(-1).to(args.device)

  for img, label in test_loader:
    img = img.to(args.device)
    label = label.to(args.device)
    readout, features, pre_pooled = model(img)
    if config['sim_all_layers']:
      features = get_features(model)
      pre_pooled = features
    else:
      pre_pooled = get_features(model)
    features = [img] + features
    all_labels.append(label.detach().cpu())
    for f, feature in enumerate(features):
      # Evaluate readout predict
      # feature_flatten = feature.mean([2, 3]).detach()
      feature_flatten = feature.view([len(feature), -1]).detach()
      readout_layer = readout_layers[f]
      readout = log_softmax(readout_layer(feature_flatten), dim=1)
      readout_predicts[f].append(readout.argmax(-1).detach().cpu())

      # Evaluate NN predict
      feature_flatten = feature_flatten.unsqueeze(1).expand([-1, n_classes, -1])
      class_vectors = all_class_vectors[f].unsqueeze(0).expand([feature_flatten.shape[0], -1, -1])
      sim = sim_loss(feature_flatten, class_vectors)
      similarity[f].append(sim.detach().cpu())

  all_labels = torch.concat(all_labels)
  for f in range(n_layers):
    similarity[f] = torch.concat(similarity[f])
    prediction = torch.argmin(similarity[f], axis=1)
    acc = (prediction == all_labels).float().mean() * 100
    print("%s Layer %i sim test accuracy:%.2f%%" % (name, f, acc))

  for f in range(n_layers):
    readout_predicts[f] = torch.concat(readout_predicts[f])
    readout_acc = (readout_predicts[f] == all_labels).float().mean() * 100
    print("%s Layer %i readout test accuracy:%.2f%%" % (name, f, readout_acc))


def test():
  model.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
      inputs, targets = inputs.to(args.device), targets.to(args.device)
      readout, features, pre_pooled = model(inputs)
      loss = xentropy_loss(readout, targets)

      test_loss += loss.item()
      _, predicted = readout.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
  print('Same Readout Test Accuracy: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))


for e in range(config['epochs']):
  model.train()
  train_bar = tqdm(train_loader, disable=False)
  train_bar.set_description("Epoch %i" % e)
  last_one_hot = defaultdict(lambda: None)
  train_acc = []
  for step, (img, label) in enumerate(train_bar):
    img = img.to(args.device)
    label = label.to(args.device)

    if config['random_baseline']:
      img = torch.randn_like(img).to(args.device)
    if config['predictive_w']:
      stream_img = []
      stream_label = []
      # make sure the samples are aligned by class between each batch
      samples_per_class = int(img.shape[0] / n_classes * 0.9)
      for c in range(n_classes):
        cls_img = img[label == c]
        if cls_img.shape[0] < samples_per_class:
          # Make up the difference by repeating samples
          short = samples_per_class - cls_img.shape[0]
          cls_img = torch.concat([cls_img, cls_img[:short]])
        stream_img.append(cls_img[:samples_per_class])
        stream_label.append(c * torch.ones(samples_per_class))
      img = torch.concat(stream_img)
      label = torch.concat(stream_label)
    readout, features, pre_pooled = model(img)
    if config['sim_all_layers']:
      features = get_features(model)
      pre_pooled = features
    else:
      pre_pooled = get_features(model)
    loss = torch.zeros(1).to(args.device)

    for f, feature in enumerate(features):
      order = feature.argsort(1, descending=True)
      # Use argmax method
      if config['one_hot_w'] or type(config['one_hot_w']) is list:
        one_hot_label = torch.zeros_like(feature).to(args.device)
        if not config['mlp']:
          one_hot_label = one_hot_label.swapaxes(1, 3)
        for k in range(config['top_k']):
          one_hot_label += F.one_hot(order[:, k], feature.shape[1]).float()
        if not config['mlp']:
          one_hot_label = one_hot_label.swapaxes(1, 3)
        one_hot_label = one_hot_label.detach()
        if type(config['one_hot_w']) is list:
          loss += config['one_hot_w'][f] * one_hot_loss(feature, one_hot_label)
        else:
          loss += config['one_hot_w'] * one_hot_loss(feature, one_hot_label)
      if config['hebbian_w']:
        loss += config['hebbian_w'] * hebbian_loss(feature)
      if config['sim_w']:
        perm = feature[torch.randperm(feature.shape[0])]
        # We want to promote neuron selectivity
        loss += config['sim_w'] * cosine_loss(feature, perm).pow_(2).mean() #TODO: only really across the channel dimension?
      if config['predictive_w'] and last_one_hot[f] is not None:
        loss += config['predictive_w'] * one_hot_loss(feature, last_one_hot[f])
      top_k = order[:, :config['top_k']].flatten()
      unique_features = top_k.unique()
      counter = Counter(top_k.detach().cpu().numpy().tolist())
      counter = {k: "%.2f%%" % (v * 100 /len(top_k)) for k, v in sorted(counter.items(), key=lambda item: -item[1])}
      if config['one_hot_w']:
        last_one_hot[f] = one_hot_label
    if config['xent_w']:
      loss += config['xent_w'] * xentropy_loss(readout, label)
    acc = (readout.argmax(-1) == label).float().mean() * 100
    train_acc.append(acc.detach().cpu().numpy())
    train_bar.set_description("Epoch %i, train accuracy: %.2f%%" % (e, np.stack(train_acc).mean()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if args.debug:
      break
  if type(scheduler) == ReduceLROnPlateau:
    scheduler.step(np.stack(train_acc).mean())
  elif type(scheduler) == CosineAnnealingLR and e < annealing_tmax:
    scheduler.step(e)
  # evaluate("After Training")
  get_sparsity()
  test()
  if config['mlp']:
    inference_speed()

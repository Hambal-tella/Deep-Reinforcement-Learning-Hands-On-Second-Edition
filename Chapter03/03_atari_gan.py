#!/usr/bin/env python
import random
import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

import gymnasium as gym
import gymnasium.spaces
from gymnasium.wrappers import TransformObservation

# Logging
log = gym.logger
log.set_level(gym.logger.INFO)

# Hyperparameters
LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16
IMAGE_SIZE = 64
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

# -------------------------------
# Wrappers
# -------------------------------
class InputWrapper(gym.ObservationWrapper):
    """Resizes and reorders channels to (C,H,W)"""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Box)
        obs_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        obs = cv2.resize(obs, (IMAGE_SIZE, IMAGE_SIZE))
        obs = np.moveaxis(obs, 2, 0)
        return obs.astype(np.float32)

# -------------------------------
# Models
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(input_shape[0], DISCR_FILTERS, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(DISCR_FILTERS, DISCR_FILTERS*2, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(DISCR_FILTERS*2, DISCR_FILTERS*4, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS*4),
            nn.ReLU(),
            nn.Conv2d(DISCR_FILTERS*4, DISCR_FILTERS*8, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS*8),
            nn.ReLU(),
            nn.Conv2d(DISCR_FILTERS*8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)

class Generator(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(LATENT_VECTOR_SIZE, GENER_FILTERS*8, 4, 1, 0),
            nn.BatchNorm2d(GENER_FILTERS*8),
            nn.ReLU(),
            nn.ConvTranspose2d(GENER_FILTERS*8, GENER_FILTERS*4, 4, 2, 1),
            nn.BatchNorm2d(GENER_FILTERS*4),
            nn.ReLU(),
            nn.ConvTranspose2d(GENER_FILTERS*4, GENER_FILTERS*2, 4, 2, 1),
            nn.BatchNorm2d(GENER_FILTERS*2),
            nn.ReLU(),
            nn.ConvTranspose2d(GENER_FILTERS*2, GENER_FILTERS, 4, 2, 1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(GENER_FILTERS, output_shape[0], 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)

# -------------------------------
# Data batch generator
# -------------------------------
def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = []
    env_gen = iter(lambda: random.choice(envs), None)
    while True:
        e = next(env_gen)
        obs, reward, terminated, truncated, info = e.step(e.action_space.sample())
        is_done = terminated or truncated
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()

# -------------------------------
# Main training
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    env_names = ["BreakoutNoFrameskip-v4", "AirRaidNoFrameskip-v4", "PongNoFrameskip-v4"]
    envs = [InputWrapper(gym.make(name)) for name in env_names]
    input_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses, dis_losses = [], []
    iter_no = 0
    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        # Generator input
        gen_input_v = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1, device=device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # Train Discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # Train Generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1

        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", iter_no, np.mean(gen_losses), np.mean(dis_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses, dis_losses = [], []

        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(batch_v.data[:64], normalize=True), iter_no)

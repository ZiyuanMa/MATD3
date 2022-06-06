#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 critic_in_dim,
                 continuous_actions=False):
        super(MAModel, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act, with_q2):
        return self.critic_model(obs, act, with_q2)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=False):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        hid1_size = 64
        hid2_size = 64
        self.fc1 = nn.Linear(
            obs_dim,
            hid1_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(
            hid1_size,
            hid2_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(
            hid2_size,
            act_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        if self.continuous_actions:
            std_hid_size = 64
            self.std_fc = nn.Linear(
                std_hid_size,
                act_dim,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs):
        hid1 = F.relu(self.fc1(obs))
        hid2 = F.relu(self.fc2(hid1))
        means = self.fc3(hid2)
        if self.continuous_actions:
            act_std = self.std_fc(hid2)
            return (means, act_std)
        return means


class CriticModel(parl.Model):
    def __init__(self, critic_in_dim):
        super(CriticModel, self).__init__()
        hid1_size = 64
        hid2_size = 64
        out_dim = 1

        self.q1 = nn.Sequential(
            nn.Linear(critic_in_dim, hid1_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.ReLU(),
            nn.Linear(hid1_size, hid2_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.ReLU(),
            nn.Linear(hid2_size, out_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

        )

        self.q2 = nn.Sequential(
            nn.Linear(critic_in_dim, hid1_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.ReLU(),
            nn.Linear(hid1_size, hid2_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())),
            nn.ReLU(),
            nn.Linear(hid2_size, out_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

        )

    def forward(self, obs_n, act_n, with_q2):
        inputs = paddle.concat(obs_n + act_n, axis=1)

        q1 = self.q1(inputs)
        q1 = paddle.squeeze(q1, axis=1)

        if with_q2:
            q2 = self.q2(inputs)
            q2 = paddle.squeeze(q2, axis=1)
            return q1, q2
        else:
            return q1

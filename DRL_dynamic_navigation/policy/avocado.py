#!/usr/bin/env python

#
# actors.py
# AVOCADO library
#
# SPDX-FileCopyrightText: 2008 University of North Carolina at Chapel Hill
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2024 University of Zaragoza
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AVOCADO, a derivative work of the RVO2 Library.
# Portions of this file are licensed under the Apache License, Version 2.0,
# and modifications are licensed under the GNU Affero General Public License,
# version 3 or later.
#
# If you use AVOCADO in academic work, please cite:
# Martinez-Baselga, D., Sebastián, E., Montijano, E., Riazuelo, L., Sagüés, C., & Montano, L. (2024). AVOCADO: Adaptive Optimal Collision Avoidance driven by Opinion. arXiv preprint arXiv:2407.00507.
# 
# For details, see the LICENSE file at the root of the repository.
# 
# Contact: diegomartinez@unizar.es
# 			esebastian@unizar.es
# 
#

import avocado
import numpy as np

class AVOCADO:
    def __init__(self, radius=0.3, time_step=0.2, a=0.3, c=0.7, d=2, 
                 kappa=14.15, epsilon=3.22, delta=0.57, 
                 alpha=[100.], bias=[0.], neighbor_dist=5.0, time_horizon=5.0, max_noise=0.0005) -> None:
        self.name = 'AVOCADO'
        self.radius = radius
        self.time_step = time_step
        self.sim = None
        self.a = a
        self.c = c
        self.d = d
        self.kappa = kappa
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.bias = bias
        self.neighbor_dist = neighbor_dist
        self.time_horizon = time_horizon
        self.max_noise = max_noise
        self.max_speed = 1.0
        self.max_robot_speed = 1.0
        self.max_neighbors = 10
        self.safety_space = 0.2
        self.max_human_speed = 1.0

        self.id_agents = []
        self.id_others = []
        self.sim = None

    def predict(self, state):
        self_state = state.self_state
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            # RVOSimulator(float timeStep, float neighborDist, size_t maxNeighbors,
            # 				 float timeHorizon, float timeHorizonObst, float radius,
            # 				 float maxSpeed, const Vector2 &velocity = Vector2());
            self.sim = avocado.PyRVOSimulator(self.time_step, self.neighbor_dist, self.max_neighbors, 
                                              self.time_horizon, self.time_horizon, self.radius + self.safety_space, 1)
   
            self.id_agents.append(self.sim.addAgent(self_state.position, const_alpha=self.alpha[0], 
                                                    a=self.a, b=self.bias[0], c=self.c, d=self.d, 
                                                    kappa=self.kappa, epsilon=self.epsilon, delta=self.delta, max_noise=self.max_noise))
            
            for i, human_state in enumerate(state.human_states):
                self.id_others.append(self.sim.addAgent(human_state.position, const_alpha=0.0))

            self.sim.processObstacles()
        else:
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(self.id_others[i], human_state.position)
                self.sim.setAgentVelocity(self.id_others[i], human_state.velocity)
        
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
       
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed * self.max_speed if speed > self.max_speed else velocity

        self.sim.setAgentPrefVelocity(self.id_agents[0], (pref_vel[0], pref_vel[1]))
        self.sim.doStep()
       
        action = self.sim.getAgentVelocity(0)

        return action
    




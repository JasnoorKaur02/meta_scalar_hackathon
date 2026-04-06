---
title: Supply Chain Disruption Env
emoji: chain
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# Supply Chain Disruption Env

A real-world OpenEnv environment for supply chain crisis management.

## Tasks
- assess_disruption (Easy)
- resolve_disruption (Medium)
- cascade_management (Hard)

## Usage
POST /reset - Start a new episode
POST /step - Take an action
GET /state - Get current state
GET /health - Health check

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Thin wrapper — real logic lives in agents/orchestrator_agent.py
# ==================================================================
from agents.orchestrator_agent import OrchestratorAgent

# Backward-compatible alias used by run_spr_agent.py and demo_app_with_front.py
AcademicSearchTree = OrchestratorAgent

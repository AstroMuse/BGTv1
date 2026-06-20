# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Backward-compatibility shim.
#                  Real logic lives in agents/*.py
# ==================================================================
from agents.retrieval_agent import RetrievalAgent, MultiSearchAgent, get_info_from_local
from agents.scoring_agent import ScoringAgent
from agents.reference_agent import ReferenceAgent
from agents.query_agent import QueryAgent, render_domain_tree, count_domain_tree

# Legacy name used by demo_app_with_front.py
MultiSearchAgent = MultiSearchAgent  # noqa: F811

# Legacy monolithic class — delegates to the specialist agents
class AcademicTreeSearchEngine:
    """
    Backward-compatibility facade. New code should use the agent classes directly.
    All methods delegate to the appropriate specialist agent.
    """

    def __init__(self):
        self.query_agent = QueryAgent()
        self.scoring_agent = ScoringAgent()
        self.reference_agent = ReferenceAgent()
        self.mretrival_processer = RetrievalAgent()

    # -- QueryAgent delegation --
    def expand_query(self, query):
        return self.query_agent.expand_query(query)

    def build_domain_tree(self, query, domain, depth=3):
        return self.query_agent.build_domain_tree(query, domain, depth)

    def generate_queries_from_docs(self, query, docs_with_nodes, searched_queries):
        return self.query_agent.generate_queries_from_docs(query, docs_with_nodes, searched_queries)

    # -- RetrievalAgent delegation --
    def search_papers_mroute(self, queries, end_date="", searched_docs=None, sources=None):
        if searched_docs is None:
            searched_docs = {}
        if sources is None:
            sources = ["arxiv"]
        return self.mretrival_processer.search(
            queries, sources=sources, end_date=end_date, searched_docs=searched_docs
        )

    # -- ScoringAgent delegation --
    def calculate_similarity(self, query, docs, search_time="", score_thresh=0.5, source=""):
        return self.scoring_agent.score(query, docs, search_time, score_thresh, source)

    def calculate_sim_bge(self, query, docs, search_time="", score_thresh=0.5, source=""):
        return self.scoring_agent.score_bge(query, docs, search_time, score_thresh, source)

    def calculate_sim_pasa(self, query, docs, search_time="", score_thresh=0.5, source=""):
        return self.scoring_agent.score_pasa(query, docs, search_time, score_thresh, source)

    # -- ReferenceAgent delegation --
    def get_doc_references(self, doc_info):
        return self.reference_agent.get_references(doc_info)

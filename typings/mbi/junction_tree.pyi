from collections.abc import Collection, Sequence

import networkx as nx

from mbi.domain import Domain

Clique = tuple[str, ...]

def hypothetical_model_size(domain: Domain, cliques: Sequence[Clique]) -> float: ...
def make_junction_tree(
    domain: Domain, cliques: Collection[Clique], elimination_order: list[str] | int | None = None
) -> tuple[nx.Graph[Clique], list[str]]: ...
def maximal_cliques(junction_tree: nx.Graph[Clique]) -> list[Clique]: ...
def message_passing_order(junction_tree: nx.Graph[Clique]) -> list[tuple[Clique, Clique]]: ...

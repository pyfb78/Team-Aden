"""
Minimal EFG Parser for Leduc Poker
Designed for TB-DAG construction with only essential fields
"""
from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, FrozenSet
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import zipfile
import io
import os



from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import deque

import numpy as np

ACTION_TO_COL = {"C": 0, "R": 2, "F": 1}
COLS = ["C", "F", "R"]

@dataclass
class GameNode:
    """Minimal node representation for TB-DAG construction"""
    path: str  # Full path like "/C:JJQQ/P1:C/P2:R"
    node_type: str  # 'chance', 'decision', 'leaf'
    player: Optional[int] = None  # 1-4 for decision nodes, None for chance/leaf
    actions: List[str] = None  # Available actions ['C', 'R', 'F']
    children: Dict[str, str] = None  # action -> child_path mapping
    payoffs: Dict[int, float] = None  # player -> payoff (only for leaves)
    chance_probs: Dict[str, float] = None  # action -> probability (only for chance)
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.children is None:
            self.children = {}
        if self.payoffs is None:
            self.payoffs = {}
        if self.chance_probs is None:
            self.chance_probs = {}
    
    def is_active(self, team_players: Set[int]) -> bool:
        """Check if this node is active for the given team"""
        return self.node_type == 'decision' and self.player in team_players
    
    def is_terminal(self) -> bool:
        return self.node_type == 'leaf'


class InfoSet:
    """Information set grouping indistinguishable nodes"""
    def __init__(self, name: str, player: int):
        self.name = name
        self.player = player
        self.nodes: Set[str] = set()  # Set of node paths
        self.actions: List[str] = []  # Available actions at this infoset
    
    def add_node(self, node_path: str):
        self.nodes.add(node_path)


class LeducEFG:
    """Extensive Form Game representation for Leduc Poker"""
    
    def __init__(self):
        self.nodes: Dict[str, GameNode] = {}  # path -> GameNode
        self.infosets: Dict[str, InfoSet] = {}  # infoset_name -> InfoSet
        self.node_to_infoset: Dict[str, str] = {}  # node_path -> infoset_name
        self.root_path: str = "/"
        self.team_1: Set[int] = {1, 3}  # Players 1 and 3
        self.team_2: Set[int] = {2, 4}  # Players 2 and 4
    
    def parse_game_file(self, game_file: str):
        """Parse the main game tree file"""
        with open(game_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('node '):
                    self._parse_node_line(line)
                elif line.startswith('infoset '):
                    self._parse_infoset_line(line)
    
    def _parse_node_line(self, line: str):
        """Parse a single node line from the game file"""
        # Remove 'node ' prefix
        line = line[5:]
        parts = line.split()
        
        if not parts:
            return
        
        path = parts[0]
        
        # Parse chance nodes
        if len(parts) > 1 and parts[1] == 'chance':
            self._parse_chance_node(path, parts)
        
        # Parse decision nodes
        elif len(parts) > 2 and parts[1] == 'player':
            self._parse_decision_node(path, parts)
        
        # Parse leaf nodes
        elif len(parts) > 1 and parts[1] == 'leaf':
            self._parse_leaf_node(path, parts)
    
    def _parse_chance_node(self, path: str, parts: List[str]):
        actions_idx = parts.index('actions')
        actions_data = parts[actions_idx + 1:]

        chance_probs = {}
        for action_prob in actions_data:
            if '=' in action_prob:
                action, prob = action_prob.split('=')
                chance_probs[action] = float(prob)

        if path == '/':
            children = {a: f"/C:{a}" for a in chance_probs}
        else:
            children = {a: f"{path}/C:{a}" for a in chance_probs}

        self.nodes[path] = GameNode(
            path=path,
            node_type='chance',
            actions=list(chance_probs.keys()),
            children=children,
            chance_probs=chance_probs
        )

    
    def _parse_decision_node(self, path: str, parts: List[str]):
        """Parse decision node: node /C:JJQQ/P1:C player 2 actions C R"""
        player = int(parts[2])
        actions_idx = parts.index('actions')
        actions = parts[actions_idx + 1:]
        
        # Build children mapping
        children = {}
        for action in actions:
            child_path = f"{path}/P{player}:{action}"
            children[action] = child_path
        
        node = GameNode(
            path=path,
            node_type='decision',
            player=player,
            actions=actions,
            children=children
        )
        self.nodes[path] = node
    
    def _parse_leaf_node(self, path: str, parts: List[str]):
        """Parse leaf node: node /C:JJQQ/.../P4:C leaf payoffs 1=-1 2=-1 3=1 4=1"""
        payoffs_idx = parts.index('payoffs')
        payoffs_data = parts[payoffs_idx + 1:]
        
        payoffs = {}
        for payoff_str in payoffs_data:
            if '=' in payoff_str:
                player, value = payoff_str.split('=')
                payoffs[int(player)] = float(value)
        
        node = GameNode(
            path=path,
            node_type='leaf',
            payoffs=payoffs
        )
        self.nodes[path] = node
    
    def _parse_infoset_line(self, line: str):
        """Parse: infoset J??? nodes /C:JJQQ /C:JJQK ..."""
        # Remove 'infoset ' prefix
        line = line[8:]
        parts = line.split()
        
        if 'nodes' not in parts:
            return
        
        infoset_name = parts[0]
        nodes_idx = parts.index('nodes')
        node_paths = parts[nodes_idx + 1:]
        
        # Extract player from first node
        if not node_paths:
            return
        
        first_node = self.nodes.get(node_paths[0])
        if not first_node or first_node.player is None:
            return
        
        # Create infoset
        infoset = InfoSet(infoset_name, first_node.player)
        infoset.actions = first_node.actions.copy()
        
        for node_path in node_paths:
            infoset.add_node(node_path)
            self.node_to_infoset[node_path] = infoset_name
        
        self.infosets[infoset_name] = infoset
    
    def get_infoset(self, node_path: str) -> Optional[InfoSet]:
        """Get the information set for a given node"""
        infoset_name = self.node_to_infoset.get(node_path)
        return self.infosets.get(infoset_name) if infoset_name else None
    
    def get_team_infosets(self, team_players: Set[int]) -> List[InfoSet]:
        """Get all infosets for a given team"""
        return [infoset for infoset in self.infosets.values() 
                if infoset.player in team_players]
    
    def get_layer(self, node_path: str) -> int:
        s = node_path.strip("/")
        return 0 if not s else s.count("/") + 1  # number of segments

class ConnectivityGraph:
    """
    Implicit connectivity graph G via hyperedges.
    For each (layer t, team infoset I), the set of layer-t ancestors of nodes in I
    forms a clique in G. We store it as a hyperedge.

    This supports correct connected components of induced subgraphs G[O].
    """

    def __init__(self, efg: LeducEFG, team_players: Set[int]):
        self.efg = efg
        self.team_players = team_players

        self.layers: Dict[int, Set[str]] = defaultdict(set)

        # ancestors[path] = list where ancestors[path][t] = ancestor at layer t
        # layer 0 -> "/"
        self.ancestors: Dict[str, List[str]] = {}

        # hyperedges_by_layer[t] = list of hyperedges (each is a list[str] nodes at layer t)
        self.hyperedges_by_layer: Dict[int, List[List[str]]] = defaultdict(list)

        # node_to_hyperedges[node] = set of (layer t, hyperedge_index)
        self.node_to_hyperedges: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)

        self._build_layers()
        self._precompute_ancestors()
        self._build_connectivity()

    def _build_layers(self):
        for node_path in self.efg.nodes:
            layer = self.efg.get_layer(node_path)
            self.layers[layer].add(node_path)

    def _precompute_ancestors(self):
        for path in self.efg.nodes.keys():
            if path == "/":
                self.ancestors[path] = ["/"]
                continue
            segs = path.strip("/").split("/")
            anc = ["/"]
            for i in range(1, len(segs) + 1):
                anc.append("/" + "/".join(segs[:i]))
            self.ancestors[path] = anc

    def _build_connectivity(self):
        # Only team infosets matter by definition for G in the TB-DAG construction
        team_infosets = self.efg.get_team_infosets(self.team_players)

        # Optional dedup: (layer, tuple(sorted(nodes))) -> existing hyperedge index
        # This can shrink memory in large games
        seen = defaultdict(dict)  # seen[layer][signature] = idx

        for infoset in team_infosets:
            nodes = list(infoset.nodes)
            if not nodes:
                continue

            # Timeability sanity check: all nodes within infoset should be same layer
            d0 = self.efg.get_layer(nodes[0])
            for n in nodes[1:]:
                dn = self.efg.get_layer(n)
                if dn != d0:
                    raise ValueError(
                        f"Infoset {infoset.name} not timeable: {nodes[0]}@{d0} vs {n}@{dn}"
                    )

            # For each earlier layer t, collect the set of ancestors at layer t.
            # If set size < 2, it contributes no edges.
            for t in range(d0 + 1):  # t = 0..d0
                S = set()
                for h in nodes:
                    # ancestors list is indexed by layer
                    anc = self.ancestors[h][t]
                    if anc in self.efg.nodes:
                        S.add(anc)

                if len(S) < 2:
                    continue

                signature = tuple(sorted(S))
                if signature in seen[t]:
                    hid = seen[t][signature]
                else:
                    hid = len(self.hyperedges_by_layer[t])
                    self.hyperedges_by_layer[t].append(list(signature))
                    seen[t][signature] = hid

                for u in signature:
                    self.node_to_hyperedges[u].add((t, hid))

        # keep node_to_hyperedges as sets; no need to convert to list

    # ---------- NEW: team information refinement helpers ----------

    def _team_info_signature(self, h: str) -> Tuple:
        """
        Compute a 'team information' signature for history h:
        for each team player p, take the infoset of p at their last
        decision along the path to h (if any).

        Two histories can only share a TB-DAG belief if these signatures match.
        """
        sig_parts: List[Optional[str]] = []

        for p in sorted(self.team_players):
            last_I: Optional[str] = None
            # ancestors[h] is in prefix order: "/", "/C:...", "/C:.../P1:...", ...
            for anc in self.ancestors[h]:
                if anc not in self.efg.nodes:
                    continue
                node = self.efg.nodes[anc]
                if getattr(node, "player", None) == p:
                    I = self.efg.node_to_infoset.get(anc)
                    if I is not None:
                        last_I = I
            sig_parts.append(last_I)

        return tuple(sig_parts)

    def _refine_components_by_team_info(
        self, comps: List[Set[str]]
    ) -> List[Set[str]]:
        """
        Refine each hypergraph-connected component so that all histories in
        a final component share the same team information signature.
        This prevents grouping states where team players have different
        private knowledge (e.g., different hole cards).
        """
        if not self.team_players:
            return comps

        refined: List[Set[str]] = []
        for comp in comps:
            buckets: Dict[Tuple, Set[str]] = defaultdict(set)
            for h in comp:
                sig = self._team_info_signature(h)
                buckets[sig].add(h)
            refined.extend(buckets.values())
        return refined

    # --------------------------------------------------------------

    def components_induced(self, O: Set[str]) -> List[Set[str]]:
        """
        Return connected components of the induced subgraph G[O].
        Assumes all nodes in O are in the same layer (true for TB-DAG inactive labels).
        """
        if not O:
            return []

        O = set(O)
        layer = self.efg.get_layer(next(iter(O)))
        for x in O:
            if self.efg.get_layer(x) != layer:
                raise ValueError("components_induced expects O to be single-layer")

        visited_nodes: Set[str] = set()
        used_hyperedges: Set[Tuple[int, int]] = set()
        comps: List[Set[str]] = []

        for start in O:
            if start in visited_nodes:
                continue

            comp = {start}
            stack = [start]
            visited_nodes.add(start)

            while stack:
                u = stack.pop()

                for (t, hid) in self.node_to_hyperedges.get(u, set()):
                    if t != layer:
                        continue

                    key = (t, hid)
                    if key in used_hyperedges:
                        continue
                    used_hyperedges.add(key)

                    for v in self.hyperedges_by_layer[t][hid]:
                        if v in O and v not in visited_nodes:
                            visited_nodes.add(v)
                            comp.add(v)
                            stack.append(v)

            comps.append(comp)

        # Refine by team information so beliefs don't mix different team hole cards
        return self._refine_components_by_team_info(comps)


BeliefLabel = FrozenSet[str]
ObsLabel = FrozenSet[str]
PrescriptionKey = Tuple[Tuple[str, str], ...]  # ((infoset_name, action), ...)

@dataclass
class TBDAGActiveNode:
    id: int
    label: BeliefLabel
    is_terminal: bool = False
    terminal_hist: Optional[str] = None
    # prescription -> inactive child id
    children: Dict[PrescriptionKey, int] = field(default_factory=dict)
    parents: Set[int] = field(default_factory=set)

@dataclass
class TBDAGInactiveNode:
    id: int
    label: ObsLabel
    # observation component children (active nodes)
    children: List[int] = field(default_factory=list)
    parents: Set[int] = field(default_factory=set)

class TeamBeliefDAG:
    def __init__(self, efg: LeducEFG, conn_graph: ConnectivityGraph, team_players: Set[int]):
        self.efg = efg
        self.conn = conn_graph
        self.team_players = set(team_players)

        self.active_nodes: List[TBDAGActiveNode] = []
        self.inactive_nodes: List[TBDAGInactiveNode] = []

        self.active_cache: Dict[BeliefLabel, int] = {}
        self.inactive_cache: Dict[ObsLabel, int] = {}

        self.root_active_id: Optional[int] = None

        self.build()

    def build(self) -> int:
        root = frozenset([self.efg.root_path])
        self.root_active_id = self._make_active_node(root)
        return self.root_active_id

    def _label_layer(self, label: FrozenSet[str]) -> int:
        it = iter(label)
        first = next(it)
        layer = self.efg.get_layer(first)
        for x in it:
            if self.efg.get_layer(x) != layer:
                raise ValueError(f"Label spans multiple layers: {first}@{layer} and {x}@{self.efg.get_layer(x)}")
        return layer

    def _make_active_node(self, belief: BeliefLabel) -> int:
        if belief in self.active_cache:
            return self.active_cache[belief]

        # enforce single-layer invariant early
        self._label_layer(belief)

        # terminal base case: Algorithm assumes singleton terminal
        if len(belief) == 1:
            h = next(iter(belief))
            if self.efg.nodes[h].is_terminal():
                nid = len(self.active_nodes)
                node = TBDAGActiveNode(id=nid, label=belief, is_terminal=True, terminal_hist=h)
                self.active_nodes.append(node)
                self.active_cache[belief] = nid
                return nid

        nid = len(self.active_nodes)
        node = TBDAGActiveNode(id=nid, label=belief)
        self.active_nodes.append(node)
        self.active_cache[belief] = nid

        team_infosets = self._get_team_infosets_in_belief(belief)
        inactive_nodes = self._get_inactive_nodes(belief)

        for prescription in self._generate_prescriptions(team_infosets):
            nxt = self._apply_prescription(belief, team_infosets, inactive_nodes, prescription)
            child_inactive = self._make_inactive_node(nxt)
            node.children[prescription] = child_inactive
            self.inactive_nodes[child_inactive].parents.add(nid)

        return nid

    def _make_inactive_node(self, observation_set: ObsLabel) -> int:
        if observation_set in self.inactive_cache:
            return self.inactive_cache[observation_set]

        # enforce single-layer invariant early
        self._label_layer(observation_set)

        nid = len(self.inactive_nodes)
        node = TBDAGInactiveNode(id=nid, label=observation_set)
        self.inactive_nodes.append(node)
        self.inactive_cache[observation_set] = nid

        comps = self.conn.components_induced(set(observation_set))
        for comp in comps:
            child_active = self._make_active_node(frozenset(comp))
            node.children.append(child_active)
            self.active_nodes[child_active].parents.add(nid)

        return nid

    def _get_team_infosets_in_belief(self, belief: BeliefLabel) -> List[Tuple[str, Set[str]]]:
        # group belief nodes by team infoset
        grouped: Dict[str, Set[str]] = defaultdict(set)
        for h in belief:
            n = self.efg.nodes[h]
            if n.is_terminal():
                continue
            if n.is_active(self.team_players):
                I = self.efg.node_to_infoset.get(h)
                if I is None:
                    raise ValueError(f"Active team decision node {h} missing infoset assignment")
                grouped[I].add(h)
        return [(iname, nodeset) for iname, nodeset in grouped.items()]

    def _get_inactive_nodes(self, belief: BeliefLabel) -> Set[str]:
        out = set()
        for h in belief:
            n = self.efg.nodes[h]
            if n.is_terminal():
                continue
            if not n.is_active(self.team_players):
                out.add(h)
        return out

    def _generate_prescriptions(self, team_infosets: List[Tuple[str, Set[str]]]) -> List[PrescriptionKey]:
        if not team_infosets:
            return [tuple()]  # empty prescription

        actions_per_infoset = []
        for infoset_name, _nodeset in team_infosets:
            I = self.efg.infosets[infoset_name]
            actions_per_infoset.append([(infoset_name, a) for a in I.actions])

        # product over infosets → a tuple of (infoset, action) pairs
        return [tuple(p) for p in product(*actions_per_infoset)]

    def _apply_prescription(
        self,
        belief: BeliefLabel,
        team_infosets: List[Tuple[str, Set[str]]],
        inactive_nodes: Set[str],
        prescription: PrescriptionKey
    ) -> ObsLabel:
        action_for_infoset = {iname: a for (iname, a) in prescription}

        nxt: Set[str] = set()

        # team infoset nodes: take the prescribed action
        for infoset_name, nodeset in team_infosets:
            a = action_for_infoset.get(infoset_name)
            if a is None:
                raise ValueError(f"Missing action for infoset {infoset_name} in prescription {prescription}")
            for h in nodeset:
                node = self.efg.nodes[h]
                if a not in node.children:
                    raise ValueError(f"Illegal action {a} at node {h} (infoset {infoset_name})")
                nxt.add(node.children[a])

        # inactive nodes: include ALL action successors
        for h in inactive_nodes:
            node = self.efg.nodes[h]
            if not node.actions:
                raise ValueError(f"Inactive nonterminal node {h} has no actions")
            for a in node.actions:
                if a not in node.children:
                    raise ValueError(f"Child missing for {h} action {a}")
                nxt.add(node.children[a])

        if not nxt:
            raise ValueError(f"Prescription produced empty successor set from belief {belief}")

        return frozenset(nxt)

    # ------------------------------------------------------------------
    # Utility helpers for debugging / analysis
    # ------------------------------------------------------------------
    def summarize(self) -> str:
        """
        Return a short textual summary of the TB-DAG.
        """
        num_active = len(self.active_nodes)
        num_inactive = len(self.inactive_nodes)
        num_terminal = sum(1 for n in self.active_nodes if n.is_terminal)
        return (
            f"TeamBeliefDAG(active={num_active}, inactive={num_inactive}, "
            f"terminal={num_terminal}, root_active_id={self.root_active_id})"
        )

    def get_terminal_payoffs(self, team: Set[int]) -> Dict[int, float]:
        """
        For each terminal active node, compute the team's payoff by summing
        the leaf node payoffs over players in 'team'.
        Returns mapping active_node_id -> payoff.
        """
        out: Dict[int, float] = {}
        for n in self.active_nodes:
            if n.is_terminal:
                pay = self.efg.nodes[n.terminal_hist].payoffs
                out[n.id] = sum(pay.get(p, 0.0) for p in team)
        return out


# ----------------------------------------------------------------------
# Simple summarizers (EFG / TB-DAG) for debugging
# ----------------------------------------------------------------------
def summarize_efg(efg: LeducEFG, max_show: int = 5):
    counts = defaultdict(int)
    layer_hist = defaultdict(int)
    for p, n in efg.nodes.items():
        counts[n.node_type] += 1
        layer_hist[efg.get_layer(p)] += 1

    print("EFG nodes:", len(efg.nodes), dict(counts))
    print("Infosets:", len(efg.infosets))
    per_player = defaultdict(int)
    for I in efg.infosets.values():
        per_player[I.player] += 1
    print("Infosets per player:", dict(per_player))
    print("Layer histogram (first 10 layers):", {k: layer_hist[k] for k in sorted(layer_hist)[:10]})

    show = list(efg.nodes.keys())[:max_show]
    print("Sample node paths:", show)

def validate_efg(efg: LeducEFG):
    # root exists
    if efg.root_path not in efg.nodes:
        raise ValueError("Root '/' not found in nodes")

    # all children exist
    for p, n in efg.nodes.items():
        for a, cp in n.children.items():
            if cp not in efg.nodes:
                raise ValueError(f"Missing child: {p} --{a}--> {cp}")

    # every decision node has infoset
    for p, n in efg.nodes.items():
        if n.node_type == "decision" and p not in efg.node_to_infoset:
            raise ValueError(f"Decision node missing infoset: {p}")

    # infoset consistency (same player & same actions)
    for I in efg.infosets.values():
        players = {efg.nodes[h].player for h in I.nodes}
        if len(players) != 1 or next(iter(players)) != I.player:
            raise ValueError(f"Infoset {I.name} inconsistent players: {players}")
        actions = {tuple(efg.nodes[h].actions) for h in I.nodes}
        if len(actions) != 1:
            raise ValueError(f"Infoset {I.name} inconsistent actions: {actions}")

    print("EFG validation: OK")

def summarize_connectivity(conn: ConnectivityGraph, max_layers: int = 22):
    print("Connectivity hyperedges by layer:")
    for t in range(max_layers):
        hedges = conn.hyperedges_by_layer.get(t, [])
        sizes = [len(h) for h in hedges]
        if sizes:
            print(f"  layer {t}: {len(hedges)} hyperedges, size avg={sum(sizes)/len(sizes):.2f}, max={max(sizes)}")
        else:
            print(f"  layer {t}: 0 hyperedges")

def debug_components(conn: ConnectivityGraph, O: Set[str]):
    comps = conn.components_induced(O)
    print("O size:", len(O), "=> #components:", len(comps), "sizes:", sorted([len(c) for c in comps], reverse=True))
    # Print actual sets for tiny O
    if len(O) <= 10:
        print("Components:", [sorted(list(c)) for c in comps])

def summarize_tbdag(dag: TeamBeliefDAG, max_active: int = 22, max_inactive: int = 22):
    A = len(dag.active_nodes)
    I = len(dag.inactive_nodes)
    T = sum(1 for n in dag.active_nodes if n.is_terminal)
    print(f"TB-DAG: active={A}, inactive={I}, terminal_active={T}, root_active={dag.root_active_id}")

    # quick invariants
    for n in dag.active_nodes:
        dag._label_layer(n.label)  # raises if multi-layer
    for n in dag.inactive_nodes:
        dag._label_layer(n.label)

    # show first few nodes
    for k in range(min(max_active, A)):
        n = dag.active_nodes[k]
        print(f" Active[{k}] layer={dag._label_layer(n.label)} |label|={len(n.label)} "
              f"terminal={n.is_terminal} #actions={len(n.children)}")
    for k in range(min(max_inactive, I)):
        n = dag.inactive_nodes[k]
        print(f" Inactive[{k}] layer={dag._label_layer(n.label)} |label|={len(n.label)} #obs_children={len(n.children)}")

    print("TB-DAG summary: OK")



# Matches parser.py
PrescriptionKey = Tuple[Tuple[str, str], ...]  # ((infoset_name, action), ...)


@dataclass
class TwoTeamDCFRConfig:
    # Defaults consistent with the DCFR(1.5,0,2) variant referenced in the paper.
    alpha: float = 1.5
    beta: float = 0.0
    gamma: float = 2.0
    eps: float = 1e-12


class DAGDCFRRegretMinimizer:
    """
    DCFR regret minimizer for a TB-DAG using Algorithm 2's:
      - NEXTSTRATEGY: top-down reach computation + regret-matching at active nodes
      - OBSERVEUTILITY: bottom-up u-backup + regret updates + parent utility propagation

    TB-DAG interface assumptions from parser.py:
      - dag.active_nodes[s].children: dict[prescription -> inactive_id]
      - dag.inactive_nodes[o].children: list[active_id]
      - dag.active_nodes[s].parents: set[inactive_id]
      - dag.root_active_id: int
      - dag.active_nodes[s].is_terminal and terminal_hist for singleton belief leaves
    """

    def __init__(self, dag, *, config: TwoTeamDCFRConfig = TwoTeamDCFRConfig()):
        self.dag = dag
        self.cfg = config

        if dag.root_active_id is None:
            raise ValueError("dag.root_active_id is None; did you call dag.build()?")

        self.root: int = int(dag.root_active_id)
        self.nA: int = len(dag.active_nodes)
        self.nI: int = len(dag.inactive_nodes)

        # Per-active-node actions (prescriptions) and child inactive ids in aligned order
        self._act_keys: List[List[PrescriptionKey]] = []
        self._act_child_inactive: List[np.ndarray] = []
        for s in range(self.nA):
            node = dag.active_nodes[s]
            keys = list(node.children.keys())
            self._act_keys.append(keys)
            if keys:
                self._act_child_inactive.append(
                    np.array([node.children[k] for k in keys], dtype=np.int32)
                )
            else:
                self._act_child_inactive.append(np.zeros((0,), dtype=np.int32))

        # Cumulative regrets R[s][i]
        self.R: List[np.ndarray] = [
            np.zeros(len(self._act_keys[s]), dtype=np.float64) for s in range(self.nA)
        ]

        # Latest behavioral policy pi[s] (array over prescriptions at s)
        self.pi: List[Optional[np.ndarray]] = [None for _ in range(self.nA)]

        # Average strategy tracking (weighted by t^gamma)
        self.avg_num: List[np.ndarray] = [
            np.zeros(len(self._act_keys[s]), dtype=np.float64) for s in range(self.nA)
        ]
        self.avg_den: np.ndarray = np.zeros(self.nA, dtype=np.float64)

        # Terminal bookkeeping (set via set_terminal_order)
        self._terminal_hists: Optional[List[str]] = None
        self._terminal_active_ids: Optional[np.ndarray] = None  # aligned to _terminal_hists

        # Active-node topological orders (forward/backward)
        self.active_topo: List[int] = self._compute_active_topological_order()
        self.active_rev_topo: List[int] = list(reversed(self.active_topo))

    def set_terminal_order(self, terminal_hists: List[str]) -> None:
        """Fix a global leaf ordering so utilities/terminal reach use arrays."""
        hist_to_active: Dict[str, int] = {}
        for n in self.dag.active_nodes:
            if n.is_terminal:
                hist_to_active[n.terminal_hist] = n.id

        ids = []
        for h in terminal_hists:
            if h not in hist_to_active:
                raise ValueError(f"Terminal history {h} not found among dag terminal active nodes")
            ids.append(hist_to_active[h])

        self._terminal_hists = list(terminal_hists)
        self._terminal_active_ids = np.array(ids, dtype=np.int32)

    def _compute_active_topological_order(self) -> List[int]:
        """
        Build an active-node DAG where edges are:
          active s -> active t iff exists inactive o in children(s) with t in children(o)
        Then Kahn topological order on reachable subgraph from root.
        """
        succ: List[Set[int]] = [set() for _ in range(self.nA)]
        for s, a_node in enumerate(self.dag.active_nodes):
            for o in a_node.children.values():
                for t in self.dag.inactive_nodes[o].children:
                    succ[s].add(t)

        # Reachable
        reachable: Set[int] = set()
        q = deque([self.root])
        reachable.add(self.root)
        while q:
            v = q.popleft()
            for w in succ[v]:
                if w not in reachable:
                    reachable.add(w)
                    q.append(w)

        indeg = {v: 0 for v in reachable}
        for v in reachable:
            for w in succ[v]:
                if w in reachable:
                    indeg[w] += 1

        dq = deque([v for v, d in indeg.items() if d == 0])
        order: List[int] = []
        while dq:
            v = dq.popleft()
            order.append(v)
            for w in succ[v]:
                if w not in indeg:
                    continue
                indeg[w] -= 1
                if indeg[w] == 0:
                    dq.append(w)

        if len(order) != len(reachable):
            # Fallback (shouldn't happen for well-formed TB-DAGs)
            return [self.root] + [v for v in range(self.nA) if v != self.root]

        if order and order[0] != self.root and self.root in order:
            order.remove(self.root)
            order.insert(0, self.root)

        return order

    def _regret_matching(self, r: np.ndarray) -> np.ndarray:
        pos = np.maximum(r, 0.0)
        s = float(pos.sum())
        if s <= self.cfg.eps:
            return np.full_like(r, 1.0 / len(r), dtype=np.float64)
        return pos / s

    def next_strategy(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algorithm 2: NEXTSTRATEGY top-down.
        Returns (x_active, x_inactive). Also stores pi[s] and updates avg accumulators.
        """
        if t < 1:
            raise ValueError("t must start at 1")

        xA = np.zeros(self.nA, dtype=np.float64)
        xO = np.zeros(self.nI, dtype=np.float64)
        xA[self.root] = 1.0

        w_avg = float(t ** self.cfg.gamma)

        for s in self.active_topo:
            if s != self.root:
                parents = self.dag.active_nodes[s].parents
                xA[s] = sum(xO[o] for o in parents) if parents else 0.0

            node = self.dag.active_nodes[s]
            if node.is_terminal or not self._act_keys[s]:
                continue

            pi = self._regret_matching(self.R[s])
            self.pi[s] = pi

            # x[sa] = x'[sa] * x[s] (Algorithm 2 line 12)
            child_o = self._act_child_inactive[s]
            np.add.at(xO, child_o, xA[s] * pi)

            # avg policy accumulation: numerator += w * reach * pi, denom += w * reach
            self.avg_den[s] += w_avg * xA[s]
            self.avg_num[s] += w_avg * xA[s] * pi

        return xA, xO

    def average_strategy_profile(self) -> Dict[int, Tuple[List[PrescriptionKey], np.ndarray]]:
        """
        Returns {active_node_id: (prescription_keys, probs)} for nonterminal decision nodes.
        probs aligns with prescription_keys and with dag.active_nodes[s].children iteration order used in __init__.
        """
        prof = {}
        for s in range(self.nA):
            if not self._act_keys[s]:
                continue
            node = self.dag.active_nodes[s]
            if node.is_terminal:
                continue
            prof[s] = (self._act_keys[s], self.average_policy(s))
        return prof


    def average_reach(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Top-down reach computation using the AVERAGE policy at each active node.
        Returns (x_active, x_inactive).
        """
        xA = np.zeros(self.nA, dtype=np.float64)
        xO = np.zeros(self.nI, dtype=np.float64)
        xA[self.root] = 1.0

        for s in self.active_topo:
            if s != self.root:
                parents = self.dag.active_nodes[s].parents
                xA[s] = sum(xO[o] for o in parents) if parents else 0.0

            node = self.dag.active_nodes[s]
            if node.is_terminal or not self._act_keys[s]:
                continue

            pi_bar = self.average_policy(s)
            child_o = self._act_child_inactive[s]
            np.add.at(xO, child_o, xA[s] * pi_bar)

        return xA, xO


    def average_terminal_reach(self) -> np.ndarray:
        if self._terminal_active_ids is None:
            raise RuntimeError("Call set_terminal_order(...) first")
        xA, _ = self.average_reach()
        return xA[self._terminal_active_ids]

    def observe_utility(self, u_terminal: np.ndarray, t: int) -> None:
        """
        Algorithm 2: OBSERVEUTILITY bottom-up + DCFR discounting of cumulative regrets.
        u_terminal aligned to set_terminal_order.
        """
        if self._terminal_active_ids is None:
            raise RuntimeError("Call set_terminal_order(...) before observe_utility")
        if len(u_terminal) != len(self._terminal_active_ids):
            raise ValueError("u_terminal length mismatch with terminal order")

        uA = np.zeros(self.nA, dtype=np.float64)
        uO = np.zeros(self.nI, dtype=np.float64)
        uA[self._terminal_active_ids] = u_terminal

        # DCFR discount (applied before adding instantaneous regrets)
        disc_base = float(t / (t + 1.0))
        disc_pos = disc_base ** self.cfg.alpha
        disc_neg = disc_base ** self.cfg.beta

        for s in self.active_rev_topo:
            node = self.dag.active_nodes[s]

            if (not node.is_terminal) and self._act_keys[s]:
                pi = self.pi[s]
                if pi is None:
                    pi = self._regret_matching(self.R[s])
                    self.pi[s] = pi

                child_o = self._act_child_inactive[s]
                u_children = uO[child_o]

                # u[s] += sum_{a'} u[sa'] * x'[sa'] (Algorithm 2 line 18)
                uA[s] += float(np.dot(pi, u_children))

                # R[sa] += u[sa] - u[s] (Algorithm 2 line 20), with DCFR discount first
                r = self.R[s]
                pos_mask = r > 0.0
                r[pos_mask] *= disc_pos
                r[~pos_mask] *= disc_neg
                r += (u_children - uA[s])

            # u[parent] += u[s] (Algorithm 2 lines 21-22)
            for o in node.parents:
                uO[o] += uA[s]

    def terminal_reach(self, xA: np.ndarray) -> np.ndarray:
        if self._terminal_active_ids is None:
            raise RuntimeError("Call set_terminal_order(...) first")
        return xA[self._terminal_active_ids]

    def average_policy(self, s: int) -> np.ndarray:
        if not self._act_keys[s]:
            return np.zeros((0,), dtype=np.float64)
        denom = float(self.avg_den[s])
        if denom <= self.cfg.eps:
            return np.full((len(self._act_keys[s]),), 1.0 / len(self._act_keys[s]), dtype=np.float64)
        return self.avg_num[s] / denom


def build_terminal_order_from_efg(efg) -> List[str]:
    """Deterministic leaf ordering: sorted leaf paths."""
    leaves = [p for p, n in efg.nodes.items() if getattr(n, "node_type", None) == "leaf"]
    leaves.sort()
    return leaves


def chance_reach_prob(efg, terminal_hist: str) -> float:
    """
    Multiply chance probabilities along the path to terminal_hist.
    Assumes parser.py path encoding where chance edges are '/C:<action>'.
    """
    if terminal_hist == "/":
        return 1.0

    segs = terminal_hist.strip("/").split("/")
    cur = "/"
    p = 1.0

    for seg in segs:
        node = efg.nodes[cur]
        if getattr(node, "node_type", None) == "chance":
            if not seg.startswith("C:"):
                raise ValueError(f"Expected chance segment at {cur}, got {seg} in terminal {terminal_hist}")
            a = seg[2:]
            p *= float(node.chance_probs[a])

        cur = ("/" + seg) if cur == "/" else (cur + "/" + seg)

    return p


def team_payoff_at_terminal(efg, terminal_hist: str, team_players: Set[int]) -> float:
    node = efg.nodes[terminal_hist]
    pay = getattr(node, "payoffs", {}) or {}
    return float(sum(pay.get(p, 0.0) for p in team_players))


def compute_infoset_action_marginals_from_tmecor(rm) -> Dict[str, Dict[str, float]]:
    """
    Approximate p(a | infoset I) from the TMECor learned on the TB-DAG.
    Returns: marg[I_name][a_char] where a_char in {'C','F','R'}.
    """
    xA_bar, _ = rm.average_reach()  # uses average policies at each active node

    counts = defaultdict(float)  # (I_name, action) -> weighted count
    totals = defaultdict(float)  # I_name -> total weight

    for s in range(rm.nA):
        node = rm.dag.active_nodes[s]
        if node.is_terminal or not rm._act_keys[s]:
            continue

        reach_s = float(xA_bar[s])
        if reach_s <= 0.0:
            continue

        pi_bar = rm.average_policy(s)  # probs over prescriptions at node s
        keys = rm._act_keys[s]

        for idx, pres in enumerate(keys):
            prob_pres = reach_s * float(pi_bar[idx])
            if prob_pres <= 0.0:
                continue

            # pres is a tuple of (infoset_name, action_char) pairs
            for (I_name, a_char) in pres:
                counts[(I_name, a_char)] += prob_pres
                totals[I_name] += prob_pres

    marg: Dict[str, Dict[str, float]] = {}
    for I_name, tot in totals.items():
        if tot <= 0.0:
            continue
        d: Dict[str, float] = {}
        for a_char in ("C", "F", "R"):
            c = counts.get((I_name, a_char), 0.0)
            if c > 0.0:
                d[a_char] = c / tot
        # renormalize if needed
        s = sum(d.values())
        if s <= 0.0:
            continue
        for k in list(d.keys()):
            d[k] /= s
        marg[I_name] = d

    return marg

def run_two_team_dcfr(
    *,
    efg,
    dag_team1,
    dag_team2,
    team1_players: Set[int],
    iterations: int,
    config: TwoTeamDCFRConfig = TwoTeamDCFRConfig(),
    log_every: int = 50,
):
    """
    Two coupled DCFR minimizers:
      x_{t+1} = NEXTSTRATEGY(R1),  y_{t+1} = NEXTSTRATEGY(R2)
      Team1 observes u1[z] = (payoff_team1(z) * chance(z)) * y[z]
      Team2 observes u2[z] = -(payoff_team1(z) * chance(z)) * x[z]

    (This is the diagonal bilinear form x^T diag(u_coeff) y.)
    """
    terminal_hists = build_terminal_order_from_efg(efg)
    Z = len(terminal_hists)

    # Diagonal coefficients: payoff_team1(z) * chance(z)
    u_coeff = np.zeros(Z, dtype=np.float64)
    for i, h in enumerate(terminal_hists):
        u_coeff[i] = team_payoff_at_terminal(efg, h, team1_players) * chance_reach_prob(efg, h)

    rm1 = DAGDCFRRegretMinimizer(dag_team1, config=config)
    rm2 = DAGDCFRRegretMinimizer(dag_team2, config=config)
    rm1.set_terminal_order(terminal_hists)
    rm2.set_terminal_order(terminal_hists)

    for t in range(1, iterations + 1):
        xA1, _ = rm1.next_strategy(t)
        xA2, _ = rm2.next_strategy(t)

        x_term = rm1.terminal_reach(xA1)
        y_term = rm2.terminal_reach(xA2)

        rm1.observe_utility(u_coeff * y_term, t)
        rm2.observe_utility(-u_coeff * x_term, t)

        if log_every and (t % log_every == 0 or t == 1 or t == iterations):
            xbar = rm1.average_terminal_reach()
            ybar = rm2.average_terminal_reach()

            # u_coeff was your (team1 payoff * chance) diagonal term
            ev_team1_avg = float(np.dot(u_coeff, xbar * ybar))

            print(f"Team 1 EV on Iteration {t}:", ev_team1_avg)


    return rm1, rm2, terminal_hists, u_coeff


# ============================================================
# PSRO (Policy Space Response Oracles) on the TB-DAG bilinear game
# ============================================================

@dataclass(frozen=True)
class DeterministicTeamPolicy:
    """
    A deterministic policy on the TB-DAG: for each active decision node s,
    pick one prescription index into rm._act_keys[s].
    """
    picks: Dict[int, int]  # active_node_id -> idx in rm._act_keys[s]

    def pick(self, s: int) -> Optional[int]:
        return self.picks.get(s)


def _rm_strategy_from_regrets(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pos = np.maximum(r, 0.0)
    s = float(pos.sum())
    if s <= eps:
        return np.full_like(r, 1.0 / len(r), dtype=np.float64)
    return pos / s


def solve_zero_sum_meta_game_regret_matching(
    M: np.ndarray,
    iters: int = 3000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate Nash equilibrium for a 2-player zero-sum matrix game with payoff M to player 1.
    Returns (x, y) as average strategies over regret-matching iterations.
    """
    rng = np.random.default_rng(seed)
    n1, n2 = M.shape
    r1 = np.zeros(n1, dtype=np.float64)
    r2 = np.zeros(n2, dtype=np.float64)

    x = np.full(n1, 1.0 / n1, dtype=np.float64)
    y = np.full(n2, 1.0 / n2, dtype=np.float64)

    x_avg = np.zeros(n1, dtype=np.float64)
    y_avg = np.zeros(n2, dtype=np.float64)

    for t in range(1, iters + 1):
        # current strategies from regrets
        x = _rm_strategy_from_regrets(r1)
        y = _rm_strategy_from_regrets(r2)

        # payoffs of pure actions
        u1_actions = M @ y                      # shape (n1,)
        u1 = float(x @ u1_actions)

        u2_actions = -(x @ M)                   # player2 payoff for each column action
        u2 = float(y @ u2_actions)

        # instantaneous regrets
        r1 += (u1_actions - u1)
        r2 += (u2_actions - u2)

        # uniform averaging (could weight by t if desired)
        x_avg += x
        y_avg += y

    x_avg /= float(iters)
    y_avg /= float(iters)
    x_avg /= x_avg.sum()
    y_avg /= y_avg.sum()
    return x_avg, y_avg


def compute_reach_from_policy(
    rm: "DAGDCFRRegretMinimizer",
    pol: DeterministicTeamPolicy,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-down reach (xA, xO) on the TB-DAG for a deterministic team policy.
    This mirrors rm.next_strategy(), but uses `pol` instead of regret-matching.
    """
    xA = np.zeros(rm.nA, dtype=np.float64)
    xO = np.zeros(rm.nI, dtype=np.float64)
    xA[rm.root] = 1.0

    for s in rm.active_topo:
        if s != rm.root:
            parents = rm.dag.active_nodes[s].parents
            xA[s] = sum(xO[o] for o in parents) if parents else 0.0

        node = rm.dag.active_nodes[s]
        if node.is_terminal or not rm._act_keys[s]:
            continue

        idx = pol.picks.get(s)
        if idx is None:
            # fallback: choose first prescription
            idx = 0

        child_o = rm._act_child_inactive[s]
        o = int(child_o[idx])
        xO[o] += xA[s]

    return xA, xO


def terminal_reach_from_policy(rm: "DAGDCFRRegretMinimizer", pol: DeterministicTeamPolicy) -> np.ndarray:
    if rm._terminal_active_ids is None:
        raise RuntimeError("Call set_terminal_order(...) first")
    xA, _ = compute_reach_from_policy(rm, pol)
    return xA[rm._terminal_active_ids]


def random_deterministic_policy(rm: "DAGDCFRRegretMinimizer", seed: int = 0) -> DeterministicTeamPolicy:
    rng = np.random.default_rng(seed)
    picks: Dict[int, int] = {}
    for s in range(rm.nA):
        if not rm._act_keys[s]:
            continue
        if rm.dag.active_nodes[s].is_terminal:
            continue
        picks[s] = int(rng.integers(0, len(rm._act_keys[s])))
    return DeterministicTeamPolicy(picks=picks)


def best_response_policy(
    rm: "DAGDCFRRegretMinimizer",
    u_terminal: np.ndarray,
) -> DeterministicTeamPolicy:
    """
    Pure (deterministic) best response on the TB-DAG to a fixed terminal utility vector `u_terminal`.
    This is the same bottom-up backup used in observe_utility(), but replacing expectation with max.
    """
    if rm._terminal_active_ids is None:
        raise RuntimeError("Call set_terminal_order(...) first")
    if len(u_terminal) != len(rm._terminal_active_ids):
        raise ValueError("u_terminal length mismatch with terminal order")

    uA = np.zeros(rm.nA, dtype=np.float64)
    uO = np.zeros(rm.nI, dtype=np.float64)
    uA[rm._terminal_active_ids] = u_terminal

    picks: Dict[int, int] = {}

    for s in reversed(rm.active_topo):
        node = rm.dag.active_nodes[s]

        if (not node.is_terminal) and rm._act_keys[s]:
            child_o = rm._act_child_inactive[s]
            u_children = uO[child_o]  # shape (num_prescriptions,)

            # tie-break deterministically by argmax
            idx = int(np.argmax(u_children))
            picks[s] = idx
            uA[s] += float(u_children[idx])

        for o in node.parents:
            uO[o] += uA[s]

    return DeterministicTeamPolicy(picks=picks)


def run_two_team_psro(
    *,
    efg,
    dag_team1,
    dag_team2,
    team1_players: Set[int],
    psro_iters: int = 10,
    meta_iters: int = 3000,
    seed: int = 0,
    log_every: int = 1,
    config: TwoTeamDCFRConfig = TwoTeamDCFRConfig(),  # only used to build rm objects
) -> Tuple[
    "DAGDCFRRegretMinimizer", "DAGDCFRRegretMinimizer",
    List[DeterministicTeamPolicy], List[DeterministicTeamPolicy],
    np.ndarray, np.ndarray,
    List[str], np.ndarray, np.ndarray
]:
    """
    PSRO loop using:
      - policies as deterministic TB-DAG prescriptions,
      - meta-solver = regret-matching on the empirical payoff matrix,
      - oracle = deterministic best response by DP on the TB-DAG.

    Returns:
      rm1, rm2,
      pop1, pop2,
      meta_x, meta_y,
      terminal_hists, u_coeff, M
    """
    terminal_hists = build_terminal_order_from_efg(efg)
    Z = len(terminal_hists)

    # Diagonal coefficients: payoff_team1(z) * chance(z)
    u_coeff = np.zeros(Z, dtype=np.float64)
    for i, h in enumerate(terminal_hists):
        u_coeff[i] = team_payoff_at_terminal(efg, h, team1_players) * chance_reach_prob(efg, h)

    rm1 = DAGDCFRRegretMinimizer(dag_team1, config=config)
    rm2 = DAGDCFRRegretMinimizer(dag_team2, config=config)
    rm1.set_terminal_order(terminal_hists)
    rm2.set_terminal_order(terminal_hists)

    # initial policies
    pop1: List[DeterministicTeamPolicy] = [random_deterministic_policy(rm1, seed=seed)]
    pop2: List[DeterministicTeamPolicy] = [random_deterministic_policy(rm2, seed=seed + 1)]

    x_terms = [terminal_reach_from_policy(rm1, pop1[0])]
    y_terms = [terminal_reach_from_policy(rm2, pop2[0])]

    # payoff matrix M[i,j] = u_coeff · (x_i * y_j)
    M = np.array([[float(np.dot(u_coeff, x_terms[0] * y_terms[0]))]], dtype=np.float64)

    meta_x = np.array([1.0], dtype=np.float64)
    meta_y = np.array([1.0], dtype=np.float64)

    for it in range(1, psro_iters + 1):
        meta_x, meta_y = solve_zero_sum_meta_game_regret_matching(M, iters=meta_iters, seed=seed + 103 * it)

        x_mix = np.sum(np.stack(x_terms, axis=0) * meta_x[:, None], axis=0)
        y_mix = np.sum(np.stack(y_terms, axis=0) * meta_y[:, None], axis=0)

        meta_value = float(np.dot(u_coeff, x_mix * y_mix))

        # Oracles: best response to the opponent mixture
        br1 = best_response_policy(rm1, u_coeff * y_mix)
        br2 = best_response_policy(rm2, (-u_coeff) * x_mix)

        x_br = terminal_reach_from_policy(rm1, br1)
        y_br = terminal_reach_from_policy(rm2, br2)

        # Expand payoff matrix correctly: old M is (n1 x n2) before adding BRs
        n1_old = len(pop1)
        n2_old = len(pop2)

        row = np.array([float(np.dot(u_coeff, x_br * y_terms[j])) for j in range(n2_old)], dtype=np.float64)
        col = np.array([float(np.dot(u_coeff, x_terms[i] * y_br)) for i in range(n1_old)], dtype=np.float64)
        corner = float(np.dot(u_coeff, x_br * y_br))

        M_new = np.zeros((n1_old + 1, n2_old + 1), dtype=np.float64)
        M_new[:n1_old, :n2_old] = M
        M_new[n1_old, :n2_old] = row
        M_new[:n1_old, n2_old] = col
        M_new[n1_old, n2_old] = corner
        M = M_new

        pop1.append(br1)
        pop2.append(br2)
        x_terms.append(x_br)
        y_terms.append(y_br)

        if log_every and (it % log_every == 0 or it == 1 or it == psro_iters):
            print(f"[psro] iter={it:03d}  meta_value≈{meta_value:.6f}  pop={len(pop1)}x{len(pop2)}")

    # recompute final meta on final matrix
    meta_x, meta_y = solve_zero_sum_meta_game_regret_matching(M, iters=meta_iters, seed=seed + 9999)
    return rm1, rm2, pop1, pop2, meta_x, meta_y, terminal_hists, u_coeff, M


def compute_infoset_action_marginals_from_policy(
    rm: "DAGDCFRRegretMinimizer",
    pol: DeterministicTeamPolicy,
) -> Dict[str, Dict[str, float]]:
    """
    Project a deterministic TB-DAG team policy into per-infoset action marginals by
    averaging over active-node reach (like compute_infoset_action_marginals_from_tmecor).
    """
    xA, _ = compute_reach_from_policy(rm, pol)

    counts = defaultdict(float)  # (I_name, action) -> weighted count
    totals = defaultdict(float)  # I_name -> total weight

    for s in range(rm.nA):
        node = rm.dag.active_nodes[s]
        if node.is_terminal or not rm._act_keys[s]:
            continue

        reach_s = float(xA[s])
        if reach_s <= 0.0:
            continue

        idx = pol.picks.get(s, 0)
        pres_key = rm._act_keys[s][idx]
        for (I_name, a_char) in pres_key:
            counts[(I_name, a_char)] += reach_s
            totals[I_name] += reach_s

    marg: Dict[str, Dict[str, float]] = {}
    for I_name, tot in totals.items():
        if tot <= 0.0:
            continue
        d: Dict[str, float] = {}
        for a_char in ("C", "F", "R"):
            c = counts.get((I_name, a_char), 0.0)
            if c > 0.0:
                d[a_char] = c / tot
        s = sum(d.values())
        if s <= 0.0:
            continue
        for k in list(d.keys()):
            d[k] /= s
        marg[I_name] = d

    return marg


def export_team_zip_from_policies(
    team: str,                      # "13" or "24"
    rm: "DAGDCFRRegretMinimizer",
    policies: List[DeterministicTeamPolicy],
    meta_probs: np.ndarray,
    infoset_dir: str,
    zip_path: str,
    seed: int = 0,
):
    """
    Export a team zip where each strategy j is derived from the j-th TB-DAG team policy
    by projecting to per-infoset marginals and sampling per-player pure strategies.

    This preserves PSRO's meta distribution over the population size L=len(policies).
    """
    assert team in ("13", "24")
    players = [int(ch) for ch in team]

    L = len(policies)
    meta_probs = np.asarray(meta_probs, dtype=np.float64)
    if len(meta_probs) != L:
        raise ValueError(f"meta_probs length {len(meta_probs)} must equal number of policies {L}")
    meta_probs = meta_probs / meta_probs.sum()

    rng = np.random.default_rng(seed)

    with zipfile.ZipFile(zip_path, "w") as zf:
        # meta-strategy.csv
        with zf.open("meta-strategy.csv", "w") as f:
            f.write(b"strategy,prob\n")
            for j in range(L):
                f.write(f"{j},{float(meta_probs[j])}\n".encode("utf-8"))

        for j, pol in enumerate(policies):
            marg = compute_infoset_action_marginals_from_policy(rm, pol)
            for player in players:
                infoset_path = os.path.join(infoset_dir, f"player_{player}_infosets.txt")
                tensor = sample_pure_player_tensor(infoset_path, marg, rng)
                fname = f"strategy{j}-player{player}.npy"
                with zf.open(fname, "w") as f:
                    np.save(f, tensor)


def _read_infoset_file(path: str):
    infosets = []
    valids = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 3:
                raise ValueError(f"{path}: line {lineno} does not have 3 parts: {line!r}")
            idx_s, I_name, actions_s = parts
            idx = int(idx_s)
            if idx != lineno:
                raise ValueError(f"{path}: expected idx {lineno} but saw {idx} on line {lineno}")
            infosets.append(I_name)
            valids.append(actions_s)
    return infosets, valids


def sample_pure_player_tensor(
    infoset_path: str,
    marg: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For a single player, build an (n_infosets x 3) tensor of one-hot probabilities
    for (C, F, R), sampling actions from marginals p(a|I).
    """
    infosets, valids = _read_infoset_file(infoset_path)
    n_infosets = len(infosets)
    tensor = np.zeros((n_infosets, 3), dtype=np.float64)

    for i, (I_name, actions_s) in enumerate(zip(infosets, valids)):
        dist = marg.get(I_name)

        if not dist:
            # Unseen / unreachable infoset in TMECor: fallback to uniform over valid actions
            k = len(actions_s)
            dist = {a: 1.0 / k for a in actions_s}

        # turn into [p_C, p_F, p_R]
        probs = np.array([
            dist.get("C", 0.0),
            dist.get("F", 0.0),
            dist.get("R", 0.0),
        ], dtype=np.float64)

        # zero out invalid moves
        if "C" not in actions_s:
            probs[0] = 0.0
        if "F" not in actions_s:
            probs[1] = 0.0
        if "R" not in actions_s:
            probs[2] = 0.0

        s = probs.sum()
        if s <= 0.0:
            # fallback uniform over valid actions
            k = len(actions_s)
            probs = np.array([
                1.0 / k if "C" in actions_s else 0.0,
                1.0 / k if "F" in actions_s else 0.0,
                1.0 / k if "R" in actions_s else 0.0,
            ], dtype=np.float64)
        else:
            probs /= s

        # sample a *deterministic* action from probs
        a_idx = int(rng.choice(3, p=probs))
        row = np.zeros(3, dtype=np.float64)
        row[a_idx] = 1.0

        # safety: ensure chosen action is legal
        if (a_idx == 0 and "C" not in actions_s) or \
           (a_idx == 1 and "F" not in actions_s) or \
           (a_idx == 2 and "R" not in actions_s):
            raise RuntimeError(f"Sampled invalid action at infoset {I_name}, actions={actions_s}")

        tensor[i] = row

    return tensor

def export_team_zip_from_tmecor(
    team: str,                      # "13" or "24"
    marg: Dict[str, Dict[str,float]],
    infoset_dir: str,               # directory containing player_i_infosets.txt
    L: int,
    zip_path: str,
    meta_probs: np.ndarray | None = None,
    seed: int = 0,
):
    """
    Build team{TEAM}.zip as specified in leduc.pdf from TMECor marginals.

    For each j in 0..L-1, for each player in TEAM, we sample a pure
    (uncorrelated) behavioral strategy and save it as strategy{j}-player{i}.npy.
    """
    assert team in ("13", "24")
    players = [int(ch) for ch in team]

    if meta_probs is None:
        meta_probs = np.full(L, 1.0 / L, dtype=np.float64)
    else:
        assert len(meta_probs) == L
        meta_probs = np.asarray(meta_probs, dtype=np.float64)
        meta_probs = meta_probs / meta_probs.sum()

    rng = np.random.default_rng(seed)

    with zipfile.ZipFile(zip_path, "w") as zf:
        # meta-strategy.csv
        with zf.open("meta-strategy.csv", "w") as f:
            for j in range(L):
                p_j = float(meta_probs[j])
                f.write(f"{j},{p_j}\n".encode("utf-8"))

        # strategies strategy{j}-player{i}.npy
        for j in range(L):
            for player in players:
                infoset_path = os.path.join(infoset_dir, f"player_{player}_infosets.txt")
                tensor = sample_pure_player_tensor(infoset_path, marg, rng)
                fname = f"strategy{j}-player{player}.npy"
                with zf.open(fname, "w") as f:
                    np.save(f, tensor)



def main():
    import argparse

    ap = argparse.ArgumentParser(description="TB-DAG solvers (DCFR or PSRO) for 2-team games (e.g., team 13 vs 24).")
    ap.add_argument("--game-file", type=str, default="leduc_tree.txt",
                    help="Path to the EFG game file (e.g., leduc_tree.txt).")
    ap.add_argument("--infoset-dir", type=str, default="leduc-infosets",
                    help="Directory containing player_i_infosets.txt files.")
    ap.add_argument("--algo", choices=["dcfr", "psro"], default="psro",
                    help="Which solver to run. 'dcfr' replicates your current code; 'psro' runs policy-space response oracles.")
    ap.add_argument("--dcfr-iters", type=int, default=2000)
    ap.add_argument("--psro-iters", type=int, default=10)
    ap.add_argument("--meta-iters", type=int, default=3000,
                    help="Meta-solver iterations (regret matching) per PSRO step.")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out13", type=str, default="team13.zip")
    ap.add_argument("--out24", type=str, default="team24.zip")
    ap.add_argument("--log-every", type=int, default=1)

    args = ap.parse_args()

    efg = LeducEFG()
    efg.parse_game_file(args.game_file)
    validate_efg(efg)

    conn13 = ConnectivityGraph(efg, {1, 3})
    conn24 = ConnectivityGraph(efg, {2, 4})

    dag13 = TeamBeliefDAG(efg, conn13, {1, 3})
    dag24 = TeamBeliefDAG(efg, conn24, {2, 4})

    cfg = TwoTeamDCFRConfig(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    if args.algo == "dcfr":
        rm13, rm24, terminal_hists, u_coeff = run_two_team_dcfr(
            efg=efg,
            dag_team1=dag13,
            dag_team2=dag24,
            team1_players={1, 3},
            iterations=args.dcfr_iters,
            config=cfg,
            log_every=args.log_every,
        )
        marg13 = compute_infoset_action_marginals_from_tmecor(rm13)
        marg24 = compute_infoset_action_marginals_from_tmecor(rm24)

        # keep previous behavior: sample L strategies from marginals with uniform meta
        L = 100
        export_team_zip_from_tmecor("13", marg13, args.infoset_dir, L=L, zip_path=args.out13, seed=args.seed)
        export_team_zip_from_tmecor("24", marg24, args.infoset_dir, L=L, zip_path=args.out24, seed=args.seed + 1)
        print(f"[dcfr] wrote {args.out13} and {args.out24}")

    else:
        rm13, rm24, pop13, pop24, meta13, meta24, terminal_hists, u_coeff, M = run_two_team_psro(
            efg=efg,
            dag_team1=dag13,
            dag_team2=dag24,
            team1_players={1, 3},
            psro_iters=args.psro_iters,
            meta_iters=args.meta_iters,
            seed=args.seed,
            log_every=args.log_every,
            config=cfg,
        )

        # Export each PSRO population policy as a separate strategy.
        export_team_zip_from_policies("13", rm13, pop13, meta13, args.infoset_dir, args.out13, seed=args.seed)
        export_team_zip_from_policies("24", rm24, pop24, meta24, args.infoset_dir, args.out24, seed=args.seed + 1)

        print(f"[psro] wrote {args.out13} (L={len(pop13)}) and {args.out24} (L={len(pop24)})")


if __name__ == "__main__":
    main()

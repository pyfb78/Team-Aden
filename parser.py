"""
Minimal EFG Parser for Leduc Poker
Designed for TB-DAG construction with only essential fields
"""

from typing import Dict, List, Set, Optional, Tuple, FrozenSet
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import zipfile
import io

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

efg = LeducEFG()
efg.parse_game_file("leduc_tree.txt")
validate_efg(efg)

conn13 = ConnectivityGraph(efg, {1, 3})
print('Graph 1 Built')
#conn24 = ConnectivityGraph(efg, {2, 4})
print('Graph 2 Built')

dag13 = TeamBeliefDAG(efg, conn13, {1, 3})
print('DAG 1 Built')
#dag24 = TeamBeliefDAG(efg, conn24, {2, 4})
print('DAG 2 Built')

for i in range(10):
    print(dag13.active_nodes[i])
    print(dag13.inactive_nodes[i])




import parser
import pickle
from pathlib import Path
from collections import defaultdict

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, FrozenSet, Iterable, Optional

try:
    from parser import Game, Node, Infoset
except Exception:
    Game = object
    Node = object
    Infoset = object


BeliefLabel = FrozenSet[str]
ObsLabel = FrozenSet[str]
PrescriptionKey = Tuple[Tuple[str, str], ...]  # (infoset_name, action)


@dataclass
class BeliefNode:
    id: int
    label: BeliefLabel
    is_terminal: bool = False
    # map: prescription -> obs-node id
    children: Dict[PrescriptionKey, int] = field(default_factory=dict)


@dataclass
class ObsNode:
    id: int
    label: ObsLabel
    # list of belief-node ids (one per public observation)
    children: List[int] = field(default_factory=list)


class TeamBeliefDAG:
    """
    Team Belief DAG construction for games parsed by parser.Game.

    This implements Algorithm 1 (MAKEACTIVENODE / MAKEINACTIVENODE)
    from Zhang-Farina-Sandholm (2023).
    """

    def __init__(self, game: "Game", team_players: Iterable[int]):
        self.game: Game = game
        self.team_players: Set[int] = set(team_players)

        # game histories (node paths)
        self.H: List[str] = list(self.game.nodes.keys())
        if not self.H:
            raise ValueError("Game has no nodes; call Game.read_efg(...) first.")

        # tree structure
        self.parent: Dict[str, Optional[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.depth: Dict[str, int] = {}
        self.terminals: Set[str] = set()
        self._edge_child: Dict[Tuple[str, str], str] = {}  # (history, action) -> child history

        # infosets belonging to the team
        self.team_infosets: Dict[str, Infoset] = {}

        # connectivity graph G on H
        self.conn_adj: Dict[str, Set[str]] = {}
        self._anc_infosets: Dict[str, Set[str]] = {}

        # TB-DAG internal storage
        self.beliefs: Dict[int, BeliefNode] = {}
        self.obs_nodes: Dict[int, ObsNode] = {}
        self._belief_index: Dict[BeliefLabel, int] = {}
        self._obs_index: Dict[ObsLabel, int] = {}
        self._next_id: int = 0

        self.layer_node_to_infoset = {}
        self.infoset_to_layer_nodes = defaultdict(list)

        # root history (use Game.order if available, else shortest path)
        if getattr(self.game, "order", None):
            self.root_history: str = self.game.order[0]
        else:
            self.root_history = min(self.H, key=len)

        # construct everything
        
        self._build_tree_structure()
        print("built structure")
        self._identify_team_infosets()
        print("identified team infosets")
        self._build_connectivity_graph()
        #self.load_conn_graph()
        print("built connectivity graph")
        self.root_belief_id: int = self._make_active_node(frozenset([self.root_history]))
        print("built root belief")
        

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def num_belief_nodes(self) -> int:
        return len(self.beliefs)

    def num_obs_nodes(self) -> int:
        return len(self.obs_nodes)

    def num_edges(self) -> int:
        e = 0
        for b in self.beliefs.values():
            e += len(b.children)
        for o in self.obs_nodes.values():
            e += len(o.children)
        return e

    # ------------------------------------------------------------------
    # Step 0: tree structure from EFG paths like:
    #   /
    #   /C:JJQQ
    #   /C:JJQQ/P1:C
    #   /C:JJQQ/P1:C/P2:R
    # ------------------------------------------------------------------

    def _build_tree_structure(self) -> None:
        """
        For this EFG format, node.path strings look like:

            /
            /C:JJQQ
            /C:JJQQ/P1:C
            /C:JJQQ/P1:C/P2:C
            /C:JJQQ/P1:C/P2:C/P3:F
            ...

        So:
          - The parent of every non-root history is obtained by stripping
            the last '/segment'.
          - The *action* that leads from parent -> child is the substring
            after the ':' in that last segment (e.g. 'JJQQ', 'C', 'R', ...).
        """
        self.children = {h: [] for h in self.H}
        self.parent = {h: None for h in self.H}
        self.terminals = {
            h for h in self.H
            if self.game.nodes[h].node_type == "leaf"
        }
        self._edge_child.clear()

        root = self.root_history

        for h in self.H:
            if h == root:
                # the root has no parent and no incoming action
                self.parent[h] = None
                continue

            # compute parent path by stripping the last segment
            parts = h.split("/")
            # examples:
            #   "/" -> ["", ""]
            #   "/C:JJQQ" -> ["", "C:JJQQ"]
            #   "/C:JJQQ/P1:C" -> ["", "C:JJQQ", "P1:C"]
            parent_parts = parts[:-1]
            if parent_parts == [""] or parent_parts == []:
                parent_path = root
            else:
                parent_path = "/".join(parent_parts)
            self.parent[h] = parent_path
            self.children.setdefault(parent_path, []).append(h)

            # infer the action label from the last segment
            last_seg = parts[-1]
            if ":" in last_seg:
                _, action = last_seg.split(":", 1)
            else:
                action = last_seg
            # register the edge (parent, action) -> h
            self._edge_child[(parent_path, action)] = h

        # depth via simple DFS/BFS from root
        self.depth.clear()

        def assign_depth(start: str) -> None:
            stack = [(start, 0)]
            while stack:
                node, d = stack.pop()
                if node in self.depth and self.depth[node] <= d:
                    continue
                self.depth[node] = d
                for ch in self.children.get(node, []):
                    stack.append((ch, d + 1))

        assign_depth(root)
        # in case there are disconnected nodes, assign depths to them too
        for h in self.H:
            if h not in self.depth:
                assign_depth(h)

    # ------------------------------------------------------------------
    # Step 1: team infosets & connectivity graph
    # ------------------------------------------------------------------

    def _identify_team_infosets(self) -> None:
        self.team_infosets = {}
        for name, infoset in self.game.infosets.items():
            if not infoset.nodes:
                continue
            some_hist = infoset.nodes[0]
            node = self.game.nodes[some_hist]
            if node.node_type == "player" and node.player in self.team_players:
                self.team_infosets[name] = infoset

    def _is_inactive_history(self, h: str) -> bool:
        node = self.game.nodes[h]
        if node.node_type != "player":
            return True  # chance node or leaf
        return node.player not in self.team_players

    def load_conn_graph(self) -> None:
        ckpt_dir = Path("tbdag_ckpts")
        self.conn_adj = {h: set() for h in self.H}
        self._anc_infosets = {h: set() for h in self.H}

        for name, infoset in self.team_infosets.items():
            for hist in infoset.nodes:
                curr = hist
                while curr is not None:
                    self._anc_infosets[curr].add(name)
                    curr = self.parent.get(curr)

        print("ancestors marked")

        nodes_by_depth: Dict[int, List[str]] = {}
        for h in self.H:
            d = self.depth.get(h, 0)
            nodes_by_depth.setdefault(d, []).append(h)

        with open(ckpt_dir / f"progress_depth_22.pkl", "rb") as f:
            state = pickle.load(f)

            self.layer_node_to_infoset = state["layer_node_to_infoset"]
            self.infoset_to_layer_nodes = defaultdict(list, state["infoset_to_layer_nodes"]) 

    def _build_connectivity_graph(self) -> None:
        # adjacency initialization
        self.conn_adj = {h: set() for h in self.H}
        # ancestor -> infoset-name map
        self._anc_infosets = {h: set() for h in self.H}

        # mark all ancestors of each node in each team infoset
        for name, infoset in self.team_infosets.items():
            for hist in infoset.nodes:
                curr = hist
                while curr is not None and curr in self._anc_infosets:
                    self._anc_infosets[curr].add(name)
                    curr = self.parent.get(curr)

        print("ancestors marked")

        # for each depth, connect nodes that can both reach some team infoset
        nodes_by_depth: Dict[int, List[str]] = {}
        for h in self.H:
            d = self.depth.get(h, 0)
            nodes_by_depth.setdefault(d, []).append(h)

        print("nodes by depth")

        # where to store checkpoints
        ckpt_dir = Path("tbdag_ckpts24")
        ckpt_dir.mkdir(exist_ok=True)

        last_depth = -1
        # with open(ckpt_dir / f"progress_depth_{last_depth}.pkl", "rb") as f:
        #     state = pickle.load(f)

        #     self.layer_node_to_infoset = state["layer_node_to_infoset"]
        #     self.infoset_to_layer_nodes = defaultdict(list, state["infoset_to_layer_nodes"])   

        for d, nodes_d in nodes_by_depth.items():
            if d <= last_depth:
                continue
            for infoset_name in self.team_infosets.keys():
                #layer_nodes = [h for h in nodes_d if infoset_name in self._anc_infosets[h]]
                # for node in layer_nodes:
                #     self.layer_node_to_infoset[node] = infoset_name
                #     self.infoset_to_layer_nodes[infoset_name].append(node)
                for node in nodes_d:
                    if infoset_name in self._anc_infosets[node]:
                        self.layer_node_to_infoset[node] = infoset_name
                        self.infoset_to_layer_nodes[infoset_name].append(node)



                # if len(layer_nodes) <= 1:
                #     continue
                # for i in range(len(layer_nodes)):
                #     hi = layer_nodes[i]
                #     for j in range(i + 1, len(layer_nodes)):
                #         hj = layer_nodes[j]
                #         self.conn_adj[hi].add(hj)
                #         self.conn_adj[hj].add(hi)

            #Pickle at each depth for retrieval later
            checkpoint = {
                "depth": d,
                "layer_node_to_infoset": self.layer_node_to_infoset,
                "infoset_to_layer_nodes": self.infoset_to_layer_nodes,
            }

            ckpt_path = ckpt_dir / f"progress_depth_{d}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("connectivity graph built")

    # ------------------------------------------------------------------
    # Step 2: TB-DAG construction (MAKEACTIVENODE / MAKEINACTIVENODE)
    # ------------------------------------------------------------------

    # def _connected_components_subset(self, subset: ObsLabel) -> List[Set[str]]:
    #     O = set(subset)
    #     seen: Set[str] = set()
    #     comps: List[Set[str]] = []
    #     for start in O:
    #         if start in seen:
    #             continue
    #         comp = set()
    #         stack = [start]
    #         seen.add(start)
    #         while stack:
    #             v = stack.pop()
    #             comp.add(v)
    #             for nbr in self.infoset_to_layer_nodes[self.layer_node_to_infoset[v]]:
    #                 if nbr != v and nbr not in seen and nbr in O:
    #                     seen.add(nbr)
    #                     stack.append(nbr)
    #             # for nbr in self.conn_adj[v]:
    #             #     if nbr in O and nbr not in seen:
    #             #         seen.add(nbr)
    #             #         stack.append(nbr)
    #         comps.append(comp)
    #     return comps

    def _connected_components_subset(self, subset: ObsLabel) -> List[Set[str]]:
        O = set(subset)
        seen: Set[str] = set()
        comps: List[Set[str]] = []

        for start in O:
            if start in seen:
                continue

            # If start has no layer-node infoset, it cannot be connected
            # to anything via the team infosets: singleton component.
            if start not in self.layer_node_to_infoset:
                comps.append({start})
                seen.add(start)
                continue

            comp = set()
            stack = [start]
            seen.add(start)
            while stack:
                v = stack.pop()
                comp.add(v)

                # Nodes without layer_node_to_infoset have no neighbors.
                if v not in self.layer_node_to_infoset:
                    continue

                for nbr in self.infoset_to_layer_nodes[self.layer_node_to_infoset[v]]:
                    if nbr != v and nbr not in seen and nbr in O:
                        seen.add(nbr)
                        stack.append(nbr)

            comps.append(comp)

        return comps


    def _make_active_node(self, B: BeliefLabel) -> int:
        # reuse if already created
        if B in self._belief_index:
            return self._belief_index[B]

        node_id = self._next_id
        self._next_id += 1
        belief_node = BeliefNode(id=node_id, label=B)
        self.beliefs[node_id] = belief_node
        self._belief_index[B] = node_id

        # terminal singleton?
        if len(B) == 1:
            only = next(iter(B))
            if only in self.terminals:
                belief_node.is_terminal = True
                return node_id

        # infosets intersecting B
        infoset_names_in_B: Set[str] = set()
        for h in B:
            info_name = getattr(self.game, "hist_to_infoset", {}).get(h)
            if info_name is not None and info_name in self.team_infosets:
                infoset_names_in_B.add(info_name)
        infoset_names = sorted(infoset_names_in_B)

        # inactive nodes in B
        J = [h for h in B if self._is_inactive_history(h)]

        # precompute B ∩ I for each infoset
        infoset_nodes_in_B: Dict[str, List[str]] = {name: [] for name in infoset_names}
        for h in B:
            name = getattr(self.game, "hist_to_infoset", {}).get(h)
            if name in infoset_nodes_in_B:
                infoset_nodes_in_B[name].append(h)

        from itertools import product

        if infoset_names:
            # actions per infoset (all nodes in I share action set)
            actions_per_infoset: List[Tuple[str, List[str]]] = []
            for name in infoset_names:
                infoset = self.team_infosets[name]
                some_hist = infoset.nodes[0]
                node = self.game.nodes[some_hist]
                actions_per_infoset.append((name, list(node.actions)))

            infoset_order = [name for name, _ in actions_per_infoset]
            action_lists = [acts for _, acts in actions_per_infoset]

            for action_tuple in product(*action_lists):
                prescription = dict(zip(infoset_order, action_tuple))

                next_histories: Set[str] = set()

                # push team infoset nodes forward according to prescription
                for name, a in prescription.items():
                    for h in infoset_nodes_in_B[name]:
                        child = self._edge_child.get((h, a))
                        if child is not None:
                            next_histories.add(child)

                # push inactive nodes along all outgoing actions
                for h in J:
                    node = self.game.nodes[h]
                    if node.node_type in ("chance", "player"):
                        for a in node.actions:
                            child = self._edge_child.get((h, a))
                            if child is not None:
                                next_histories.add(child)

                if not next_histories:
                    continue

                Ba: ObsLabel = frozenset(next_histories)
                obs_id = self._make_inactive_node(Ba)
                key: PrescriptionKey = tuple(sorted(prescription.items()))
                belief_node.children[key] = obs_id
        else:
            # no team infosets intersect B: just push inactive nodes forward
            next_histories: Set[str] = set()
            for h in J:
                node = self.game.nodes[h]
                if node.node_type in ("chance", "player"):
                    for a in node.actions:
                        child = self._edge_child.get((h, a))
                        if child is not None:
                            next_histories.add(child)

            if next_histories:
                Ba: ObsLabel = frozenset(next_histories)
                obs_id = self._make_inactive_node(Ba)
                belief_node.children[tuple()] = obs_id  # empty prescription

        return node_id

    def _make_inactive_node(self, O: ObsLabel) -> int:
        if O in self._obs_index:
            return self._obs_index[O]

        node_id = self._next_id
        self._next_id += 1
        obs_node = ObsNode(id=node_id, label=O)
        self.obs_nodes[node_id] = obs_node
        self._obs_index[O] = node_id

        for comp in self._connected_components_subset(O):
            child_belief_id = self._make_active_node(frozenset(comp))
            obs_node.children.append(child_belief_id)

        return node_id


if __name__ == "__main__": 
    leduc = Game()
    leduc.read_efg("leduc_short.txt")
    dag = TeamBeliefDAG(leduc, team_players=[1, 3])
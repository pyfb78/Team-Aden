import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Iterable

from team_belief_dag import TeamBeliefDAG, BeliefNode, ObsNode  # type: ignore


@dataclass
class DAGForCFR:
    dag: TeamBeliefDAG
    num_nodes: int
    root: int

    belief_ids: List[int]
    obs_ids: List[int]

    is_belief: List[bool]
    is_terminal: List[bool]

    children: List[List[int]]
    parents: List[List[int]]

    num_actions: List[int]
    action_children: List[List[int]]

    terminal_nodes: List[int]
    terminal_hist: Dict[int, str]
    leaf_for_hist: Dict[str, int]

    topo_order: List[int]
    topo_active: List[int]

    active_nodes: List[int]
    active_index: Dict[int, int]

    @classmethod
    def from_team_dag(cls, dag: TeamBeliefDAG) -> "DAGForCFR":
        """
        Build a CFR-friendly view from a TeamBeliefDAG.

        Assumptions about TeamBeliefDAG:
          - dag._next_id: number of allocated node ids (0.._next_id-1)
          - dag.beliefs: Dict[int, BeliefNode]
              BeliefNode has fields: id, label (frozenset[str]),
              is_terminal: bool, children: Dict[prescription_key, obs_id]
          - dag.obs_nodes: Dict[int, ObsNode]
              ObsNode has fields: id, label (frozenset[str]),
              children: List[belief_id]
          - dag.root_belief_id: int
        """
        num_nodes = dag._next_id
        belief_ids = sorted(dag.beliefs.keys())
        obs_ids = sorted(dag.obs_nodes.keys())
        root = dag.root_belief_id

        is_belief = [False] * num_nodes
        is_terminal = [False] * num_nodes
        children: List[List[int]] = [[] for _ in range(num_nodes)]
        parents: List[List[int]] = [[] for _ in range(num_nodes)]
        num_actions = [0] * num_nodes
        action_children: List[List[int]] = [[] for _ in range(num_nodes)]

        terminal_hist: Dict[int, str] = {}
        leaf_for_hist: Dict[str, int] = {}

        # Belief nodes (active/terminal)
        for bid, belief in dag.beliefs.items():
            is_belief[bid] = True

            if getattr(belief, "is_terminal", False):
                is_terminal[bid] = True
                if len(belief.label) == 1:
                    (h,) = tuple(belief.label)
                    terminal_hist[bid] = h
                    leaf_for_hist[h] = bid

            # Outgoing edges: one action per prescription key
            if belief.children:
                keys = list(belief.children.keys())
                keys.sort()  # deterministic ordering
                num_actions[bid] = len(keys)

                node_action_children: List[int] = []
                for key in keys:
                    obs_id = belief.children[key]
                    node_action_children.append(obs_id)

                    if obs_id not in children[bid]:
                        children[bid].append(obs_id)
                        parents[obs_id].append(bid)

                action_children[bid] = node_action_children

        # Obs nodes (inactive)
        for oid, obs in dag.obs_nodes.items():
            for child_belief in obs.children:
                children[oid].append(child_belief)
                parents[child_belief].append(oid)

        # Topological order over all nodes
        indeg = [0] * num_nodes
        for v in range(num_nodes):
            for w in children[v]:
                indeg[w] += 1

        from collections import deque
        q = deque()
        for v in range(num_nodes):
            if indeg[v] == 0:
                q.append(v)

        topo_order: List[int] = []
        while q:
            v = q.popleft()
            topo_order.append(v)
            for w in children[v]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)

        # Active nodes = belief nodes with at least one action
        active_nodes = [
            v for v in topo_order if is_belief[v] and num_actions[v] > 0
        ]
        active_index: Dict[int, int] = {s: i for i, s in enumerate(active_nodes)}
        topo_active = active_nodes[:]  # already in topo order

        terminal_nodes = [v for v in range(num_nodes) if is_terminal[v]]

        return cls(
            dag=dag,
            num_nodes=num_nodes,
            root=root,
            belief_ids=belief_ids,
            obs_ids=obs_ids,
            is_belief=is_belief,
            is_terminal=is_terminal,
            children=children,
            parents=parents,
            num_actions=num_actions,
            action_children=action_children,
            terminal_nodes=terminal_nodes,
            terminal_hist=terminal_hist,
            leaf_for_hist=leaf_for_hist,
            topo_order=topo_order,
            topo_active=topo_active,
            active_nodes=active_nodes,
            active_index=active_index,
        )


class DagCFRPlayer:
    """
    CFR for a single team on its TB-DAG.

    - next_strategy(): forward pass with regret matching, compute reach x[s]
    - observe_utility(leaf_util): backward pass, update regrets
    - average_strategy(): get average local policy at each active node
    """

    def __init__(self, dag_view: DAGForCFR):
        self.dag = dag_view
        self.num_nodes = dag_view.num_nodes
        self.active_nodes = dag_view.active_nodes
        self.active_index = dag_view.active_index
        self.num_active = len(self.active_nodes)

        # Regrets and strategy sums indexed by active-node index
        self.regret: List[List[float]] = []
        self.strategy_sum: List[List[float]] = []
        self.current_policy: List[List[float]] = []

        for s in self.active_nodes:
            k = self.dag.num_actions[s]
            self.regret.append([0.0] * k)
            self.strategy_sum.append([0.0] * k)
            self.current_policy.append([0.0] * k)

        # Reach x[s] (realization weights for THIS team)
        self.reach: List[float] = [0.0] * self.num_nodes

    # -------------------------------
    # Forward pass: compute strategy
    # -------------------------------
    def next_strategy(self) -> List[float]:
        """
        Run regret-matching at each active node, propagate reach x[s]
        from the root along the DAG, and update strategy_sum.

        Returns:
            reach: list of length num_nodes, reach[s] for each DAG node s.
        """
        N = self.num_nodes
        dag = self.dag

        # Reset reach
        reach = [0.0] * N
        reach[dag.root] = 1.0

        # Reset current policy
        for idx, s in enumerate(self.active_nodes):
            k = dag.num_actions[s]
            if len(self.current_policy[idx]) != k:
                self.current_policy[idx] = [0.0] * k
            else:
                for a in range(k):
                    self.current_policy[idx][a] = 0.0

        # Topological sweep
        for s in dag.topo_order:
            if dag.is_belief[s] and dag.num_actions[s] > 0:
                # Decision node
                idx = self.active_index[s]
                k = dag.num_actions[s]

                # Regret-matching
                pos_sum = 0.0
                for a in range(k):
                    r = self.regret[idx][a]
                    if r > 0:
                        pos_sum += r

                if pos_sum <= 0.0:
                    # Uniform
                    for a in range(k):
                        self.current_policy[idx][a] = 1.0 / k
                else:
                    for a in range(k):
                        r = self.regret[idx][a]
                        self.current_policy[idx][a] = max(r, 0.0) / pos_sum

                # Flow to obs children; accumulate strategy_sum
                for a in range(k):
                    child_obs = dag.action_children[s][a]
                    prob = self.current_policy[idx][a]
                    flow = reach[s] * prob
                    reach[child_obs] += flow
                    self.strategy_sum[idx][a] += flow

            else:
                # Obs node or terminal belief: just copy flow forward
                for child in dag.children[s]:
                    reach[child] += reach[s]

        self.reach = reach
        return reach

    # -------------------------------
    # Backward pass: update regrets
    # -------------------------------
    def observe_utility(self, leaf_util: Dict[int, float]) -> None:
        """
        Given terminal utilities for this team at each DAG leaf node,
        run the backward pass and update regrets.

        leaf_util[s_leaf] is assumed to already be multiplied by
        (opponent × chance reach), i.e., counterfactual utility.
        """
        N = self.num_nodes
        dag = self.dag

        u: List[float] = [0.0] * N

        # Initialize leaves
        for leaf_id, val in leaf_util.items():
            u[leaf_id] += val

        # Reverse topo sweep
        for s in reversed(dag.topo_order):
            if dag.is_belief[s] and dag.num_actions[s] > 0:
                idx = self.active_index[s]
                k = dag.num_actions[s]

                # Expected utility at s under current policy
                v_s = 0.0
                for a in range(k):
                    child_obs = dag.action_children[s][a]
                    u_sa = u[child_obs]
                    v_s += self.current_policy[idx][a] * u_sa

                # Regret updates
                for a in range(k):
                    child_obs = dag.action_children[s][a]
                    u_sa = u[child_obs]
                    self.regret[idx][a] += u_sa - v_s

                # Value for upstream
                u[s] += v_s

            # Propagate to parents (works for both belief and obs)
            for p in dag.parents[s]:
                u[p] += u[s]

    # -------------------------------
    # Average strategy
    # -------------------------------
    def average_strategy(self) -> Dict[int, List[float]]:
        """
        Return average local policy at each active node s:
        dict: s -> [prob(action 0), prob(action 1), ...]
        aligned with dag.action_children[s].
        """
        avg: Dict[int, List[float]] = {}
        dag = self.dag

        for idx, s in enumerate(self.active_nodes):
            k = dag.num_actions[s]
            if k == 0:
                continue
            total = sum(self.strategy_sum[idx][:k])
            if total <= 0.0:
                # Never visited; default to uniform
                avg[s] = [1.0 / k] * k
            else:
                avg[s] = [
                    self.strategy_sum[idx][a] / total
                    for a in range(k)
                ]

        return avg

def compute_chance_prob_for_history(game, hist: str) -> float:
    """
    Given a Game and a terminal history path string like
    "/C:JJQQ/P1:C/P2:C/.../C:K/...",
    compute the product of chance action probabilities along the path.

    Assumes:
      - Root path is "/"
      - game.nodes[path] -> Node
      - Node.node_type == "chance" means this is a chance node
      - Node.action_probs maps action string -> probability
      - Child paths encode the action as "C:{action}" for chance,
        "P1:{action}" etc for players.
    """
    if hist not in game.nodes:
        raise KeyError(f"History {hist} not found in game.nodes")

    # Split path into segments
    # Example: "/C:JJQQ/P1:C/P2:C" -> ["C:JJQQ", "P1:C", "P2:C"]
    segments = hist.strip("/").split("/")
    prefix = "/"  # start at root
    prob = 1.0

    for seg in segments:
        # Node BEFORE taking seg
        if prefix not in game.nodes:
            raise KeyError(f"Prefix {prefix} not found in game.nodes")

        node: Node = game.nodes[prefix]
        if node.node_type == "chance":
            # seg encodes "C:action" (we ignore left side)
            if ":" not in seg:
                raise ValueError(f"Expected 'C:action' segment after chance node, got {seg}")
            _, action = seg.split(":", 1)
            if action not in node.action_probs:
                raise KeyError(f"Action {action} not in action_probs for chance node {prefix}")
            prob *= node.action_probs[action]

        # Move to child prefix
        if prefix == "/":
            prefix = "/" + seg
        else:
            prefix = prefix + "/" + seg

    return prob


def payoff_to_team13(game, hist: str) -> float:
    """
    Payoff to team {1,3} at terminal history 'hist'.

    Assumes:
      - game.nodes[hist].node_type == "leaf"
      - node.payoffs: Dict[int, float] with 1-based player indices.
    """
    if hist not in game.nodes:
        raise KeyError(f"History {hist} not found in game.nodes")
    node: Node = game.nodes[hist]
    if node.node_type != "leaf":
        raise ValueError(f"History {hist} is not a leaf node (type {node.node_type})")

    payoffs = node.payoffs or {}
    # Adjust these if your player numbering is different:
    u1 = payoffs.get(1, 0.0)
    u3 = payoffs.get(3, 0.0)
    return u1 + u3

class TwoTeamCFREngine:
    """
    CFR engine for a two-team zero-sum game:

        Team 1 = {1,3} -> dag_team1
        Team 2 = {2,4} -> dag_team2

    Underlying extensive-form game (with chance) is in dag_team1.game.
    """

    def __init__(self, dag_team1: TeamBeliefDAG, dag_team2: TeamBeliefDAG):
        # Wrap TB-DAGs for CFR
        self.view1 = DAGForCFR.from_team_dag(dag_team1)
        print("CFR view 1 built")
        self.view2 = DAGForCFR.from_team_dag(dag_team2)
        print("CFR view 2 built")

        self.player1 = DagCFRPlayer(self.view1)
        print("Player 1 built")
        self.player2 = DagCFRPlayer(self.view2)
        print("Player 2 built")

        # Underlying game (we assume both DAGs built from the same tree)
        self.game = dag_team1.game

        # Terminal histories as strings (paths)
        self.term_histories: List[str] = sorted(self.view1.leaf_for_hist.keys())

        # Precompute chance reach at each terminal history
        self.chance_prob: Dict[str, float] = {
            h: compute_chance_prob_for_history(self.game, h)
            for h in self.term_histories
        }

    def iterate(self, num_iters: int, verbose_every: int = 0) -> None:
        """
        Run num_iters iterations of CFR.
        """
        for t in range(1, num_iters + 1):
            # 1) Each team computes its current strategy & reach
            reach1 = self.player1.next_strategy()
            reach2 = self.player2.next_strategy()

            # 2) Build counterfactual utilities at leaves for each team
            leaf_util_1: Dict[int, float] = {}
            leaf_util_2: Dict[int, float] = {}

            for h in self.term_histories:
                s1 = self.view1.leaf_for_hist[h]
                s2 = self.view2.leaf_for_hist[h]

                pi1 = reach1[s1]  # Team 1 realization reach
                pi2 = reach2[s2]  # Team 2 realization reach
                pi_c = self.chance_prob[h]

                payoff = payoff_to_team13(self.game, h)  # payoff to Team 1

                # Team 1 counterfactual utility: u1(z) * (chance × opponent reach)
                cf1 = payoff * pi2 * pi_c
                leaf_util_1[s1] = leaf_util_1.get(s1, 0.0) + cf1

                # Team 2 counterfactual utility: -u1(z) * (chance × Team1 reach)
                cf2 = -payoff * pi1 * pi_c
                leaf_util_2[s2] = leaf_util_2.get(s2, 0.0) + cf2

            # 3) Backward pass on both TB-DAGs
            self.player1.observe_utility(leaf_util_1)
            self.player2.observe_utility(leaf_util_2)

            if verbose_every and t % verbose_every == 0:
                print(f"[CFREngine] Completed iteration {t}")
                with open("team13cfr.pickle", "wb") as f:
                    pickle.dump(self.player1, f)
                with open("team24cfr.pickle", "wb") as f:
                    pickle.dump(self.player2, f)

    def average_strategies(self) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
        """
        Returns:
            (avg_strategy_team1, avg_strategy_team2)
        Each is: dict node_id -> list of action probs at that belief node.
        """
        avg1 = self.player1.average_strategy()
        avg2 = self.player2.average_strategy()
        return avg1, avg2

def main():
    # Load the two TB-DAGs (filenames can be changed if needed)
    # dag13: TeamBeliefDAG = pickle.load(open("TeamBeliefDag.pickle", "rb"))
    # print("Dag 13 pickled")
    # dag24: TeamBeliefDAG = pickle.load(open("TeamBeliefDag24.pickle", "rb"))
    # print("Dag 24 pickled")

    # engine = TwoTeamCFREngine(dag_team1=dag13, dag_team2=dag24)
    # with open("CFREngine.pickle", "wb") as f:
    #     pickle.dump(engine, f)
    with open("CFREngine.pickle", "rb") as f:
        engine = pickle.load(f)

    print("Engine Built")

    
    num_iters = 1000000  # tweak as needed
    engine.iterate(num_iters=num_iters, verbose_every=500)

    avg1, avg2 = engine.average_strategies()

    # print("Average strategy for Team {1,3}:")
    # for s, probs in sorted(avg1.items()):
    #     print(f"  belief node {s}: {probs}")

    # print("\nAverage strategy for Team {2,4}:")
    # for s, probs in sorted(avg2.items()):
    #     print(f"  belief node {s}: {probs}")


if __name__ == "__main__":
    main()

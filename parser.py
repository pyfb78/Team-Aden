from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Node:
    """Represents a node in the game tree."""
    path: str
    node_type: str  # "chance", "player", or "leaf"
    player: Optional[int] = None  # For player nodes
    actions: List[str] = None  # For chance and player nodes
    action_probs: Dict[str, float] = None  # For chance nodes
    payoffs: Dict[int, float] = None  # For leaf nodes

    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.action_probs is None:
            self.action_probs = {}
        if self.payoffs is None:
            self.payoffs = {}


@dataclass
class Infoset:
    """Represents an information set."""
    name: str
    nodes: List[str]  # List of node paths in this infoset


class Game:
    def __init__(self, players=None):
        self.players = players or [] # List of players [P1, P2, P3, P4]
        self.hist_to_infoset = {} # Maps from game tree nodes to infoset encoding
        self.infosets = {} # Maps from infoset encoding to infoset object
        self.order = [] # List of game tree nodes in top-down order
        self.nodes = {} # Maps from node path to Node object

    def read_efg(self, path: str):
        """
        Reads an EFG file and returns a Game object

        Args:
            path (str): Path to the EFG file

        Returns:
            None

        Reads an extensive form game representation from leduc_tree.txt and populates the game object with the game tree nodes,
        the players, the infosets, and the order of the game tree nodes.
        """
        players_set = set()

        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse infoset lines
                if line.startswith("infoset "):
                    content = line[8:]  # Remove "infoset " prefix
                    parts = content.split(" nodes ", 1)
                    if len(parts) != 2:
                        raise ValueError(f"Line {line_num}: Invalid infoset format")

                    infoset_name = parts[0]
                    node_paths = parts[1].split()

                    # Create infoset object
                    infoset = Infoset(name=infoset_name, nodes=node_paths)
                    self.infosets[infoset_name] = infoset

                    # Map each node to this infoset
                    for node_path in node_paths:
                        self.hist_to_infoset[node_path] = infoset_name

                # Parse node lines
                elif line.startswith("node "):
                    content = line[5:]  # Remove "node " prefix

                    # Split into path and rest
                    parts = content.split(maxsplit=1)
                    if len(parts) < 2:
                        raise ValueError(f"Line {line_num}: Invalid format, expected path and node info")

                    node_path = parts[0]
                    rest = parts[1]

                    # Add to order
                    self.order.append(node_path)

                    # Parse based on node type
                    if rest.startswith("chance actions "):
                        # Chance node
                        actions_str = rest[15:]  # Skip "chance actions "
                        action_probs = {}
                        actions = []

                        for action_prob in actions_str.split():
                            if '=' not in action_prob:
                                raise ValueError(f"Line {line_num}: Invalid action format: {action_prob}")
                            action, prob_str = action_prob.split('=', 1)
                            actions.append(action)
                            action_probs[action] = float(prob_str)

                        node = Node(
                            path=node_path,
                            node_type="chance",
                            actions=actions,
                            action_probs=action_probs
                        )
                        self.nodes[node_path] = node

                    elif rest.startswith("leaf payoffs "):
                        # Leaf node
                        payoffs_str = rest[13:]  # Skip "leaf payoffs "
                        payoffs = {}

                        for player_payoff in payoffs_str.split():
                            if '=' not in player_payoff:
                                raise ValueError(f"Line {line_num}: Invalid payoff format: {player_payoff}")
                            player_str, payoff_str = player_payoff.split('=', 1)
                            player_num = int(player_str)
                            payoffs[player_num] = float(payoff_str)
                            players_set.add(player_num)

                        node = Node(
                            path=node_path,
                            node_type="leaf",
                            payoffs=payoffs
                        )
                        self.nodes[node_path] = node

                    elif rest.startswith("player "):
                        # Player node
                        rest_parts = rest[7:].split(" actions ", 1)
                        if len(rest_parts) != 2:
                            raise ValueError(f"Line {line_num}: Invalid player node format")

                        player_num = int(rest_parts[0])
                        actions = rest_parts[1].split()

                        players_set.add(player_num)

                        node = Node(
                            path=node_path,
                            node_type="player",
                            player=player_num,
                            actions=actions
                        )
                        self.nodes[node_path] = node

                    else:
                        raise ValueError(f"Line {line_num}: Unknown node type in: {rest[:50]}")

                else:
                    raise ValueError(f"Line {line_num}: Expected line to start with 'node ' or 'infoset ', got: {line[:50]}")

        self.players = sorted(list(players_set))
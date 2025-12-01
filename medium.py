def write_medium_team_game(file_path):
    path = ""
    states = ['L', 'M', 'H']
    probs = {'L': 0.2, 'M': 0.5, 'H': 0.3}

    # Actions
    p1_actions = ['C', 'R']  # cautious / risky
    p2_actions = ['S', 'F']  # stop / fight
    p3_actions = ['X', 'Y']  # settle / escalate
    p4_actions = ['A', 'P']  # aggressive / passive

    lines = []

    # ---------- Infosets ----------
    # P1: does not observe the state S
    lines.append("infoset I1 nodes " + " ".join(f"/C:{s}" for s in states))

    # P2: sees P1 action but not S
    lines.append("infoset I2_C nodes " + " ".join(f"/C:{s}/P1:C" for s in states))
    lines.append("infoset I2_R nodes " + " ".join(f"/C:{s}/P1:R" for s in states))

    # P3: moves only after P2:F, sees P1 action but not S
    lines.append("infoset I3_C nodes " + " ".join(f"/C:{s}/P1:C/P2:F" for s in states))
    lines.append("infoset I3_R nodes " + " ".join(f"/C:{s}/P1:R/P2:F" for s in states))

    # P4: moves only after escalation (P3:Y), does not know S or P1 action
    lines.append(
        "infoset I4 nodes "
        + " ".join(f"/C:{s}/P1:{a1}/P2:F/P3:Y" for s in states for a1 in p1_actions)
    )

    # ---------- Nodes ----------
    # Root chance node
    lines.append(
        "node / chance actions " + " ".join(f"{s}={probs[s]}" for s in states)
    )

    # Player 1 nodes
    for s in states:
        lines.append(f"node /C:{s} player 1 actions C R")

    # Player 2 nodes (after P1)
    for s in states:
        for a1 in p1_actions:
            lines.append(f"node /C:{s}/P1:{a1} player 2 actions S F")

    # Player 3 nodes (after P2:F)
    for s in states:
        for a1 in p1_actions:
            lines.append(f"node /C:{s}/P1:{a1}/P2:F player 3 actions X Y")

    # Player 4 nodes (after P3:Y)
    for s in states:
        for a1 in p1_actions:
            lines.append(f"node /C:{s}/P1:{a1}/P2:F/P3:Y player 4 actions A P")

    # ---------- Payoffs ----------
    # Base values for team {1,3}; team {2,4} gets the negative
    base_P2S = {'L': -1, 'M': 0, 'H': 2}   # P2 stops
    base_FX  = {'L':  1, 'M': 1, 'H': 1}   # P3 settles (X)
    base_FYA = {'L': -3, 'M': -1, 'H': 1}  # P3 escalates (Y), P4 aggressive (A)
    base_FYP = {'L': -2, 'M':  1, 'H': 3}  # P3 escalates (Y), P4 passive (P)

    def mult(a1):
        # Risky action by P1 doubles magnitude of payoffs
        return 1 if a1 == 'C' else 2

    def add_leaf(path, v):
        u1 = u3 = v / 2.0
        u2 = u4 = -v / 2.0
        lines.append(
            f"node {path} leaf payoffs 1={u1} 2={u2} 3={u3} 4={u4}"
        )

    # Case 1: P2 stops (S)
    for s in states:
        for a1 in p1_actions:
            v = base_P2S[s] * mult(a1)
            path = f"/C:{s}/P1:{a1}/P2:S"
            add_leaf(path, v)

    # Case 2: P2 fights, P3 settles (X)
    for s in states:
        for a1 in p1_actions:
            v = base_FX[s] * mult(a1)
            path = f"/C:{s}/P1:{a1}/P2:F/P3:X"
            add_leaf(path, v)

    # Case 3: P2 fights, P3 escalates (Y), P4 aggressive (A)
    for s in states:
        for a1 in p1_actions:
            v = base_FYA[s] * mult(a1)
            path = f"/C:{s}/P1:{a1}/P2:F/P3:Y/P4:A"
            add_leaf(path, v)

    # Case 4: P2 fights, P3 escalates (Y), P4 passive (P)
    for s in states:
        for a1 in p1_actions:
            v = base_FYP[s] * mult(a1)
            path = f"/C:{s}/P1:{a1}/P2:F/P3:Y/P4:P"
            add_leaf(path, v)

    # Write to file
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

write_medium_team_game("medium.txt")
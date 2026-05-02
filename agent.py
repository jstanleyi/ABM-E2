from mesa import Agent


class AudienceAgent(Agent):
    """
    Audience member in the Standing Ovation model.

    Initial rule:
    - Stand if private perceived quality >= threshold.

    Later rule:
    - Follow the majority of visible neighbors.
    - Visibility uses a cone-shaped field of view.
    """

    def __init__(self, model, pos, threshold=0.5):
        super().__init__(model)
        self.pos = pos
        self.threshold = threshold
        self.quality = self.random.random()

        # Initial decision based on private quality signal
        self.standing = self.quality >= self.threshold
        self.next_standing = self.standing

    def get_visible_neighbors(self):
        """
        Cone visibility rule.

        Coordinate convention:
        - y = 0 is the front row.
        - Larger y means farther back.
        - Agent sees left/right neighbors and a widening cone in front.
        """
        x, y = self.pos
        visible_positions = []

        # Same-row left and right neighbors
        for dx in [-1, 1]:
            nx = x + dx
            if 0 <= nx < self.model.width:
                visible_positions.append((nx, y))

        # Cone in front
        # distance 1: x-1, x, x+1
        # distance 2: x-2, ..., x+2
        # etc.
        for distance in range(1, y + 1):
            visible_y = y - distance
            for visible_x in range(x - distance, x + distance + 1):
                if 0 <= visible_x < self.model.width:
                    visible_positions.append((visible_x, visible_y))

        neighbors = []
        for position in visible_positions:
            neighbors.extend(self.model.grid.get_cell_list_contents([position]))

        return neighbors

    def majority_decision(self):
        """
        Majority rule.

        If irreversible_diffusion is False:
        - Agents can switch between standing and sitting.

        If irreversible_diffusion is True:
        - Once an agent stands, the agent remains standing.
        """
        # Extension: if irreversible diffusion is turned on, standing is absorbing.
        # Once standing, the agent cannot sit down again.
        if self.model.irreversible_diffusion and self.standing:
            return True

        neighbors = self.get_visible_neighbors()

        # Edge case:
        # Front-row agents may have very few visible neighbors. The paper does
        # not specify a precise tie/no-neighbor rule, so I use random choice.
        if len(neighbors) == 0:
            return self.random.choice([True, False])

        standing_count = sum(neighbor.standing for neighbor in neighbors)
        sitting_count = len(neighbors) - standing_count

        if standing_count > sitting_count:
            return True

        if sitting_count > standing_count:
            return False

        # Tie case
        # The paper states that ties can be handled randomly, so this is a
        # reasonable implementation choice.
        return self.random.choice([True, False])

    def compute_next_state(self):
        """Used for synchronous updating."""
        self.next_standing = self.majority_decision()

    def advance(self):
        """Apply synchronous update."""
        self.standing = self.next_standing

    def update_now(self):
        """Used for asynchronous updating."""
        self.standing = self.majority_decision()

    def mismatch_score(self):
        """
        Incentive-based asynchronous updating.

        Agents least like their visible neighbors update first.
        """
        neighbors = self.get_visible_neighbors()

        if len(neighbors) == 0:
            return 0

        different_count = sum(
            neighbor.standing != self.standing for neighbor in neighbors
        )

        return different_count / len(neighbors)
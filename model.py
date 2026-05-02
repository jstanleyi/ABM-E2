from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agent import AudienceAgent


class StandingOvationModel(Model):
    """
    Mesa implementation of the Standing Ovation model.

    Includes:
    - Cone visibility
    - Synchronous updating
    - Random asynchronous updating
    - Incentive-based asynchronous updating
    - Optional irreversible diffusion
    """

    def __init__(
        self,
        *,
        width=20,
        height=20,
        threshold=0.75,
        update_rule="random_async",
        irreversible_diffusion=False,
        seed=None,
    ):
        super().__init__(seed=seed)

        # The paper's computational example uses a square auditorium with
        # 400 seats. The default 20 x 20 grid reproduces that baseline size,
        # while the GUI allows the user to vary it.
        self.width = width
        self.height = height

        # The paper uses a threshold of 0.5 in one formal example.
        # I expose threshold as a parameter because changing the initial
        # standing proportion is important for exploring model dynamics.
        self.threshold = threshold

        # This parameter selects among the three update procedures discussed
        # in the paper.
        self.update_rule = update_rule

        # Extension: when True, standing becomes irreversible.
        # This is included to compare coordination dynamics with diffusion
        # dynamics and to examine the conditions under which S-shaped curves
        # emerge more clearly.
        self.irreversible_diffusion = irreversible_diffusion

        # Non-torus grid because an auditorium does not wrap around at edges.
        self.grid = MultiGrid(width, height, torus=False)
        self.audience = []
        self.step_count = 0

        # Create one audience member per seat.
        # y = 0 is treated as the front row; larger y values are farther back.
        for y in range(height):
            for x in range(width):
                agent = AudienceAgent(
                    model=self,
                    pos=(x, y),
                    threshold=threshold,
                )
                self.grid.place_agent(agent, (x, y))
                self.audience.append(agent)

        # Store initial majority for informational efficiency.
        # The paper defines informational efficiency across simulation runs.
        # In this GUI version, I track whether the current majority matches the
        # initial majority within a single run.
        self.initial_standing_count = self.count_standing()
        self.initial_majority_standing = (
            self.initial_standing_count > self.total_agents() / 2
        )

        self.datacollector = DataCollector(
            model_reporters={
                # Main diffusion/coordination outcome: number standing over time.
                "Standing": lambda m: m.count_standing(),

                # Same information as a proportion, which makes runs with
                # different grid sizes easier to compare.
                "PercentStanding": lambda m: m.percent_standing(),

                # Paper-inspired metric: local disagreement with visible majority.
                "StickInMuds": lambda m: m.compute_stick_in_muds(),

                # Paper-inspired metric adapted for a single GUI run.
                "InformationalEfficiency": lambda m: m.compute_information_efficiency(),
            }
        )

        self.datacollector.collect(self)

    def total_agents(self):
        return len(self.audience)

    def count_standing(self):
        return sum(agent.standing for agent in self.audience)

    def percent_standing(self):
        return self.count_standing() / self.total_agents()

    def compute_stick_in_muds(self):
        """
        Share of agents whose current action differs from the local majority.
        """
        stick_count = 0
        comparable_count = 0

        for agent in self.audience:
            neighbors = agent.get_visible_neighbors()

            if len(neighbors) == 0:
                continue

            standing_count = sum(neighbor.standing for neighbor in neighbors)
            sitting_count = len(neighbors) - standing_count

            # If there is no local majority, the agent is not counted as either
            # agreeing or disagreeing with the local majority.
            if standing_count == sitting_count:
                continue

            local_majority_standing = standing_count > sitting_count
            comparable_count += 1

            if agent.standing != local_majority_standing:
                stick_count += 1

        if comparable_count == 0:
            return 0

        return stick_count / comparable_count

    def compute_information_efficiency(self):
        """
        Whether the current majority matches the initial majority.
        """
        current_majority_standing = self.count_standing() > self.total_agents() / 2
        return int(current_majority_standing == self.initial_majority_standing)

    def step(self):
        if self.update_rule == "synchronous":
            self.step_synchronous()
        elif self.update_rule == "random_async":
            self.step_random_async()
        elif self.update_rule == "incentive_async":
            self.step_incentive_async()
        else:
            raise ValueError(f"Unknown update rule: {self.update_rule}")

        self.step_count += 1
        self.datacollector.collect(self)

    def step_synchronous(self):
        """
        All agents compute from the same previous state, then update together.
        """
        for agent in self.audience:
            agent.compute_next_state()

        for agent in self.audience:
            agent.advance()

    def step_random_async(self):
        """
        Agents update one at a time in random order.
        """
        agents = list(self.audience)
        self.random.shuffle(agents)

        for agent in agents:
            agent.update_now()

    def step_incentive_async(self):
        """
        Agents least like their visible neighbors update first.
        """
        agents = sorted(
            self.audience,
            key=lambda agent: agent.mismatch_score(),
            reverse=True,
        )

        for agent in agents:
            agent.update_now()
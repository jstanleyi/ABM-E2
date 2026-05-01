from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from mesa.visualization import Slider

from model import StandingOvationModel


def agent_portrayal(agent):
    """
    Visual display:
    - Standing agents are black.
    - Sitting agents are light gray.
    """
    if agent.standing:
        return {
            "color": "black",
            "size": 80,
            "marker": "s",
        }

    return {
        "color": "lightgray",
        "size": 80,
        "marker": "s",
    }


model_params = {
    "width": Slider("Width", value=20, min=5, max=50, step=1),
    "height": Slider("Height", value=20, min=5, max=50, step=1),
    "threshold": Slider(
        "Initial standing threshold",
        value=0.75,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "update_rule": {
        "type": "Select",
        "value": "random_async",
        "values": ["synchronous", "random_async", "incentive_async"],
        "label": "Update rule",
    },
    "irreversible_diffusion": {
        "type": "Checkbox",
        "value": False,
        "label": "Irreversible diffusion: once standing, cannot sit",
    },
    "seed": {
        "type": "InputText",
        "value": "42",
        "label": "Random seed",
    },
}


def standing_post_process(ax):
    ax.set_title("Number of Standing Agents")
    ax.set_xlabel("Step")
    ax.set_ylabel("Standing count")


def percent_post_process(ax):
    ax.set_title("Percent Standing")
    ax.set_xlabel("Step")
    ax.set_ylabel("Percent")


def sm_post_process(ax):
    ax.set_title("Stick-in-the-Muds")
    ax.set_xlabel("Step")
    ax.set_ylabel("Share locally opposed")


def ie_post_process(ax):
    ax.set_title("Informational Efficiency")
    ax.set_xlabel("Step")
    ax.set_ylabel("0/1")


space_component = make_space_component(
    agent_portrayal=agent_portrayal,
    backend="matplotlib",
)

standing_plot = make_plot_component(
    "Standing",
    post_process=standing_post_process,
    backend="matplotlib",
)

percent_plot = make_plot_component(
    "PercentStanding",
    post_process=percent_post_process,
    backend="matplotlib",
)

sm_plot = make_plot_component(
    "StickInMuds",
    post_process=sm_post_process,
    backend="matplotlib",
)

ie_plot = make_plot_component(
    "InformationalEfficiency",
    post_process=ie_post_process,
    backend="matplotlib",
)


model = StandingOvationModel(
    width=20,
    height=20,
    threshold=0.75,
    update_rule="random_async",
    irreversible_diffusion=False,
    seed=42,
)


page = SolaraViz(
    model,
    components=[
        space_component,
        standing_plot,
        percent_plot,
        sm_plot,
        ie_plot,
    ],
    model_params=model_params,
    name="Standing Ovation Model - SY",
)

page
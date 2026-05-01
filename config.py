from dataclasses import dataclass

@dataclass
class Config:
    class PlannerConfig:
        class astar_planner:
            max_nodes: int = 3000
            goal_threshold: float = 0.15
            # action here is 2 d:
            # dim1: velocity of ee
            # dim2: gripper open/close (1 or -1)
            action_space_A_star_2d: list = [
                (-2.0, 0),
                (-2.0, 1),
                (-1.0, 0),
                (-1.0, 1),
                (-0.5, 0),
                (-0.5, 1),
                (0.5, 0),
                (0.5, 1),
                (1.0, 0),
                (1.0, 1),
                (2.0, 0),
                (2.0, 1)
            ]
            action_space_A_star_1d: list = [(-2.0,), (-1.0,), (-0.5,), (0.5,), (1.0,), (2.0,)]

    class ModelConfig:
        latent_dim: int = 2
        action_dim: int = 1
        hidden_dim: int = 64
        learning_rate: float = 1e-3
        epochs: int = 1000
        batch_size: int = 256
        encoder_type: str = "simple"  # "simple" or "Dreamerv3" or "identity"
        n_samples: int = 3000
        data_type: str = "visual"
             # "visual" or "non_visual"

    class VisualizerConfig:
            action_space_visualize_2d: list = [
                (-2.0, 0),
                (-2.0, 1),
                (-1.0, 0),
                (-1.0, 1),
                (1.0, 0),
                (1.0, 1),
                (2.0, 0),
                (2.0, 1)
            ]
            action_space_Visualizer_1d: list = [(-2.0,),(-0.5,),(0.0,),(0.5,),(2.0,)]
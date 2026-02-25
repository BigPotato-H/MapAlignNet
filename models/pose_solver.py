import theseus as th
import torch

class TheseusRefiner:
    def __init__(self, initial_pose):
        self.optim_layer = th.LayeredOptimizer(
            objective=th.Objective(),
            optimizer_cls=th.GaussNewton,
            max_iterations=5
        )
        self.pose_variable = th.SE3(name='pose', dtype=torch.float32)
        self.initial_pose = initial_pose

    def refine(self, pose_estimation, uncertainty):
        # Define the weighted cost function using uncertainties
        weights = 1.0 / (uncertainty + 1e-8)

        # Add cost terms to the objective in Theseus
        for i in range(6):  # 6 pose components
            cost_weight = weights[:, i].mean()  # Mean weight for each batch
            residual = pose_estimation[:, i] - self.initial_pose[:, i]
            cost = th.CostFunction(residual, weight=cost_weight, dtype=torch.float32)
            self.optim_layer.objective.add(cost)

        # Run optimization to refine the initial pose
        refined_pose = self.optim_layer.optimize({'pose': self.initial_pose})
        return refined_pose

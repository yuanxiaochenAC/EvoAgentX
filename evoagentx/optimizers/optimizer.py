from ..core.module import BaseModule
from ..workflow.workflow import WorkFlow


class Optimizer(BaseModule):

    workflow: WorkFlow
    optimization_log: list = []

    def step(self):

        """
        Perform a single optimization step. 

        The optimization of a workflow may include:
        - Analyze task execution trajectory to identify failed or inefficient tasks. 
        - Optimize these tasks either by evolving the workflow or agents. 
        - Adjust workflow graph (add, remove, or update tasks).
        - Log the optimization process.
        """
        raise NotImplementedError(f"``step`` function for {type(self).__name__} is not implemented!")
    

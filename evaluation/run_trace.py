from langsmith import evaluate
from runners import agent_planning
from evaluators import steps_count, planning_judge

class Tracer():
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.output_path = "evaluation/results"

    def run_agent_trace(self, experiment_name: str, k: int, save_local: bool = False) -> None:
        """
        Run an evaluation of the agent using LangSmith for tracing.
        """

        print(f"Run the experiment: {experiment_name}")

        results = evaluate(
            agent_planning,
            data=self.dataset_name,
            experiment_prefix=experiment_name,
            evaluators=[planning_judge, steps_count],
            num_repetitions=k
        )

        if save_local:
            self._save_local(results, experiment_name)

    def _save_local(self, results, name) -> None:
        """
        Save the evaluations results to the local file
        """

        df = results.to_pandas()
        path = f"{self.output_dir}/{name}.json"
        df.to_json(path, orient="records", force_ascii=False, indent=4)

        print(f"Results saved to {path}")


if __name__ == "__main__":
    tracer = Tracer("agent_bench")
    tracer.run_agent_trace("agent_planning", 3)






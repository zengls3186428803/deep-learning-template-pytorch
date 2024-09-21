from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)


class ShowInfoCallback(TrainerCallback):
    def __init__(self):
        pass

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_end(args, state, control, **kwargs)
        if (
            state.is_local_process_zero
            and state.global_step % state.logging_steps == 0
            and state.log_history
            and "loss" in state.log_history[-1]
            and "eval_loss" not in state.log_history[-1]
        ):
            print(f"LatestLogInfo: {state.log_history[-1]}")

from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)


class SaveModelByEvalAccCallback(TrainerCallback):
    def __init__(self, base_acc=0.0, eval_acc_epsilon=0.05):
        self.base_acc = base_acc
        self.eval_acc_epsilon = eval_acc_epsilon
        self.origin_output_dir = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        result = super().on_step_end(args, state, control, **kwargs)
        if (
            state.is_local_process_zero
            and state.global_step % state.logging_steps == 0
            and state.log_history
            and "loss" in state.log_history[-1]
            and "eval_loss" not in state.log_history[-1]
        ):
            # print(f"LatestLogInfo: {state.log_history[-1]}")
            pass
        return result

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.origin_output_dir is not None:
            args.output_dir = self.origin_output_dir
        return super().on_save(args, state, control, **kwargs)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        metrics = kwargs.get("metrics", None)
        if metrics:
            eval_accuracy = metrics.get("eval_accuracy", None)
            if eval_accuracy:
                if eval_accuracy >= self.base_acc + self.eval_acc_epsilon:
                    print(
                        f"<<<on_evaluate>>> is called, cur_acc:{eval_accuracy}, base_acc:{self.base_acc}"
                    )
                    control.should_save = True
                    if self.origin_output_dir is None:
                        self.origin_output_dir = args.output_dir
                    args.output_dir = f"{self.origin_output_dir}/checkpoint-{str(state.global_step)}-acc{eval_accuracy:.3f}"
                    self.base_acc = eval_accuracy
        return super().on_evaluate(args, state, control, **kwargs)

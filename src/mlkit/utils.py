def checkpoint_model_and_save_current_best(step: int, checkpoint_every: int):
    if step % checkpoint_every == 0:
        print(f"Checkpoint model at step {step}")
    save_model_if_is_current_best()


def save_model_if_is_current_best():
    print("Checking if model is current best")

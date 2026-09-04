"""
Trains the model to imitate your recorded human gameplay (behavioral cloning),
producing a new named checkpoint under TrainedWeights/<name>/.

This is supervised learning: for every recorded (state, action) pair, it
nudges the origin/shell/card actor networks to make the action you actually
took more likely, given that state. It's a much faster and denser learning
signal than pure PPO self-play, which only gets a handful of reward signals
per multi-minute match.

Usage:
    python tools/recorder.py            # record a few matches first
    python tools/behavioral_clone.py --name my_deck_v1
    python tools/behavioral_clone.py --name my_deck_v2 --init_from my_deck_v1 --epochs 10

Then play/keep training with:
    python run.py --model my_deck_v1
"""
import argparse
import glob
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF's C++ level logs
import warnings
warnings.filterwarnings("ignore")

import pickle

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

from CRAgent import Agent

DATA_DIR_DEFAULT = "Resources/Data/HumanGames"


def load_samples(data_dir):
    samples = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        with open(f, "rb") as fh:
            samples.extend(pickle.load(fh))
    return samples


def bc_step(actor, encoded_state, target_idx):
    with tf.GradientTape() as tape:
        probs = actor(encoded_state)
        target_prob = probs[0][target_idx]
        loss = -tf.math.log(target_prob + 1e-8)
    grads = tape.gradient(loss, actor.trainable_variables)
    actor.optimizer.apply_gradients(zip(grads, actor.trainable_variables))
    return float(loss)


def ensure_checkpoint_dirs(path):
    for sub in ("StateWeights", "OriginWeights", "ShellWeights", "CardWeights"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)


def checkpoint_has_policy_weights(path):
    required = (
        "OriginWeights/origin_actor.index",
        "ShellWeights/shell_actor.index",
        "CardWeights/card_actor.index",
    )
    return all(os.path.exists(os.path.join(path, file_name)) for file_name in required)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="new checkpoint folder name under TrainedWeights/")
    parser.add_argument("--init_from", default=None,
                         help="existing TrainedWeights subfolder to start from (omit to train from the repo's shipped base weights)")
    parser.add_argument("--data_dir", default=DATA_DIR_DEFAULT)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    samples = load_samples(args.data_dir)
    print(f"Loaded {len(samples)} recorded actions from {args.data_dir}")
    if not samples:
        print("ERROR: no recorded data found. Run tools/recorder.py while playing a few matches first.")
        return

    try:
        agent = Agent(load=False)
        if args.init_from:
            init_path = f"TrainedWeights/{args.init_from}/"
            print(f"Initializing from {init_path}")
            if not checkpoint_has_policy_weights(init_path):
                raise FileNotFoundError(
                    f"Checkpoint '{init_path}' is missing policy weights. "
                    "Use a valid checkpoint name or omit --init_from to train from scratch."
                )
            agent.load(path=init_path)
        else:
            base_path = "TrainedWeights/"
            if checkpoint_has_policy_weights(base_path):
                print("Initializing from the repo's shipped base weights (TrainedWeights/)")
                agent.load(path=base_path)
            else:
                print("No base policy weights found; initializing fresh models from scratch.")

        # Rebuild fresh optimizers after loading. load_weights() can drag in stale/
        # mismatched optimizer-slot variables from the checkpoint (harmless model
        # weights restore fine, but the optimizer state does not line up with a
        # freshly-constructed Agent) which then crashes the first apply_gradients
        # call. Discarding and recompiling avoids that entirely - we don't need
        # the old optimizer momentum for a fresh behavioral-cloning run anyway.
        agent.compile(origin_lr=1e-3, shell_lr=1e-3, card_lr=1e-3)

        # Warm up the state autoencoder's reconstruction on your recorded states too,
        # so it represents your deck's on-screen cards/field well before the
        # actor heads try to learn from its encoding.
        print("Fitting state autoencoder on recorded states...")
        for s in samples:
            try:
                agent.state_autoencoder.fit(s["state"], verbose=0)
            except Exception as e:
                print("  (autoencoder fit step skipped due to:", e, ")")

        print(f"\nTraining on {len(samples)} samples for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            total_loss, n = 0.0, 0
            for s in samples:
                enc = agent.state_autoencoder.encode(s["state"])
                total_loss += bc_step(agent.origin_actor, enc, s["origin_idx"])
                total_loss += bc_step(agent.shell_actor, enc, s["shell_idx"])
                total_loss += bc_step(agent.card_actor, enc, s["card_idx"])
                n += 3
            print(f"  epoch {epoch + 1}/{args.epochs}  avg loss {total_loss / n:.4f}")

        out_path = f"TrainedWeights/{args.name}/"
        ensure_checkpoint_dirs(out_path)
        agent.save(path=out_path)
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Saved fine-tuned model to {out_path}")
        print(f"Run it with: python run.py --model {args.name}")
        print(f"=========================")
    except Exception as e:
        print(f"\n=== TRAINING FAILED ===")
        print(f"Error: {e}")
        print(f"========================")
        os._exit(1)

    # Exit immediately rather than letting the interpreter's normal shutdown/
    # garbage-collection run - that's what prints TF's harmless but noisy
    # "unrestored checkpoint values" warnings on the way out.
    os._exit(0)


if __name__ == "__main__":
    main()

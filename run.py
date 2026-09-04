"""
Unified entry point: pick which trained model and which deck to play with
from the command line, instead of MainLoop.py's hardcoded single checkpoint.

Examples:
    python run.py                                   # base model, whatever deck is currently active
    python run.py --model my_deck_v1                 # play/keep-training a named checkpoint
    python run.py --model my_deck_v1 --deck my_deck   # also switch active deck first
    python run.py --episodes 5

Workflow for a new deck end-to-end:
    python tools/deck_manager.py capture --name my_deck
    python tools/deck_manager.py activate --name my_deck
    python tools/recorder.py                # play a few matches manually
    python tools/behavioral_clone.py --name my_deck_v1
    python run.py --model my_deck_v1 --deck my_deck
"""
import argparse

from CRBot import CRBot
from CRAgent import Agent
from CRHandler import Handler
from tools.deck_manager import activate_deck


def model_path(name):
    if not name or name == "base":
        return "TrainedWeights/"  # the repo's original, un-suffixed checkpoint
    return f"TrainedWeights/{name}/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base",
                         help="checkpoint folder name under TrainedWeights/ (default: base = original weights)")
    parser.add_argument("--deck", default=None,
                         help="deck name under Resources/Decks/ to activate before running")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--spells", action="store_true", help="pass if your deck has spell cards")
    args = parser.parse_args()

    if args.deck:
        if not activate_deck(args.deck):
            return

    path = model_path(args.model)
    print(f"Using model checkpoint: {path}")

    bot = CRBot()
    agent = Agent(load=False)
    agent.load(path=path)
    env = Handler(args.spells)

    for ep in range(1, args.episodes + 1):
        duration, total_reward, pc, ec = bot.run_episode(agent, env)
        bot.print_episode_stats(ep, duration, total_reward, pc, ec)
        agent.train()
        agent.save(path=path)


if __name__ == "__main__":
    main()

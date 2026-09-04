"""
Deck management for the Clash Royale RL bot.

Card recognition works by template-matching the 4 cards in your hand against
whatever .png files are sitting in Resources/Cards/. This tool lets you:

  1) capture a new deck's 8 card images directly from a live game window
     (this is far more reliable than grabbing card art off the internet,
     since it matches the exact crop/scale/compression the bot expects)
  2) switch which deck is "active" (i.e. copied into Resources/Cards/)

Usage (run from the repo root, with your venv active):

    python tools/deck_manager.py capture --name my_new_deck
    python tools/deck_manager.py activate --name my_new_deck
    python tools/deck_manager.py list
"""
import argparse
import os
import shutil

from CRHandler import Handler
from PIL import Image

DECKS_DIR = "Resources/Decks"
CARDS_DIR = "Resources/Cards"

# Same crop box CRHandler.get_cards() uses for hand slot 1 (35x43 px).
HAND_SLOT_1_BOX = (58, 350, 93, 393)


def capture_deck(name):
    os.makedirs(os.path.join(DECKS_DIR, name), exist_ok=True)
    handler = Handler(load_elixir_model=False)

    print("\nMake sure BlueStacks + Clash Royale is visible on screen.")
    print("For each card: put it in HAND SLOT 1 (leftmost slot), then type its name here and press Enter.")
    print("Type 'done' when you've captured all 8 cards.\n")

    captured = []
    while True:
        card_name = input("Card currently in slot 1 (or 'done'): ").strip()
        if card_name.lower() == "done":
            break
        if not card_name:
            continue
        frame = handler.get_frame()
        img = Image.fromarray(frame).crop(HAND_SLOT_1_BOX)
        out_path = os.path.join(DECKS_DIR, name, f"{card_name}.png")
        img.save(out_path)
        captured.append(card_name)
        print(f"  saved -> {out_path}")

    print(f"\nCaptured {len(captured)} cards for deck '{name}': {captured}")
    if len(captured) != 8:
        print(f"WARNING: a Clash Royale deck has 8 cards, you captured {len(captured)}. "
              f"Card recognition will only work for cards you captured.")


def activate_deck(name):
    src = os.path.join(DECKS_DIR, name)
    if not os.path.isdir(src):
        print(f"No deck named '{name}' found in {DECKS_DIR}/. Run 'capture --name {name}' first.")
        return False

    os.makedirs(CARDS_DIR, exist_ok=True)
    for f in os.listdir(CARDS_DIR):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(CARDS_DIR, f))

    count = 0
    for f in os.listdir(src):
        if f.lower().endswith(".png"):
            shutil.copy(os.path.join(src, f), os.path.join(CARDS_DIR, f))
            count += 1

    with open(os.path.join(DECKS_DIR, ".active"), "w") as fh:
        fh.write(name)

    print(f"Activated deck '{name}': {count} card templates now in {CARDS_DIR}/")
    return True


def list_decks():
    if not os.path.isdir(DECKS_DIR):
        print("No decks saved yet. Use 'capture --name <deckname>' first.")
        return

    active = None
    active_path = os.path.join(DECKS_DIR, ".active")
    if os.path.exists(active_path):
        active = open(active_path).read().strip()

    found = False
    for d in sorted(os.listdir(DECKS_DIR)):
        if d.startswith("."):
            continue
        found = True
        cards = [f for f in os.listdir(os.path.join(DECKS_DIR, d)) if f.lower().endswith(".png")]
        marker = "  <-- active" if d == active else ""
        print(f"- {d}: {len(cards)} cards{marker}")
    if not found:
        print("No decks saved yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="capture a new deck's 8 cards from the live game")
    cap.add_argument("--name", required=True)

    act = sub.add_parser("activate", help="make a saved deck the active one in Resources/Cards")
    act.add_argument("--name", required=True)

    sub.add_parser("list", help="list saved decks")

    args = parser.parse_args()
    os.makedirs(DECKS_DIR, exist_ok=True)

    if args.cmd == "capture":
        capture_deck(args.name)
    elif args.cmd == "activate":
        activate_deck(args.name)
    elif args.cmd == "list":
        list_decks()

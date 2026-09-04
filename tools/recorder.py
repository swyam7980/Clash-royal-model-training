"""
Records YOU playing Clash Royale manually, and turns it into labeled training
data (state -> action) for behavioral_clone.py.

How it works:
- A background thread keeps grabbing the current game state every ~0.5s using
  the same Handler.get_state() the bot itself uses, so training data matches
  the bot's real input format exactly.
- A global mouse listener watches for the two-click sequence you actually use
  to play a card: click on one of the 4 hand-card slots, then click on the
  battlefield to place it. When it sees that pair, it converts the placement
  click's screen position back into the bot's internal
  (origin_tile, shell_tile) action representation, and saves
  (state, origin_idx, shell_idx, card_idx) as one training sample.

Usage:
    pip install pynput
    python -m tools.recorder

Then just play Clash Royale normally in BlueStacks. Press Ctrl+C in this
terminal when you're done (e.g. after a match or a few matches) to save.
Run it again for more sessions - behavioral_clone.py will use all saved
sessions together.

NOTE: Handler.get_window_dimensions() force-focuses the BlueStacks window
every time it's called (that's existing repo behavior, not something this
script adds). Since it's polled every ~0.5s here, don't be alarmed if the
window keeps stealing focus/foreground - that's expected and needed for the
screenshots to work, just don't let it interrupt your clicking.
"""
import argparse
import os
import pickle
import threading
import time
from datetime import datetime

import pythoncom
from pynput import mouse

from CRHandler import Handler
from ActionMapper import ActionMapper

OUT_DIR = "Resources/Data/HumanGames"
ACTION_WINDOW = 3.0   # max seconds between card-slot click and placement click
POLL_INTERVAL = 0.5   # how often to snapshot game state in the background

# Hand-slot bounding boxes in the rescaled 244x419 "frame" space, taken
# directly from CRHandler.get_cards()'s own crop boxes.
CARD_SLOT_BOXES_FRAME = [
    (58, 350, 93, 393),
    (105, 350, 140, 393),
    (150, 350, 185, 393),
    (195, 350, 230, 393),
]


def build_tile_lookup():
    """Inverts ActionMapper's forward (origin, shell) -> tile logic so we can
    go the other way: tile the human clicked -> (origin_idx, shell_idx)."""
    am = ActionMapper()
    origins = am.get_origin_square_locations()
    lookup = {}
    for origin_idx, (ox, oy) in enumerate(origins):
        for shell_idx in range(9):
            dx, dy = am.get_tile(shell_idx)
            tile = (ox + dx, oy + dy)
            lookup.setdefault(tile, (origin_idx, shell_idx))
    return lookup


def screen_to_frame(handler, screen_x, screen_y):
    """Converts a raw screen click into the rescaled 244x419 frame space that
    get_frame()/get_cards() operate in."""
    window_dimensions = handler.get_window_dimensions()
    scalars = handler.get_window_scalars()
    frame_x = (screen_x - window_dimensions[0]) / scalars[0]
    frame_y = (screen_y - window_dimensions[1]) / scalars[1]
    return frame_x, frame_y


def card_slot_at(frame_x, frame_y):
    for idx, (l, t, r, b) in enumerate(CARD_SLOT_BOXES_FRAME):
        if l <= frame_x <= r and t <= frame_y <= b:
            return idx
    return None


def screen_to_tile(handler, screen_x, screen_y):
    """Inverts CRHandler.gen_choice_data's tile->screen formula."""
    window_dimensions = handler.get_window_dimensions()
    scalars = handler.get_window_scalars()
    window_bottom_left = (window_dimensions[0], window_dimensions[3])
    bottom_left_tile = (
        window_bottom_left[0] + 24 * scalars[0],
        window_bottom_left[1] - 111 * scalars[1],
    )
    tile_size = (11 * scalars[0], 8 * scalars[1])
    x = round((screen_x - bottom_left_tile[0]) / tile_size[0])
    y = round((bottom_left_tile[1] - screen_y) / tile_size[1])
    x = max(0, min(17, x))
    y = max(0, min(13, y))
    return x, y


class Recorder:
    def __init__(self, spells=False):
        self.handler = Handler(spells=spells)
        self.lookup = build_tile_lookup()
        self.latest_state = None
        self.state_lock = threading.Lock()
        self.samples = []
        self.pending_card = None
        self.pending_card_time = 0
        self.running = True

    def poll_state(self):
        pythoncom.CoInitialize()  # this thread needs its own COM apartment for win32com calls
        try:
            while self.running:
                try:
                    state = self.handler.get_state()
                    with self.state_lock:
                        self.latest_state = state
                except Exception as e:
                    print("state capture failed (will keep retrying):", e)
                time.sleep(POLL_INTERVAL)
        finally:
            pythoncom.CoUninitialize()

    def on_click(self, x, y, button, pressed):
        if not pressed or button != mouse.Button.left:
            return

        frame_x, frame_y = screen_to_frame(self.handler, x, y)
        slot = card_slot_at(frame_x, frame_y)
        if slot is not None:
            self.pending_card = slot
            self.pending_card_time = time.time()
            print(f"[card slot {slot} selected] waiting for placement click...")
            return

        # Not a card-slot click. Only treat it as a placement if a card was
        # just selected recently - otherwise it's an unrelated click (menus,
        # emotes, etc.) and should be ignored.
        if self.pending_card is None or (time.time() - self.pending_card_time) > ACTION_WINDOW:
            return

        raw_tile = screen_to_tile(self.handler, x, y)
        # The action space's origin grid tops out at x=15, y=11 with a +-1
        # shell offset, so the model can never reach x=17 or y=13 no matter
        # what. Clamp those edge clicks to the nearest reachable tile instead
        # of throwing the sample away, so it still learns "play near here".
        tile = (min(raw_tile[0], 16), min(raw_tile[1], 12))
        if tile != raw_tile:
            print(f"  click at {raw_tile} is outside the model's reachable tiles, clamped to {tile}")

        if tile not in self.lookup:
            print(f"  click at tile {tile} isn't a valid placement tile, skipping")
            self.pending_card = None
            return

        origin_idx, shell_idx = self.lookup[tile]
        with self.state_lock:
            state = self.latest_state

        if state is None:
            print("  no game state captured yet, skipping this action")
            self.pending_card = None
            return

        self.samples.append({
            "state": state,
            "origin_idx": origin_idx,
            "shell_idx": shell_idx,
            "card_idx": self.pending_card,
            "t": time.time(),
        })
        print(f"  recorded action #{len(self.samples)}  card={self.pending_card}  tile={tile}")
        self.pending_card = None

    def run(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        threading.Thread(target=self.poll_state, daemon=True).start()

        ms_listener = mouse.Listener(on_click=self.on_click)
        ms_listener.start()

        print("Recording started. Play Clash Royale normally in BlueStacks:")
        print("  1) click the card in your hand, 2) click where to place it.")
        print("Press Ctrl+C here when done to save this session.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            ms_listener.stop()
            self.save()

    def save(self):
        if not self.samples:
            print("No actions recorded, nothing saved.")
            return
        fname = os.path.join(OUT_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(fname, "wb") as fh:
            pickle.dump(self.samples, fh)
        print(f"\nSaved {len(self.samples)} actions to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spells", action="store_true", help="pass if your deck has spell cards")
    args = parser.parse_args()
    Recorder(spells=args.spells).run()

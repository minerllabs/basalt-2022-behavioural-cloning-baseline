# Code for loading OpenAI MineRL VPT datasets
# (NOTE: Not the original code!)
import json
import glob
import os
import random
from multiprocessing import Process, Queue, Event

import numpy as np
import cv2

from openai_vpt.agent import ACTION_TRANSFORMER_KWARGS, resize_image, AGENT_RESOLUTION
from openai_vpt.lib.actions import ActionTransformer

QUEUE_TIMEOUT = 10

CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

# Mapping from JSON keyboard buttons to MineRL actions
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MINEREC_ORIGINAL_HEIGHT_PX = 720
# Matches a number in the MineRL Java code
# search the code Java code for "constructMouseState"
# to find explanations
CAMERA_SCALER = 360.0 / 2400.0

# If GUI is open, mouse dx/dy need also be adjusted with these scalers.
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


def data_loader_worker(tasks_queue, output_queue, quit_workers_event):
    """
    Worker for the data loader.
    """
    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    # Assume 16x16
    cursor_image = cursor_image[:16, :16, :]
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_image = cursor_image[:, :, :3]

    while True:
        task = tasks_queue.get()
        if task is None:
            break
        trajectory_id, video_path, json_path = task
        video = cv2.VideoCapture(video_path)
        # Note: In some recordings, the game seems to start
        #       with attack always down from the beginning, which
        #       is stuck down until player actually presses attack
        attack_is_stuck = False
        # Scrollwheel is allowed way to change items, but this is
        # not captured by the recorder.
        # Work around this by keeping track of selected hotbar item
        # and updating "hotbar.#" actions when hotbar selection changes.
        last_hotbar = 0

        with open(json_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)

        for i in range(len(json_data)):
            if quit_workers_event.is_set():
                break
            step_data = json_data[i]

            if i == 0:
                # Check if attack will be stuck down
                if step_data["mouse"]["newButtons"] == [0]:
                    attack_is_stuck = True
            elif attack_is_stuck:
                # Check if we press attack down, then it might not be stuck
                if 0 in step_data["mouse"]["newButtons"]:
                    attack_is_stuck = False
            # If still stuck, remove the action
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

            action, is_null_action = json_action_to_env_action(step_data)

            # Update hotbar selection
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            # Read frame even if this is null so we progress forward
            ret, frame = video.read()
            if ret:
                # Skip null actions as done in the VPT paper
                # NOTE: in VPT paper, this was checked _after_ transforming into agent's action-space.
                #       We do this here as well to reduce amount of data sent over.
                if is_null_action:
                    continue
                if step_data["isGuiOpen"]:
                    camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                    cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                    cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                    composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)
                output_queue.put((trajectory_id, frame, action), timeout=QUEUE_TIMEOUT)
            else:
                print(f"Could not read frame from video {video_path}")
        video.release()
        # Signal that this task is done
        # Yes we are using "None"s to tell when worker is done
        # and when individual work-items are done...
        output_queue.put((trajectory_id, None, None), timeout=QUEUE_TIMEOUT)
        if quit_workers_event.is_set():
            break
    # Tell that we ended
    output_queue.put(None)

class DataLoader:
    """
    Generator class for loading batches from a dataset

    This only returns a single step at a time per worker; no sub-sequences.
    Idea is that you keep track of the model's hidden state and feed that in,
    along with one sample at a time.

    + Simpler loader code
    + Supports lower end hardware
    - Not very efficient (could be faster)
    - No support for sub-sequences
    - Loads up individual files as trajectory files (i.e. if a trajectory is split into multiple files,
      this code will load it up as a separate item).
    """
    def __init__(self, dataset_dir, n_workers=8, batch_size=8, n_epochs=1, max_queue_size=8):
        assert n_workers >= batch_size, "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids
        # Create tuples of (video_path, json_path) for each unique_id
        demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))

        assert n_workers <= len(demonstration_tuples), f"n_workers should be lower or equal than number of demonstrations {len(demonstration_tuples)}"

        # Repeat dataset for n_epochs times, shuffling the order for
        # each epoch
        self.demonstration_tuples = []
        for i in range(n_epochs):
            random.shuffle(demonstration_tuples)
            self.demonstration_tuples += demonstration_tuples

        self.task_queue = Queue()
        self.n_steps_processed = 0
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put((trajectory_id, *task))
        for _ in range(n_workers):
            self.task_queue.put(None)

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                ),
                daemon=True
            )
            for output_queue in self.output_queues
        ]
        for process in self.processes:
            process.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_actions = []
        batch_episode_id = []

        for i in range(self.batch_size):
            workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(timeout=QUEUE_TIMEOUT)
            if workitem is None:
                # Stop iteration when first worker runs out of work to do.
                # Yes, this has a chance of cutting out a lot of the work,
                # but this ensures batches will remain diverse, instead
                # of having bad ones in the end where potentially
                # one worker outputs all samples to the same batch.
                raise StopIteration()
            trajectory_id, frame, action = workitem
            batch_frames.append(frame)
            batch_actions.append(action)
            batch_episode_id.append(trajectory_id)
            self.n_steps_processed += 1
        return batch_frames, batch_actions, batch_episode_id

    def __del__(self):
        self.quit_workers_event.set()
        for process in self.processes:
            process.terminate()
            process.join()

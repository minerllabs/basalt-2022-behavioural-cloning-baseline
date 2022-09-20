import json
import os

# DO NOT CHANGE THESE VALUES
# These will be enforced in the evaluation server:
# if you tinker with them, your submissions will almost certainly fail!

# 10 for evaluation + 1 leaderboard video
EVAL_EPISODES = int(os.getenv("AICROWD_NUM_EVAL_EPISODES", 11))
# This is only used to limit steps when debugging is on.
# Environments will automatically return done=True once
# the environment-specific timeout is reached
EVAL_MAX_STEPS = int(os.getenv("AICROWD_NUM_EVAL_MAX_STEPS", 1e9))

settings = json.load(open('aicrowd.json'))
if settings.get('debug', False):
    # if debug flag is set to true, evaluation will only run a single episode for 100 steps.
    # Again, this will be enforced by the evaluation server, and do not do more episodes/steps
    # than this
    EVAL_EPISODES = int(os.getenv("AICROWD_NUM_EVAL_EPISODES", 2))
    EVAL_MAX_STEPS = int(os.getenv("AICROWD_NUM_EVAL_MAX_STEPS", 100))

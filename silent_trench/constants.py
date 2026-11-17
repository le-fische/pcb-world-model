"""Constants and enumerations for Silent Trench."""
from enum import Enum, auto


class ActionType(str, Enum):
    # Free actions (0 Power)
    MOVE = "MOVE"
    SECURE = "SECURE"
    WAIT = "WAIT"
    TORPEDO = "TORPEDO"

    # Tactical actions (Requires turn + Power)
    DEEP_COVER = "DEEP_COVER"
    ACTIVE_SONAR = "ACTIVE_SONAR"
    OVERCHARGE = "OVERCHARGE"

    # Bonus / Defensive upgrades (Requires Power)
    STRIKE_REPORTS = "STRIKE_REPORTS"
    ACOUSTIC_MASKING = "ACOUSTIC_MASKING"
    DEPLOY_SENSOR = "DEPLOY_SENSOR"


class NodeType(str, Enum):
    STANDARD = "STANDARD"
    LISTENING_POST = "LISTENING_POST"    # Yields +1 Power per turn when controlled
    THERMAL_LAYER = "THERMAL_LAYER"      # Acoustic damping; grants free Deep Cover on entry
    CHOKEPOINT = "CHOKEPOINT"            # Strategic ridge or trench wall bottleneck
    ABYSSAL_FLOOR = "ABYSSAL_FLOOR"      # Ultra-deep sector; high strategic control value


class GameStatus(int, Enum):
    ONGOING = 0
    P0_WIN = 1
    P1_WIN = 2
    DRAW = 3


class OceanRegion(str, Enum):
    ATLANTIC = "ATLANTIC"
    PACIFIC = "PACIFIC"


# Economy and balance defaults
ACTIONS_PER_TURN = 2
MAX_POWER = 6
STARTING_POWER = 2
MAX_TURNS = 60
WIN_CONTROL_NODES = 3  # Controlling 3 Listening Posts wins via naval domination

# Power Costs
COST_DEEP_COVER = 1
COST_ACTIVE_SONAR = 2  # LOCATE
COST_OVERCHARGE = 1
COST_STRIKE_REPORTS = 1
COST_ACOUSTIC_MASKING = 1
COST_DEPLOY_SENSOR = 1

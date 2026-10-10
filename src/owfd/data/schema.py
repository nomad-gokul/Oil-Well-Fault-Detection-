"""Static facts about the Petrobras 3W dataset (v2.0.0), taken from its ``dataset.ini``."""

from __future__ import annotations

from enum import StrEnum

DATASET_VERSION = "2.0.0"

#: Offset between an event's label and its transient label (e.g. 101 = transient of 1).
TRANSIENT_OFFSET = 100

#: Fault class label -> human-readable name.
CLASS_NAMES: dict[int, str] = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line",
}

#: Event types whose instances contain a labelled transient period (1xx labels).
CLASSES_WITH_TRANSIENT: frozenset[int] = frozenset({1, 2, 5, 6, 7, 8, 9})

#: Continuous sensor variables (pressures in Pa, temperatures in degC, flows in m3/s,
#: choke openings in %).
SENSOR_VARIABLES: tuple[str, ...] = (
    "ABER-CKGL",
    "ABER-CKP",
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "PT-P",
    "P-TPT",
    "QBS",
    "QGL",
    "T-JUS-CKP",
    "T-MON-CKP",
    "T-PDG",
    "T-TPT",
)

#: Discrete valve states (0 = closed, 0.5 = partially open/transition, 1 = open).
VALVE_STATE_VARIABLES: tuple[str, ...] = (
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
)

LABEL_COLUMN = "class"
STATE_COLUMN = "state"


class Source(StrEnum):
    """Origin of an instance."""

    REAL = "real"
    SIMULATED = "simulated"
    DRAWN = "drawn"


def base_class(label: int) -> int:
    """Map a raw 3W label (0..9 or 101..109) to its event class 0..9."""
    return label - TRANSIENT_OFFSET if label >= TRANSIENT_OFFSET else label


def is_transient(label: int) -> bool:
    """Return True for transient labels (101..109)."""
    return label >= TRANSIENT_OFFSET

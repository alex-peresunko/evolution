"""Creature state as a struct of arrays.

Every creature's per-tick state lives in one row of a CreatureTable: floats in `fl`, ints (bools
as 0/1, None ticks as NONE) in `it`, and its gene values in `genes`. Creature reads and writes
its fields through the descriptors below, so Python code keeps using `c.health`, `c.pos.x` and
so on, while the compiled code (perception, sim.act_kernel, collision) works on the arrays in
place, with no gathering and writing back.

A creature is in exactly one table at a time (`c._t`, row `c._r`). A new creature gets a table
of its own; World.perceive moves the living ones into the world's table, in world.creatures()
order (`arrange`), and a creature that leaves it gets a table of its own again. A table's
`members` can go stale (a member arranged into another table); `holds` and `arrange` notice.

Types are part of the behaviour (the golden trace hashes reprs): float fields that Python can
leave holding an int (health, fat, stamina, score) have an is-int flag; fields set only from
the genes keep their Python value and mirror a float copy into the table (Mirror).
"""
import math
import operator

import numpy as np

from ..config import NUM_MEMORY, NUM_SIGNALS, NUM_WHISKERS
from ..species import ALL_SPECIES

NONE = -(2 ** 62)          # an int field that is None

# Float columns. The first 23 are sim.act_kernel's and must stay in this order (it asserts so).
(X, Y, ANGLE, TURN, HEALTH, MAX_HEALTH, FAT, STAMINA, MAX_STAMINA, SIZE, MAX_SPEED, GEN_SIZE,
 EFF_MAX_SPEED, SPEED, HURT, MEAL_HEALTH, SCORE, TRAIT_UPKEEP, LIFESPAN, MATE_INSTINCT,
 TARGET_DIST, SMELL, SIGNAL, SIGHT, SIGHT_ANGLE, GEN_MAX_SPEED, ATTRACTIVENESS, EFFECTIVE_SIGHT,
 KNOCK_X, KNOCK_Y) = range(30)
MEMORY = slice(30, 30 + NUM_MEMORY)
WHISKERS = slice(MEMORY.stop, MEMORY.stop + NUM_WHISKERS)
TARGET_WHISKERS = slice(WHISKERS.stop, WHISKERS.stop + NUM_WHISKERS)   # inf: nothing seen
WHISKER_ANGLES = slice(TARGET_WHISKERS.stop, TARGET_WHISKERS.stop + NUM_WHISKERS)
CALLS = slice(WHISKER_ANGLES.stop, WHISKER_ANGLES.stop + NUM_SIGNALS)   # SIGNAL is their sum
N_FL = CALLS.stop

# Int columns. The first 21 are sim.act_kernel's and must stay in this order.
(SPECIES, AGE, THIRST, HUNGER, EATING, STUNNED, IMMUNE, ADULT, WINDED, IN_WATER, WILLING,
 MATE_OVERRIDE, LAST_BITE, LAST_HIT, LAST_TRAMPLE, LAST_TRAMPLED, MEAL_CARCASS, STAMINA_INT,
 FAT_INT, SCORE_INT, ACTS, HEALTH_INT, ID, REPRODUCTIONS, LAST_REPRODUCTION,
 LAST_IMPACT) = range(26)
N_IT = 26

SPECIES_AT = {id(s): n for n, s in enumerate(ALL_SPECIES)}
GENE_KEYS = [list(s.gene_bounds) for s in ALL_SPECIES]   # gene columns of each species
N_GENES = max(len(k) for k in GENE_KEYS)

_is = operator.is_


class CreatureTable:
    def __init__(self, rows=0):
        self.members = []
        self.stale = False
        # (inputs, outputs) matrices of the last brain run, row n for members[n]; None once
        # the rows have been rearranged since.
        self.brain_io = None
        self._allocate(rows)

    def _allocate(self, rows):
        self.fl = np.zeros((rows, N_FL))
        self.it = np.zeros((rows, N_IT), dtype=np.int64)
        self.genes = np.zeros((rows, N_GENES))
        self.flv = memoryview(self.fl)
        self.itv = memoryview(self.it)

    @classmethod
    def own(cls, creature):
        """A one-row table holding only `creature` (row 0); fill the row before reading it."""
        table = cls(1)
        table.members = [creature]
        return table

    def __len__(self):
        return len(self.members)

    def holds(self, creatures):
        """Whether row n is creatures[n] for every n, and nothing else is in the table."""
        members = self.members
        return (not self.stale and len(members) == len(creatures)
                and all(map(_is, members, creatures)))

    def arrange(self, creatures):
        """Make row n hold creatures[n]: members that are not in `creatures` move to tables of
        their own, and creatures from other tables move in."""
        if self.holds(creatures):
            return
        keep = set(map(id, creatures))
        for c in self.members:
            if c._t is self and id(c) not in keep:
                c._release()
        n = len(creatures)
        rows = np.array([c._r if c._t is self else -1 for c in creatures], dtype=np.intp)
        fl = np.empty((n, N_FL))
        it = np.empty((n, N_IT), dtype=np.int64)
        genes = np.empty((n, N_GENES))
        here = rows >= 0
        fl[here] = self.fl[rows[here]]
        it[here] = self.it[rows[here]]
        genes[here] = self.genes[rows[here]]
        for j in np.flatnonzero(~here).tolist():
            c = creatures[j]
            other, r = c._t, c._r
            fl[j], it[j], genes[j] = other.fl[r], other.it[r], other.genes[r]
            other.stale = True
        self.fl, self.it, self.genes = fl, it, genes
        self.flv, self.itv = memoryview(fl), memoryview(it)
        for j, c in enumerate(creatures):
            c._t, c._r = self, j
        self.members = list(creatures)
        self.stale = False
        self.brain_io = None


def rows_in(table, creatures):
    """The rows of `creatures` in `table` as an index array, or None unless all are in it."""
    rows = []
    for c in creatures:
        if c._t is not table:
            return None
        rows.append(c._r)
    return np.array(rows, dtype=np.intp)


def fertile(fl, it, tick, max_health, threshold, max_reproductions, cooldown):
    """Creature.fertile(tick) of every row, vectorized: the same comparisons on the same
    values. `max_health` is the species' max health per SPECIES index."""
    last = it[:, LAST_REPRODUCTION]
    return ((it[:, ADULT] != 0)
            & (fl[:, HEALTH] > max_health[it[:, SPECIES]] * threshold)
            & (it[:, REPRODUCTIONS] < max_reproductions)
            & ((last == NONE) | (tick - last > cooldown)))


# --- descriptors of Creature's fields --------------------------------------------------------

class Float:
    __slots__ = ("col",)

    def __init__(self, col):
        self.col = col

    def __get__(self, c, owner=None):
        if c is None:
            return self
        return c._t.flv[c._r, self.col]

    def __set__(self, c, value):
        c._t.flv[c._r, self.col] = value


class Number:
    """A float field that keeps an int as an int (flag column `flag`)."""
    __slots__ = ("col", "flag")

    def __init__(self, col, flag):
        self.col, self.flag = col, flag

    def __get__(self, c, owner=None):
        if c is None:
            return self
        t, r = c._t, c._r
        value = t.flv[r, self.col]
        return int(value) if t.itv[r, self.flag] else value

    def __set__(self, c, value):
        t, r = c._t, c._r
        t.flv[r, self.col] = value
        t.itv[r, self.flag] = type(value) is int


class OptionalFloat:
    """A float field that may be None (stored as NaN)."""
    __slots__ = ("col",)

    def __init__(self, col):
        self.col = col

    def __get__(self, c, owner=None):
        if c is None:
            return self
        value = c._t.flv[c._r, self.col]
        return None if value != value else value

    def __set__(self, c, value):
        c._t.flv[c._r, self.col] = math.nan if value is None else value


class Int:
    __slots__ = ("col",)

    def __init__(self, col):
        self.col = col

    def __get__(self, c, owner=None):
        if c is None:
            return self
        return c._t.itv[c._r, self.col]

    def __set__(self, c, value):
        c._t.itv[c._r, self.col] = value


class Bool(Int):
    __slots__ = ()

    def __get__(self, c, owner=None):
        if c is None:
            return self
        return c._t.itv[c._r, self.col] != 0


class Tick(Int):
    """An int field that may be None (stored as NONE)."""
    __slots__ = ()

    def __get__(self, c, owner=None):
        if c is None:
            return self
        value = c._t.itv[c._r, self.col]
        return None if value == NONE else value

    def __set__(self, c, value):
        c._t.itv[c._r, self.col] = NONE if value is None else value


class Mirror:
    """A field set from the genes: the Python value as assigned, plus a float copy in the
    table for the compiled code."""
    __slots__ = ("col", "key")

    def __init__(self, col, name):
        self.col, self.key = col, "_mirror_" + name

    def __get__(self, c, owner=None):
        if c is None:
            return self
        return c.__dict__[self.key]

    def __set__(self, c, value):
        c.__dict__[self.key] = value
        c._t.flv[c._r, self.col] = value


class Floats:
    """A list of floats in the columns `cols`; `inf_is_none`: inf reads as None."""
    __slots__ = ("cols", "inf_is_none")

    def __init__(self, cols, inf_is_none=False):
        self.cols, self.inf_is_none = cols, inf_is_none

    def __get__(self, c, owner=None):
        if c is None:
            return self
        values = c._t.fl[c._r, self.cols].tolist()
        if self.inf_is_none:
            return [None if v == math.inf else v for v in values]
        return values

    def __set__(self, c, values):
        if self.inf_is_none:
            values = [math.inf if v is None else v for v in values]
        c._t.fl[c._r, self.cols] = values


class Mirrors:
    """Like Mirror, for an array of floats (whisker_angles)."""
    __slots__ = ("cols", "key")

    def __init__(self, cols, name):
        self.cols, self.key = cols, "_mirror_" + name

    def __get__(self, c, owner=None):
        if c is None:
            return self
        return c.__dict__[self.key]

    def __set__(self, c, values):
        c.__dict__[self.key] = values
        c._t.fl[c._r, self.cols] = values


class RowVector:
    """A creature's position or knock velocity, stored in its table row (columns cx, cy), with
    the API of the geometry's vector type. Arithmetic results are plain geometry vectors,
    computed by the geometry's own type, so the floats come out as before."""
    __slots__ = ("_c", "_cx", "_cy")

    def __init__(self, creature, cx, cy):
        self._c, self._cx, self._cy = creature, cx, cy

    @property
    def x(self):
        c = self._c
        return c._t.flv[c._r, self._cx]

    @x.setter
    def x(self, value):
        c = self._c
        c._t.flv[c._r, self._cx] = value

    @property
    def y(self):
        c = self._c
        return c._t.flv[c._r, self._cy]

    @y.setter
    def y(self, value):
        c = self._c
        c._t.flv[c._r, self._cy] = value

    def set(self, value):
        c = self._c
        t, r = c._t, c._r
        t.flv[r, self._cx] = value.x
        t.flv[r, self._cy] = value.y

    def vec(self):
        """A copy as the geometry's vector type."""
        c = self._c
        t, r = c._t, c._r
        return c.geometry.vec2(t.flv[r, self._cx], t.flv[r, self._cy])

    def __add__(self, other):
        return self.vec() + _plain(other)

    def __radd__(self, other):
        return other + self.vec()

    def __sub__(self, other):
        return self.vec() - _plain(other)

    def __rsub__(self, other):
        return other - self.vec()

    def __mul__(self, scalar):
        return self.vec() * scalar

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self.vec() / scalar

    def __iadd__(self, other):
        self.set(self.vec() + _plain(other))
        return self

    def __isub__(self, other):
        self.set(self.vec() - _plain(other))
        return self

    def distance_to(self, other):
        return self.vec().distance_to(_plain(other))

    def length(self):
        return self.vec().length()

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return (self.x, self.y)[i]

    def __iter__(self):
        yield self.x
        yield self.y

    def __eq__(self, other):
        try:
            return self.x == other[0] and self.y == other[1]
        except (TypeError, IndexError):
            return NotImplemented

    __hash__ = None

    def __repr__(self):
        return repr(self.vec())


def _plain(value):
    return value.vec() if type(value) is RowVector else value


class VectorField:
    """`pos` / `knock`: reads give the creature's RowVector, assignment copies x and y in."""
    __slots__ = ("cx", "cy", "key")

    def __init__(self, cx, cy, name):
        self.cx, self.cy, self.key = cx, cy, "_view_" + name

    def __get__(self, c, owner=None):
        if c is None:
            return self
        view = c.__dict__.get(self.key)
        if view is None:
            view = c.__dict__[self.key] = RowVector(c, self.cx, self.cy)
        return view

    def __set__(self, c, value):
        t, r = c._t, c._r
        t.flv[r, self.cx] = value.x
        t.flv[r, self.cy] = value.y

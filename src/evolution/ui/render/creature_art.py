"""Creature sprites, drawn facing +x. The art hangs on a `Spine`, so one drawing serves every
walk frame and the carcass's death pose. Lengths are in u, the body radius."""
import math
import random
from functools import cache
from typing import NamedTuple

import pygame

from .rig import Pen, Spine, band, knee, patch, ribbon, smooth_profile, stations, turned
from .shapes import blur
from .style import (
    CARNIVORE_PALETTE,
    HERBIVORE_PALETTE,
    SS,
    WALK_FRAMES,
    faded_palette,
    mix,
    tinted_palette,
)

BLACK = (0, 0, 0)


class Limb(NamedTuple):
    station: float   # where it joins the spine
    hip: float       # lateral offset of that joint
    upper: float     # segment lengths
    lower: float
    bend: float      # the knee points forward (+1) or back (-1)
    reach: float     # fore-aft swing of the foot while walking
    stance: float    # lateral offset of the walking foot


class Anatomy(NamedTuple):
    xs: tuple        # spine stations, tail tip first
    width: object    # station -> half-width of the body
    hips: float      # stations the walk bends the spine around
    shoulders: float
    limbs: tuple


class Foot(NamedTuple):
    x: float
    y: float
    heading: float
    lift: float      # 0 planted .. 1 mid-swing


class Pose(NamedTuple):
    spine: Spine
    feet: tuple      # one Foot per limb and side, sides in the order (1, -1)
    head_turn: float  # the carnivore's skull, relative to the end of its neck


_H_WIDTH = ((-1.9, 0.0), (-1.84, 0.03), (-1.35, 0.085), (-0.95, 0.2), (-0.5, 0.46), (-0.05, 0.55),
            (0.45, 0.45), (0.8, 0.25), (1.3, 0.155), (1.84, 0.115), (2.02, 0.17), (2.17, 0.155),
            (2.27, 0.1), (2.32, 0.0))
HERBIVORE = Anatomy(
    xs=stations(-1.9, 2.32, 0.07, [x for x, _ in _H_WIDTH]),
    width=smooth_profile(_H_WIDTH), hips=-0.5, shoulders=0.45,
    limbs=(Limb(0.42, 0.34, 0.21, 0.2, -1, 0.24, 0.6), Limb(-0.52, 0.36, 0.22, 0.2, 1, 0.24, 0.62)))

SKULL_BASE = 0.84   # station of the carnivore's skull joint; the neck runs on under the skull
_C_WIDTH = ((-2.3, 0.0), (-2.24, 0.03), (-1.55, 0.11), (-0.95, 0.23), (-0.4, 0.39), (0.1, 0.42),
            (0.5, 0.36), (0.84, 0.27), (1.1, 0.2), (1.2, 0.0))
CARNIVORE = Anatomy(
    xs=stations(-2.3, 1.2, 0.07, [x for x, _ in _C_WIDTH]),
    width=smooth_profile(_C_WIDTH), hips=-0.2, shoulders=0.5,
    limbs=(Limb(-0.2, 0.27, 0.3, 0.27, 1, 0.36, 0.47), Limb(0.5, 0.3, 0.14, 0.13, -1, 0.05, 0.4)))
ANATOMY = {False: HERBIVORE, True: CARNIVORE}

# Half of the carnivore's skull from above, from the neck joint to the snout tip.
_SKULL_HALF = ((-0.05, 0.0), (-0.03, 0.14), (0.0, 0.26), (0.05, 0.36), (0.18, 0.43), (0.34, 0.42), (0.5, 0.34), (0.64, 0.275),
               (0.8, 0.25), (0.93, 0.2), (1.01, 0.11), (1.04, 0.0))
_SKULL_MID = 0.52
jaw_width = smooth_profile(_SKULL_HALF[1:])


def skull_outline(k=1.0, dx=0.0, grow=0.0):
    """The carnivore's skull in its own frame (x forward from the neck joint), shrunk by k
    toward its centre, shifted by dx and widened by grow."""
    pts = [(_SKULL_MID + (x - _SKULL_MID) * k + dx + (grow if x > _SKULL_MID else -grow),
            -(y * k + grow if y else 0.0)) for x, y in _SKULL_HALF]
    return pts + [(x, -y) for x, y in reversed(pts[:-1])]


def _step(p, reach, duty=0.6):
    """(fore-aft foot offset, lift) at cycle position p: the foot is planted and slides back
    for `duty` of the cycle, then swings forward."""
    p %= 1.0
    if p < duty:
        return reach * (1 - 2 * p / duty), 0.0
    t = (p - duty) / (1 - duty)
    return reach * (2 * t * t * (3 - 2 * t) - 1), math.sin(math.pi * t)


@cache
def walk_pose(carn, frame):
    """The pose `frame` / WALK_FRAMES of the way through a stride. The +y fore foot (the +y
    hind foot of the biped) lands at frame 0."""
    body = ANATOMY[carn]
    cycle = frame / WALK_FRAMES
    ph = 2 * math.pi * cycle
    hips, shoulders = body.hips, body.shoulders
    tail = hips - body.xs[0]
    if carn:
        yaw = 0.085

        def sway(x):
            if x >= hips:
                return -yaw * math.cos(ph) * (x - hips)
            d = (hips - x) / tail
            return -yaw * math.cos(ph - 2.3 * d) * (x - hips) * (1 + 0.9 * d)

        offsets = (0.0, 0.5, 0.5, 0.0)   # arms swing against the legs
        head_turn = 0.6 * yaw * math.cos(ph)
    else:
        yaw = 0.13
        trunk, neck = shoulders - hips, body.xs[-1] - shoulders
        mid = (hips + shoulders) / 2

        def sway(x):
            if x < hips:
                d = (hips - x) / tail
                return yaw * math.cos(ph - 2.4 * d) * (x - hips) * (1 + 0.7 * d)
            if x <= shoulders:
                return -yaw * math.cos(ph) / trunk * ((x - mid) ** 2 - (trunk / 2) ** 2)
            d = (x - shoulders) / neck
            return (-yaw * math.cos(ph - 0.9 * d) * neck * (d - 0.82 * d * d)
                    + 0.05 * d * d * math.sin(ph - 1.6))

        offsets = (0.0, 0.5, 0.56, 0.06)
        head_turn = 0.0
    feet = []
    for i, limb in enumerate(body.limbs):
        for j, side in enumerate((1, -1)):
            dx, lift = _step(cycle + offsets[2 * i + j], limb.reach)
            feet.append(Foot(limb.station - 0.05 * limb.bend + dx, side * limb.stance, 0.0, lift))
    return Pose(Spine.swaying(body.xs, sway), tuple(feet), head_turn)


def limb_joints(body, pose):
    """(limb, side, hip, knee, foot) for every leg; hip and knee are points, foot a Foot."""
    out = []
    feet = iter(pose.feet)
    for limb in body.limbs:
        for side in (1, -1):
            foot = next(feet)
            hip = pose.spine.place(limb.station, side * limb.hip)
            a = pose.spine.at(limb.station)[2]
            prefer = (limb.bend * math.cos(a) - 0.7 * side * math.sin(a),
                      limb.bend * math.sin(a) + 0.7 * side * math.cos(a))
            out.append((limb, side, hip, knee(hip, foot[:2], limb.upper, limb.lower, prefer), foot))
    return out


def _segment(pen, p, p0, p1, r0, r1):
    pen.capsule(mix(p["leg"], BLACK, 0.45), p0, p1, r0 + 0.03, r1 + 0.03)
    pen.capsule(mix(p["leg"], p["rim"], 0.5), p0, p1, r0, r1)
    pen.capsule(mix(p["rim"], p["core"], 0.45), p0, p1, r0 * 0.5, r1 * 0.5)


def _skin(pen, spine, body, rim, core, steps=9):
    """The body from above: dark at the flanks, light along the back."""
    xs = body.xs
    pen.poly(mix(rim, BLACK, 0.5), ribbon(spine, body.width, xs, grow=0.035))
    for i in range(steps):
        t = i / (steps - 1)
        pen.poly(mix(rim, core, t ** 0.8), ribbon(spine, body.width, xs, k=1 - 0.72 * t))


def _scutes(pen, spine, width, x0, x1, step, edge, fill):
    """A row of bony plates along the back."""
    x = x0
    while x <= x1:
        px, py, a = spine.at(x)
        r = 0.018 + 0.06 * width(x)
        pen.oval(edge, (px, py), r + 0.016, r * 0.62 + 0.016, a, n=10)
        pen.oval(fill, (px - 0.012 * math.cos(a), py - 0.012 * math.sin(a)), r * 0.8, r * 0.45, a,
                 n=10)
        x += step


def _eye(pen, p, pt, rx, ry, rot, pupil, dead):
    pen.oval(mix(p["rim"], BLACK, 0.6), pt, rx + 0.02, ry + 0.02, rot, n=12)
    if dead:
        pen.oval(BLACK, pt, rx, ry * 0.25, rot, n=8)
        return
    pen.oval(p["eye"], pt, rx, ry, rot, n=12)
    pen.oval(p["pupil"], pt, pupil[0], pupil[1], rot, n=8)


def draw_herbivore(pen, p, pose, dead=False):
    """A long-necked browser: column legs, a blotched back and a whip of a tail."""
    body, spine = HERBIVORE, pose.spine
    width = body.width
    dark = mix(p["leg"], BLACK, 0.45)
    for _limb, _side, hip, kn, foot in limb_joints(body, pose):
        r = 0.15 * (1 + 0.1 * foot.lift)
        pad = foot[:2]
        pen.oval(dark, pad, r + 0.03, r * 0.94 + 0.03, foot.heading)
        pen.oval(mix(p["leg"], p["rim"], 0.5), pad, r, r * 0.94, foot.heading)
        for a in (-0.85, -0.3, 0.3, 0.85):
            nail = turned(pad, foot.heading + a, [(r * 0.98, 0.0)])[0]
            pen.oval(p["claw"], nail, 0.05, 0.036, foot.heading + a, n=8)
        _segment(pen, p, kn, pad, 0.13, 0.12)
        _segment(pen, p, hip, kn, 0.17, 0.135)

    _skin(pen, spine, body, p["rim"], p["core"])

    over = pen.layer()
    rnd = random.Random(11)
    blotch = mix(p["stripe"], p["rim"], 0.35)
    x = -1.3
    while x < 1.8:
        for side in (1, -1):
            jx, jr = rnd.uniform(-0.03, 0.03), rnd.uniform(0.0, 0.025)
            over.poly(blotch + (165,), patch(spine, width, x + side * 0.055 + jx, side * 0.4,
                                             0.075 + jr, 0.2 + jr * 2))
            over.poly(blotch + (120,), patch(spine, width, x + 0.11 - side * 0.05 + jx, side * 0.76,
                                             0.05 + jr, 0.12))
        x += 0.215
    for x in (-1.78, -1.66, -1.53, -1.4):
        over.poly(blotch + (190,), band(spine, width, x, 0.03, reach=0.95))
    for _ in range(46):
        x, f = rnd.uniform(-1.3, 1.85), rnd.uniform(-0.88, 0.88)
        over.oval(p["spot"] + (150,), spine.place(x, f * width(x)), 0.022, 0.018, n=6)
    for station, lat in ((body.shoulders - 0.02, 0.24), (body.hips + 0.05, 0.25)):
        for side in (1, -1):   # shoulder blades and hip bones under the skin
            over.oval(p["spot"] + (46,), spine.place(station, side * lat), 0.2, 0.11,
                      spine.at(station)[2] + side * 0.25)
    over.poly(p["spot"] + (70,), patch(spine, width, 2.16, 0, 0.11, 0.55))
    pen.stamp(over)

    _scutes(pen, spine, width, -1.25, 1.78, 0.155, mix(p["stripe"], BLACK, 0.3), p["crest"])

    ha = spine.at(2.03)[2]
    for side in (1, -1):
        _eye(pen, p, spine.place(2.03, side * 0.14), 0.045, 0.034, ha, (0.026, 0.026), dead)
        pen.oval(p["nose"], spine.place(2.25, side * 0.045), 0.022, 0.016, ha, n=8)


def draw_carnivore(pen, p, pose, dead=False):
    """A two-legged hunter: heavy skull, striped back, the tail balancing the head."""
    body, spine = CARNIVORE, pose.spine
    width = body.width
    dark = mix(p["leg"], BLACK, 0.45)
    joints = limb_joints(body, pose)
    for limb, side, hip, kn, foot in joints[2:]:   # arms
        hand = foot[:2]
        for a in (-0.4, 0.35):
            tip = turned(hand, foot.heading + a, [(0.1, 0.0)])[0]
            pen.capsule(p["claw"], hand, tip, 0.03, 0.012)
        _segment(pen, p, kn, hand, 0.055, 0.045)
        _segment(pen, p, hip, kn, 0.075, 0.055)
    for limb, side, hip, kn, foot in joints[:2]:
        ankle = foot[:2]
        grow = 1 + 0.12 * foot.lift
        toes = [(-0.5, 0.21), (0.0, 0.28), (0.5, 0.21), (math.pi - side * 0.5, 0.09)]
        for a, length in toes:
            tip = turned(ankle, foot.heading + a, [(length * grow, 0.0)])[0]
            claw = turned(ankle, foot.heading + a, [((length + 0.08) * grow, 0.0)])[0]
            pen.capsule(dark, ankle, tip, 0.085, 0.055)
            pen.capsule(p["claw"], tip, claw, 0.035, 0.008)
        for a, length in toes:
            tip = turned(ankle, foot.heading + a, [(length * grow, 0.0)])[0]
            pen.capsule(mix(p["leg"], p["rim"], 0.5), ankle, tip, 0.055, 0.03)
        _segment(pen, p, kn, ankle, 0.1, 0.07)
        _segment(pen, p, hip, kn, 0.19, 0.12)

    _skin(pen, spine, body, p["rim"], p["core"])

    over = pen.layer()
    stripe = p["stripe"]
    x = -2.05
    while x < 0.75:
        k = 0.5 + 0.5 * min(1.0, width(x) / 0.3)
        over.poly(stripe + (185,), band(spine, width, x, 0.045 * k + 0.012, reach=0.9,
                                        sweep=0.16 * k))
        x += 0.2 if x > -1.2 else 0.17
    ridge = tuple(x for x in body.xs if -2.0 <= x <= 0.8)
    over.poly(mix(stripe, p["rim"], 0.3) + (150,), ribbon(spine, width, ridge, k=0.16))
    rnd = random.Random(5)
    for _ in range(30):
        x, f = rnd.uniform(-1.6, 0.8), rnd.uniform(-0.85, 0.85)
        over.oval(p["snout"] + (120,), spine.place(x, f * width(x)), 0.02, 0.016, n=6)
    for side in (1, -1):   # the hips showing under the skin
        over.oval(p["snout"] + (40,), spine.place(body.hips, side * 0.2), 0.24, 0.1,
                  spine.at(body.hips)[2] + side * 0.2)
    pen.stamp(over)
    _scutes(pen, spine, width, -1.9, 0.7, 0.13, mix(stripe, BLACK, 0.2), mix(stripe, p["rim"], 0.6))

    nx, ny, na = spine.at(SKULL_BASE)
    draw_skull_skin(pen, p, (nx, ny), na + pose.head_turn, dead)


def draw_skull_skin(pen, p, joint, angle, dead=False):
    def place(pts):
        return turned(joint, angle, pts)

    pen.poly(mix(p["rim"], BLACK, 0.5), place(skull_outline(grow=0.035)))
    pen.poly(p["rim"], place(skull_outline()))
    for side in (1, -1):
        for x in (0.47, 0.57, 0.67, 0.77, 0.87, 0.96):
            y = jaw_width(x)
            pen.poly(p["fang"], place([(x - 0.03, side * (y - 0.03)), (x + 0.03, side * (y - 0.03)),
                                       (x + 0.008, side * (y + 0.04))]))
    for i, k in enumerate((0.93, 0.8, 0.66, 0.5)):
        pen.poly(mix(p["rim"], p["core"], 0.3 + 0.23 * i), place(skull_outline(k, -0.02 * i)))
    snout = place([(0.74, 0.0)])[0]
    pen.oval(mix(p["core"], p["snout"], 0.55), snout, 0.26, 0.12, angle)
    pen.oval(p["snout"], place([(0.8, 0.0)])[0], 0.17, 0.07, angle)
    for x in (0.5, 0.62, 0.74, 0.86):   # the knobbly ridge down the snout
        pen.oval(mix(p["rim"], p["core"], 0.5), place([(x, 0.0)])[0], 0.035, 0.028, angle, n=8)
    for side in (1, -1):
        pen.oval(mix(p["core"], p["snout"], 0.3), place([(0.12, side * 0.2)])[0], 0.1, 0.12,
                 angle, n=12)
        eye = place([(0.3, side * 0.29)])[0]
        _eye(pen, p, eye, 0.075, 0.045, angle + side * 0.3, (0.02, 0.042), dead)
        pen.oval(p["stripe"], place([(0.31, side * 0.22)])[0], 0.15, 0.055, angle + side * 0.28,
                 n=12)
        pen.oval(p["nose"], place([(0.95, side * 0.07)])[0], 0.032, 0.02, angle + side * 0.4, n=8)


DRAW = {False: draw_herbivore, True: draw_carnivore}


def tinted(carn, cb, hl):
    return faded_palette(tinted_palette(CARNIVORE_PALETTE if carn else HERBIVORE_PALETTE, cb), hl)


def silhouette_shadow(sprite, alpha):
    shadow = blur(sprite, 2)
    shadow.fill((0, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
    shadow.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
    return shadow


def build_creature(carn, sb, hl, frame, cb):
    """Unrotated sprite facing +x, its shadow silhouette, and the x offset of the
    sprite centre from the creature origin."""
    x0, x1, y_half = (-2.36, 1.96, 0.86) if carn else (-1.96, 2.38, 0.86)
    w = math.ceil((x1 - x0) * sb) + 2
    h = math.ceil(2 * y_half * sb) + 2
    ox = 1 - x0 * sb
    big = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
    DRAW[carn](Pen(big, ox * SS, h * SS / 2, sb * SS), tinted(carn, cb, hl), walk_pose(carn, frame))
    sprite = pygame.transform.smoothscale(big, (w, h))
    return sprite, silhouette_shadow(sprite, 110), w / 2 - ox

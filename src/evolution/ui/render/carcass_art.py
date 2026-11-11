"""Carcass sprites, facing +x: the dead creature in its death pose, opened up and then stripped
to the skeleton as its meat goes. Lengths are in u, the body radius."""
import math
import random
from functools import cache

import pygame

from .creature_art import (
    ANATOMY,
    DRAW,
    SKULL_BASE,
    Foot,
    Pose,
    jaw_width,
    limb_joints,
    silhouette_shadow,
    skull_outline,
    tinted,
)
from .rig import Pen, Spine, patch, smooth_profile, turned
from .shapes import blur, ellipse_pts
from .style import CARCASS_LEVELS, SS, mix

BONE = (230, 222, 200)
BONE_EDGE = (110, 98, 80)
SOCKET = (78, 64, 50)
TOOTH = (250, 248, 236)
SKIN_UNTIL = 0.5   # share of the meat below which no skin is left

# Spine heading (radians) by station: back arched, neck and tail thrown to opposite sides.
_CURL = {False: ((-1.9, -1.0), (-1.2, -0.5), (-0.5, -0.08), (0.45, 0.0), (1.2, -0.7), (2.32, -1.15)),
         True: ((-2.3, -1.0), (-1.4, -0.45), (-0.4, -0.05), (0.3, 0.0), (0.84, -0.5), (1.2, -0.6))}
# Where each limb lies, per limb and side (1, -1): (station past the joint, lateral reach,
# foot heading).
_SPLAY = {False: ((0.14, 0.72, 0.7), (-0.1, 0.73, -2.2), (-0.16, 0.74, 2.3), (0.12, 0.75, -0.9)),
          True: ((0.3, 0.72, 0.3), (-0.34, 0.68, -2.6), (0.16, 0.42, 0.6), (0.1, 0.44, -1.2))}
_RIB_CAGE = {False: (0.4, -0.55), True: (0.38, -0.45)}
_HERBIVORE_SKULL = 1.86


@cache
def death_pose(carn):
    body = ANATOMY[carn]
    spine = Spine.bent(body.xs, smooth_profile(_CURL[carn]))
    feet = []
    splay = iter(_SPLAY[carn])
    for limb in body.limbs:
        for side in (1, -1):
            dx, lat, heading = next(splay)
            x, y = spine.place(limb.station + dx, side * lat)
            feet.append(Foot(x, y, spine.at(limb.station)[2] + heading, 0.0))
    return Pose(spine, tuple(feet), 0.0)


def _bone(pen, pt, rx, ry, rot=0.0):
    pen.oval(BONE_EDGE, pt, rx + 0.025, ry + 0.025, rot, n=14)
    pen.oval(BONE, pt, rx, ry, rot, n=14)


def _rod(pen, pts, width, knobs=True):
    """A long bone through `pts`, with knobbed ends."""
    pen.line(BONE_EDGE, pts, width + 0.05)
    if knobs:
        for pt in (pts[0], pts[-1]):
            pen.oval(BONE_EDGE, pt, width * 0.75 + 0.025, width * 0.75 + 0.025, n=12)
    pen.line(BONE, pts, width)
    if knobs:
        for pt in (pts[0], pts[-1]):
            pen.oval(BONE, pt, width * 0.75, width * 0.75, n=12)


def _carnivore_skull(pen, joint, angle):
    pen.poly(BONE_EDGE, turned(joint, angle, skull_outline()))
    pen.poly(BONE, turned(joint, angle, skull_outline(0.9)))
    for side in (1, -1):
        for x in (0.47, 0.57, 0.67, 0.77, 0.87, 0.96):
            y = jaw_width(x)
            pen.poly(TOOTH, turned(joint, angle, [(x - 0.035, side * (y - 0.03)),
                                                  (x + 0.035, side * (y - 0.03)),
                                                  (x + 0.01, side * (y + 0.05))]))
        for (x, y), (rx, ry), rot in (((0.3, 0.26), (0.075, 0.055), 0.3),    # orbit
                                      ((0.6, 0.15), (0.11, 0.04), 0.1),      # antorbital opening
                                      ((0.1, 0.2), (0.06, 0.04), -0.4),      # temporal opening
                                      ((0.93, 0.07), (0.035, 0.02), 0.0)):   # nostril
            pen.poly(SOCKET, turned(joint, angle, ellipse_pts(x, side * y, rx, ry, side * rot, 16)))


def _herbivore_skull(pen, joint, angle):
    pen.poly(BONE_EDGE, turned(joint, angle, ellipse_pts(0.22, 0, 0.26, 0.165)))
    pen.poly(BONE, turned(joint, angle, ellipse_pts(0.22, 0, 0.23, 0.135)))
    for side in (1, -1):
        for (x, y), (rx, ry) in (((0.16, 0.085), (0.045, 0.032)), ((0.38, 0.045), (0.025, 0.016))):
            pen.poly(SOCKET, turned(joint, angle, ellipse_pts(x, side * y, rx, ry, n=14)))


def _skeleton(pen, carn, pose):
    body, spine = ANATOMY[carn], pose.spine
    width = body.width
    for limb, _side, hip, kn, foot in limb_joints(body, pose):
        thick = 0.035 if limb.upper < 0.18 else 0.08
        ankle = foot[:2]
        _rod(pen, [hip, kn], thick)
        _rod(pen, [kn, ankle], thick)
        toes = (-0.45, 0.0, 0.45) if carn else (-0.75, -0.25, 0.25, 0.75)
        for a in toes[:2] if thick < 0.05 else toes:
            tip = turned(ankle, foot.heading + a, [(0.75 * limb.lower if carn else 0.15, 0.0)])[0]
            _rod(pen, [ankle, tip], thick * 0.45, knobs=False)
            pen.oval(BONE_EDGE, tip, thick * 0.35, thick * 0.35, n=8)
    for station, lat, rx, ry, rot in ((body.hips + 0.05, 0.17, 0.22, 0.09, 0.35),      # pelvis
                                      (body.shoulders - 0.12, 0.24, 0.17, 0.06, -0.6)):  # shoulder
        for side in (1, -1):
            _bone(pen, spine.place(station, side * lat), rx, ry, spine.at(station)[2] + side * rot)
    x_from, x_to = _RIB_CAGE[carn]
    for k in range(8):
        x = x_from + (x_to - x_from) * k / 7
        for side in (1, -1):
            _rod(pen, [spine.place(x - 0.22 * t * t,
                                   side * (0.06 + width(x) * 0.88 * math.sin(t * math.pi / 2)))
                       for t in (0.0, 0.25, 0.5, 0.75, 1.0)], 0.045, knobs=False)
    neck_end = SKULL_BASE if carn else _HERBIVORE_SKULL
    x = body.xs[0] + 0.04
    while x <= neck_end:
        px, py, a = spine.at(x)
        w = 0.018 + 0.12 * min(width(x), 0.5)
        _bone(pen, (px, py), 0.02, w * 1.7, a)
        _bone(pen, (px, py), 0.05, w * 0.75, a)
        x += 0.11
    px, py, a = spine.at(neck_end)
    (_carnivore_skull if carn else _herbivore_skull)(pen, (px, py), a)


def _opening(spine, width, rnd, size):
    """Ragged outline of the hole eaten into the flank and belly; `size` 0..1."""
    cx, cf = -0.1 + 0.1 * size, 0.42 * (1 - size)
    rx, rf = 0.2 + 0.62 * size, 0.42 + 0.5 * size
    pts = []
    for i in range(22):
        a = 2 * math.pi * i / 22
        k = rnd.uniform(0.82, 1.08)
        sx = cx + rx * k * math.cos(a)
        pts.append(spine.place(sx, max(-0.97, min(0.97, cf + rf * k * math.sin(a))) * width(sx)))
    return pts


def _raw(pen, spine, width, outline, meat):
    """Flesh inside `outline`: dark at the torn edge, wet red in the middle."""
    cx = sum(x for x, _ in outline) / len(outline)
    cy = sum(y for _, y in outline) / len(outline)
    rim, mid = mix((60, 26, 22), (92, 12, 16), meat), mix((120, 60, 48), (178, 36, 40), meat)
    for i in range(5):
        t = i / 4
        k = 1 - 0.5 * t
        pen.poly(mix(rim, mid, t), [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in outline])


def _wounds(pen, spine, width, rnd):
    """Bite marks on an otherwise whole body."""
    for x, f in ((0.62, -0.5), (-0.1, 0.72), (-0.42, 0.6)):
        for k in range(3):
            sx = x + k * 0.07
            p0 = spine.place(sx, (f - 0.22) * width(sx))
            p1 = spine.place(sx + 0.06 + rnd.uniform(0, 0.04), (f + 0.2) * width(sx))
            pen.capsule((70, 8, 12), p0, p1, 0.028, 0.012)
            pen.capsule((168, 30, 34), p0, p1, 0.014, 0.004)


def _draw_body(pen, carn, pose, meat, cb, rnd):
    body, spine = ANATOMY[carn], pose.spine
    width = body.width
    if meat > SKIN_UNTIL:
        DRAW[carn](pen, tinted(carn, cb, 0), pose, dead=True)
        size = (1 - meat) / (1 - SKIN_UNTIL)
        if size <= 0:
            _wounds(pen, spine, width, rnd)
            return
        hole = _opening(spine, width, rnd, size)
        _raw(pen, spine, width, hole, meat)
        bones, mask = pen.layer(), pen.layer()
        _skeleton(bones, carn, pose)
        mask.poly((255, 255, 255, 255), hole)
        pen.stamp(bones, mask)
        gloss = pen.layer()
        gloss.poly((150, 24, 30, int(150 * (1 - size))), hole)
        pen.stamp(gloss)
        return
    left = meat / SKIN_UNTIL
    if left > 0:
        for x0, x1, k in ((body.xs[0] + 0.5, body.hips, 0.45), (body.shoulders, body.xs[-1] - 0.5,
                                                                0.5)):
            xs = [x for x in body.xs if x0 <= x <= x1]   # sinew along the tail and the neck
            sinew = mix((84, 44, 34), (120, 22, 26), left)
            pen.poly(sinew, [spine.place(x, k * left * width(x)) for x in xs]
                     + [spine.place(x, -k * left * width(x)) for x in reversed(xs)])
        _raw(pen, spine, width, patch(spine, width, -0.05, 0.0, 0.3 + 0.5 * left, 0.5 + 0.4 * left,
                                      n=20), meat)
    _skeleton(pen, carn, pose)
    if left > 0:
        gloss = pen.layer()
        for i in range(4):
            t = i / 3
            gloss.poly(mix((96, 16, 20), (184, 46, 46), t) + (int(210 * left ** 1.3),),
                       patch(spine, width, -0.08, 0.0, (0.2 + 0.45 * left) * (1 - 0.35 * t),
                             (0.25 + 0.45 * left) * (1 - 0.35 * t), n=16))
        pen.stamp(gloss)


def build_carcass(carn, sb, level, cb):
    """Unrotated carcass sprite facing +x with its blood stain, the shadow of the body, and the
    x offset of the sprite centre from the carcass position. `level` 0 is a fresh kill,
    CARCASS_LEVELS - 1 bare bones."""
    meat = 1 - level / (CARCASS_LEVELS - 1)
    x0, x1, y_half = (-2.3, 2.1, 1.2) if carn else (-2.0, 2.2, 1.3)
    w = math.ceil((x1 - x0) * sb) + 2
    h = math.ceil(2 * y_half * sb) + 2
    ox = 1 - x0 * sb
    size = (w * SS, h * SS)
    rnd = random.Random(sb * 31 + carn * 7 + level)

    stain = Pen(pygame.Surface(size, pygame.SRCALPHA), ox * SS, h * SS / 2, sb * SS)
    blood = mix((70, 52, 34), (110, 12, 14), meat) + (int(110 + 110 * meat),)
    spread = 0.55 + 0.45 * min(1.0, 2 * (1 - meat) + 0.3)
    for _ in range(8):
        stain.oval(blood, (rnd.uniform(-0.7, 0.45), rnd.uniform(-0.4, 0.4)),
                   rnd.uniform(0.3, 0.62) * spread, rnd.uniform(0.2, 0.42) * spread,
                   rnd.uniform(0, math.pi))

    body = stain.layer()
    _draw_body(body, carn, death_pose(carn), meat, cb, rnd)

    sprite = pygame.transform.smoothscale(blur(stain.surf, 6), (w, h))
    body = pygame.transform.smoothscale(body.surf, (w, h))
    sprite.blit(body, (0, 0))
    return sprite, silhouette_shadow(body, 70), w / 2 - ox

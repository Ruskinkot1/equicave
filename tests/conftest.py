import numpy as np
import pytest


def shell(radius=9.0, n_ring=36, zs=(-8, -4, 0, 4, 8), hole=None):
    """Atoms on a cylinder: a tube cavity along z. `hole`: angular window (rad) left open."""
    pts = []
    for z in zs:
        for a in np.linspace(0, 2 * np.pi, n_ring, endpoint=False):
            if hole is not None and hole[0] <= a <= hole[1]:
                continue
            pts.append((radius * np.cos(a), radius * np.sin(a), float(z)))
    return np.array(pts)


def solid_ball(radius=12.0, spacing=1.8, seed=0):
    """A dense ball of atoms (no cavity)."""
    g = np.arange(-radius, radius + spacing, spacing)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return pts[np.linalg.norm(pts, axis=1) <= radius]


def ball_with_pocket(radius=12.0, pocket_center=(0, 0, 6.0), pocket_r=4.5, spacing=1.8):
    """Dense ball with a spherical cavity carved near its surface (a pocket open on one side)."""
    pts = solid_ball(radius, spacing)
    return pts[np.linalg.norm(pts - np.asarray(pocket_center), axis=1) > pocket_r]


@pytest.fixture
def pocket_ball():
    return ball_with_pocket()

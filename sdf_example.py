from utils.sdf import *

@sdf3
def box(center=np.zeros(3), half_size=np.ones(3)):
    center = np.array(center, dtype=float)
    half_size = np.array(half_size, dtype=float)

    def f(p):
        d = np.abs(p - center) - half_size
        outside = np.maximum(d, 0)
        inside = np.max(d, axis=1)
        outside_dist = np.linalg.norm(outside, axis=1)
        return np.where(np.any(d > 0, axis=1), outside_dist, inside)

    return f

f = box((2, 0, 0), (1, 1, 1))
f.save("box.stl")
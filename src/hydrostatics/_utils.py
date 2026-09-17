import gmsh
import numpy as np
from numpy.typing import NDArray

vec3D = np.ndarray[(3,), np.dtype[np.float64]]
mat3D = np.ndarray[(3,3), np.dtype[np.float64]]

def norm(a: NDArray) -> float:
    return np.sqrt(np.dot(a,a))

def normalized(a: NDArray) -> NDArray:
    return a/norm(a)

def rot_xy(theta) -> mat3D:
    return np.array((   (np.cos(theta),     -np.sin(theta), 0   ),
                        (np.sin(theta),     np.cos(theta),  0   ),
                        (0,                 0,              1.  )))

def rot_xz(theta) -> mat3D:
    return np.array((   (np.cos(theta), 0,      -np.sin(theta)  ),
                        (0,             1.,     0               ),
                        (np.sin(theta), 0,      np.cos(theta)   )))

def rot_yz(theta) -> mat3D:
    return np.array((   (1.,    0,              0               ),
                        (0,     np.cos(theta),  -np.sin(theta)  ),
                        (0,     np.sin(theta),  np.cos(theta)   )))

def rot_matrix(ax: vec3D, theta: float) -> mat3D:
    u = normalized(ax)
    I = np.eye(3)
    ux = np.array(( (0, -u[2], u[1]),
                    (u[2], 0, -u[0]),
                    (-u[1], u[0], 0)))
    uxu = u[:,None]*u
    return np.cos(theta)*I + np.sin(theta)*ux + (1.-np.cos(theta))*uxu
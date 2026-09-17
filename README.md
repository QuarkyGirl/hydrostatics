# [Python Sailboat Hydrostatics Library](https://www.nellhall.com/hydrostatics)

[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://www.nellhall.com/hydrostatics)

A work-in-progress package for evaluting hydrostatic quantities for
sailboat design and optimization. Many of the meshing functionalities
are implemented using Gmsh.

Current functionalities include

- Import STL and remesh as 3d volume.
- Solve for waterline and trim given displacement, heel.
- Calculate volume, center of buoyancy, righting moment given
  displacment, heel.
- Plot GZ curves.
- Plot meshes.

The central object used is `Hull`, representing a 3D hull generated from
an STL file. `Hull.split` and `Hull.solve` generate `SplitHull` and
`SolveHull` objects, respectively, which represent the area of the hull
below a waterplane. `SplitHull` and `SolveHull` share methods that allow
for calculation of hydrostatic quantities.

## Basic Usage

The following examples may be found in
`examples/example_basic_usage.py`.

Load an STL file of a hull, create a 3d volumetric mesh, and plot the
result.

``` python
from hydrostatics import Hull
import numpy as np
from matplotlib import pyplot as plt

print("1. load and plot hull from stl")
hull = Hull("example-hull.stl", center_of_mass = (9.5,0,2))
hull.plot()
```

Find the displacement and center of buoyancy for a given waterline,
solving for neutral trim. Note that submerged regions are plotted as a
Gmsh post-processing view.

``` python
print("\n2. get hydrostatics for given waterline at zero heel, neutral trim")
rho = 62.4 # density of water in chosen units
waterline = 3.0 # waterline to cut at
submerged = hull.solve(waterline=waterline, rho=rho)
print(f"displacement = {submerged.displacement}")
print(f"center of buoyancy = {submerged.center_of_buoyancy}")
submerged.plot()
```

Find the center of buoyancy for a given displacement and heel angle,
solving for neutral trim, and plot the shape of the submerged region.

``` python
print("\n3. get hydrostatics for given displacement at non-zero heel, neutral trim")
disp = 2300 # target displacement
heel = np.pi*30/180 # heel angle
heeled = hull.solve(displacement=disp, heel=heel, rho=rho)
print(f"displacement = {heeled.displacement}")
print(f"center of buoyancy = {heeled.center_of_buoyancy}")
heeled.plot()
```

Calculate righting moment for a variety of heel angles (i.e. GZ
stability curves) and plot the results.

``` python
print("\n4. get stability curves for given displacement at a range heels, neutral trim")
thetas = np.linspace(0,np.pi,num=40)
heeled_hulls = [hull.solve(displacement=disp, heel=theta, rho=rho) 
   for theta in thetas]

# only get the local x-component:
heeling_moments = [h.righting_moment().dot(h.x) for h in heeled_hulls]
plt.figure()
plt.plot(thetas, heeling_moments)
plt.show()
```

## Dependencies

- [NumPy](https://numpy.org)
- [SciPy](https://scipy.org)
- [Gmsh](https://gmsh.info)

## Future directions

This package is very much a work in progress. Future planned
functionalities include:

- Waterplane characteristics
- Wetted surface area
- Block and prismatic coefficients
- Curve of station areas

from hydrostatics import Hull
import numpy as np
from matplotlib import pyplot as plt

print("1. load and plot hull from stl")
hull = Hull("example-hull.stl", center_of_mass = (9.5,0,2))
hull.plot()

print("\n2. get hydrostatics for given waterline at zero heel, neutral trim")
rho = 62.4 # density of water in chosen units
waterline = 3.0 # waterline to cut at
submerged = hull.solve(waterline=waterline, rho=rho)
print(f"displacement = {submerged.displacement}")
print(f"center of buoyancy = {submerged.center_of_buoyancy}")
submerged.plot()

print("\n3. get hydrostatics for given displacement at non-zero heel, neutral trim")
disp = 2300 # target displacement
heel = np.pi*30/180 # heel angle
heeled = hull.solve(displacement=disp, heel=heel, rho=rho)
print(f"displacement = {heeled.displacement}")
print(f"center of buoyancy = {heeled.center_of_buoyancy}")
heeled.plot()

print("\n4. get stability curves for given displacement at a range heels, neutral trim")
thetas = np.linspace(0,np.pi,num=40)
heeled_hulls = [hull.solve(displacement=disp, heel=theta, rho=rho) 
    for theta in thetas]

# only get the local x-component:
heeling_moments = [h.righting_moment().dot(h.x) for h in heeled_hulls]
plt.figure()
plt.plot(thetas, heeling_moments)
plt.show()
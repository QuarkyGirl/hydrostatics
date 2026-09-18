from functools import cached_property
from warnings import warn

import gmsh
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.optimize import root, root_scalar

from . import _views, _utils
from ._gmsh import GmshObject, GmshModel

class SplitHull:
    pass

class SolvedHull:
    pass

class Hull(GmshModel):
    """
    Load mesh from stl file, create 3D mesh, and calculate waterplanes.
    Creates a new Gmsh model.

    Parameters
    ----------
    filename : str
        Filename of mesh that defines hull geometry. Must point to an stl file
        composed of closed, watertight surfaces, with longitudinal direction of
        hull along x-axis.
    center_of_mass: array_like, shape(3,)
        Cartesian coordinates of center of mass of hull.
    model_name: str or None
        Name of model to be created in Gmsh. Must be unique. If None (default), 
        filename will be used as model name.

    Attributes
    ----------
    model_name
    volume
    center_of_mass : ndarray, shape (3,)
        Cartesian coordates of center of mass of hull set by user. Can be
        modified.
    """
    _all_model_names = set()

    def __init__(
            self, 
            filename: str,
            center_of_mass: ArrayLike,
            model_name: str | None = None) -> None:
        super().__init__(filename if model_name is None else model_name)
        self.center_of_mass = np.asarray(center_of_mass)
        self._import_stl(filename)
        self._volume_field = _views.Field(lambda x: np.ones((x.shape[0],1)), vectorized=True)
        if self._check_warn():
            warn("Mesh generated from imported STL file " + filename + " has "
                "no 3D elements, suggesting STL may need repair (e.g. may not"
                "be watertight) or mesh generation may have failed.", 
                RuntimeWarning, stacklevel=2)
    
    @property
    def volume(self) -> float:
        """
        Total volume of hullform, not accounting for any sort of waterplane.

        Returns
        -------
        volume : float
            Total volume of hull.
        """
        self._check_load()
        return self._volume
    
    @cached_property
    def _volume(self) -> float:
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        print(element_tags)
        result = 0.
        for tags in element_tags:
            volumes = gmsh.model.mesh.getElementQualities(
                elementTags=list(tags), 
                qualityName="volume"
            )
            result += np.sum(volumes)
        print(result)
        return result


    def split(self, normal: ArrayLike, waterline: float, rho: float = 1.) -> SplitHull:
        """
        Split hull along waterplane defined by ``normal.dot(x) == waterline``.

        Parameters
        ----------
        normal : array_like, shape (3,)
            Vector normal to plane. Will be normalized.
        waterline : float
            Constant term in plane equation.
        rho : float
            Density of water. Default is 1.

        Returns
        -------
        split_hull : SplitHull
            Portion of hull below waterplane.
        """
        self._check_load()
        return SplitHull(self, normal, waterline, rho=rho)
    
    def solve(
            self,
            waterline: float | None = None,
            displacement: float | None = None,
            heel: float = 0.,
            trim: float = 0.,
            solve_trim: bool = True,
            rho: float = 1.,
            tol: float = 1.e-5) -> SolvedHull:
        """
        Solve for waterplane given some set of constraints. Either
        ``waterline`` or ``displacement`` must be provided, but not both.
        If ``solve_trim = True``, ``trim`` will be solved for equilibrium 
        (i.e., zero longitudinal righting moment); otherwise, the argument 
        ``trim`` will be used.

        Parameters
        ----------
        waterline : float or None
            Constant term in plane equation.
        displacement : float or None
            Target displacement of hull.
        heel : float
            Heel of hull. Default ``0``.
        trim : float
            Trim angle (x-z rotation) of waterplane. If `solve_trim`` is True,
            ``trim`` is used as initial state of solver. Default ``0``.
        solve_trim : bool
            If True, trim will be solved for neutral (zero longitudinal 
            righting moment). If False, ``trim`` argument is used. Default 
            True.
        rho : float
            Density of water
        tol : float
            Tolerance passed to SciPy solver.

        Returns
        -------
        solved_hull : SolvedHull
            Portion of hull below waterplane. Includes details of solver
            termination.
        """
        self._check_load()
        if solve_trim:
            if (displacement is not None) and (waterline is None):
                return self._solve_waterline_trim(displacement, heel, trim, rho, tol)
            elif (waterline is not None) and (displacement is None):
                return self._solve_trim(waterline, heel, trim, rho, tol)
            else:
                raise ValueError(
                    "Exactly one of waterline and displacment must be provided"
                )
        else:
            normal = self._normal_vector(heel, trim)
            if (displacement is not None) and (waterline is None):
                return self._solve_waterline(displacement, normal, rho, tol)
            elif (waterline is not None) and (displacement is None):
                return self._split(normal, waterline, rho=rho)
            else:
                raise ValueError(
                    "Exactly one of waterline and displacment must be provided"
                )
    
    def plot(self) -> None:
        """
        Plot meshes in hull model using Gmsh.
        """
        self._check_load()
        _views.hide_all()
        gmsh.fltk.run()

    def _normal_vector(self, heel: float, trim: float):
        return _utils.rot_yz(heel) @ _utils.rot_xz(trim) @ np.array((0,0,1))

    def _solve_waterline_trim(
            self,
            displacement: float, 
            heel: float,
            initial_trim: float,
            rho: float,
            tol: float) -> SolvedHull:
        def objective_function(x):
            waterline, trim = x
            split = self.split(self._normal_vector(heel, trim), waterline, rho=rho)
            result = (split.displacement - displacement, split.righting_moment().dot(split.y))
            return result

        approx_normal = self._normal_vector(heel, initial_trim)
        box_lb, box_ub = self._bbox()
        
        initial_tol = max(tol,1.e-2)
        initial_sol = self._solve_waterline(displacement,approx_normal, 
            rho, initial_tol)
        waterline_guess = initial_sol.waterline

        sol = root(objective_function, 
            x0=(waterline_guess,initial_trim), tol=tol, method='hybr')
        waterline, trim = sol.x
        return SolvedHull(self, self._normal_vector(heel, trim), 
            waterline, rho, sol.success, sol.message)
    
    def _solve_trim(
            self,
            waterline: float, 
            heel: float,
            initial_trim: float,
            rho: float,
            tol: float) -> SolvedHull:
        def objective_function(trim):
            split = self.split(self._normal_vector(heel, trim), waterline, rho=rho)
            result = split.righting_moment()[1]
            return result

        approx_normal = self._normal_vector(heel, 0.)
        box_lb, box_ub = self._bbox()

        sol = root_scalar(objective_function, 
            x0=initial_trim, x1=initial_trim+0.01, xtol=tol)
        trim = sol.root
        return SolvedHull(self, self._normal_vector(heel, trim), 
            waterline, rho, sol.converged, sol.flag)

    def _solve_waterline(
            self,
            displacement: float,
            normal: ArrayLike,
            rho: float,
            tol: float) -> SolvedHull:
        _normal = _utils.normalized(np.asarray(normal))
        lb, ub = self._bbox()
        coord_choices = np.array((lb,ub)).T
        corners = np.array(np.meshgrid(*coord_choices)).reshape(3,-1).T
        corner_wls = corners @ _normal
        bracket = (np.amin(corner_wls), np.amax(corner_wls))
        def objective_function(w_l):
            split = self.split(_normal, w_l, rho=rho)
            result = split.displacement - displacement
            return result
        sol = root_scalar(objective_function, bracket=bracket)
        waterline = sol.root
        return SolvedHull(self, _normal, waterline, rho, sol.converged, sol.flag)
    
    def _import_stl(self, filename: str) -> None:
        gmsh.merge(filename)
        gmsh.model.mesh.createTopology()

        surfaces = [tag for dim, tag in gmsh.model.getEntities(dim=2)]
        loop_tag = gmsh.model.geo.addSurfaceLoop(surfaces)
        volume_tag = gmsh.model.geo.addVolume([loop_tag])

        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3) 
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
    
    def _check_warn(self) -> bool:
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        if len(element_types) > 0:
            return False
        return True

class SplitHull(GmshObject):
    """
    Split hull along a plane defined by ``normal.dot(x) = waterline``.

    Parameters
    ----------
    hull : Hull
        Base hull to split.
    normal : array_like, shape (3,)
        Normal vector to plane. Will be normalized.
    waterline : float
        Constant term in plane equation.
    rho : float
        Density of water in appropriate units. Default is 1., in which case
        displacements and moments are essentially in units of rho.
    
    Attributes
    ----------
    hull
    normal
    waterline
    volume
    center_of_buoyancy
    x
    y
    """
    def __init__(
            self, 
            hull: Hull,
            normal: ArrayLike, 
            waterline: float,
            rho: float = 1.) -> None:
        super().__init__()
        self._hull = hull
        self._normal = _utils.normalized(np.asarray(normal))
        self._waterline = waterline
        self._split_view = _views.Split(self.hull._volume_field, normal, -1.*waterline)
        self.rho = rho
        self._x = None
        self._y = None


    @property
    def hull(self) -> Hull:
        """
        Base hull to split.

        Returns
        -------
        hull : Hull
            Base hull to split.
        """
        return self._hull
    
    @property
    def normal(self) -> _utils.vec3D:
        """
        Normal vector to plane. Is normalized.

        Returns
        -------
        normal : ndarray, shape (3,)
            Normal vector to plane. Is normalized.
        """
        return self._normal
    
    @property
    def waterline(self) -> float:
        """
        Constant term in plane equation.

        Returns
        -------
        waterline : float
            Constant term in plane equation.
        """
        return self._waterline

    @property
    def volume(self) -> float:
        """
        Volume of region below waterplane.

        Returns
        -------
        volume : float
            Volume below waterplane.
        """
        return self._split_view.volume
    
    @property
    def displacement(self) -> float:
        """
        Displacement of region below waterplane. If ``rho = 1.0``, same as
        ``volume``.

        Returns 
        -------
        displacement : float
            displacement of region below waterplane.
        """
        return self.rho * self.volume

    @property
    def center_of_buoyancy(self) -> _utils.vec3D:
        """
        Center of buoyancy (center of volume) of region below waterplane.

        Returns
        -------
        cob : float
            Center of buoyancy below waterplane.
        """
        return self._split_view.center_of_buoyancy

    def righting_moment(self) -> _utils.vec3D:
        """
        Righting moment of region below waterline.

        Returns
        -------
        moment : ndarray, shape (3,)
            Righting moment of region below waterline.
        """
        r = self.center_of_buoyancy - self.hull.center_of_mass
        buoyancy = self.displacement * self.normal
        moment = np.cross(r, buoyancy)
        return moment
    
    @cached_property
    def _default_x(self) -> _utils.vec3D:
        x = np.array((1.,0,0))
        return _utils.normalized(x - np.dot(self.normal,x)*self.normal)
    
    @property
    def x(self) -> _utils.vec3D:
        """
        Local x vector in waterplane. Default is projection of global x vector
        ``(1,0,0)`` onto the waterplane. Can be set.

        Returns
        -------
        x : ndarray, shape (3,)
            Local x vector in waterplane.
        """
        return self._default_x if self._x is None else self._x

    @x.setter
    def x(self, value) -> None:
        self._x = value

    @cached_property
    def _default_y(self) -> _utils.vec3D:
        return _utils.normalized(-np.cross(np.array((1.,0,0)),self.normal))

    @property
    def y(self) -> _utils.vec3D:
        """
        Local y vector in waterplane. Default is vector perpendicular to both
        ``x`` and ``normal``. Can be set.

        Returns
        -------
        y : ndarray, shape (3,)
            Local y vector in waterplane.
        """
        return self._default_y if self._y is None else self._y

    @y.setter
    def y(self, value) -> None:
        self._y = value
    
    def plot(self) -> None:
        """
        Plot 3D mesh and submerged view in Gmsh.
        """
        self._split_view.show_only()
        gmsh.view.option.setNumber(self._split_view.tag, "ShowScale", 0)
        gmsh.fltk.run()
    


class SolvedHull(SplitHull):
    """
    Result object for ``Hull.solve`` representing region of hull below 
    waterplane. In addition to methods from ``SplitHull``, also includes
    information about termination of ``Hull.solve``. See ``SplitHull`` for
    more details.

    Parameters
    ----------
    hull : Hull
        Base hull to split.
    normal : array_like, shape (3,)
        Normal vector to plane. Will be normalized.
    waterline : float
        Constant term in plane equation.
    rho : float
        Density of water in appropriate units. Default is 1., in which case
        displacements and moments are essentially in units of rho.
    success : bool
        Whether the solver terminated successfully
    message : str
        Details of solver termination.
    
    Attributes
    ----------
    hull
    normal
    waterline
    volume
    center_of_buoyancy
    message
    success
    """
    def __init__(
            self, 
            hull: Hull,
            normal: ArrayLike,
            waterline: float,
            rho: float,
            success: bool,
            message: str):
        super().__init__(hull, normal, waterline, rho=rho)
        self._success = success
        self._message = message

    @property
    def success(self) -> bool:
        """
        Whether the solver terminated successfully.
        
        Returns
        -------
        success : bool
            Whether the solver terminated successfully.
        """
        return self._success
    
    @property
    def message(self):
        """
        Details of solver termination.

        Returns
        -------
        message : str
            Details of solver termination.
        """
        return self._message
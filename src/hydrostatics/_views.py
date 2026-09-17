from collections.abc import Callable
from weakref import finalize

import gmsh
import numpy as np
from numpy.typing import NDArray, ArrayLike

from . import _utils

def hide_all():
    for view_tag in gmsh.view.getTags():
        gmsh.view.option.setNumber(view_tag, "Visible", 0)

class GmshView:
    """
    Wrapper to Gmsh view
    
    Parameters
    ----------
    tag : int
        Unique identifier for view in Gmsh
    """
    def __init__(self, tag: int) -> None:
        self._tag = tag
        finalize(self, GmshView._remove, tag)
    
    @property
    def tag(self) -> int:
        """
        Get Gmsh view tag.

        Returns
        -------
        tag : int
            Gmsh view tag.
        """
        return self._tag

    @property
    def index(self) -> int:
        """
        Get index of Gmsh view tag in Gmsh's internal list of views.

        Returns
        -------
        index : int
            Index Gmsh view tag.
        """
        return gmsh.view.getIndex(self.tag)

    @staticmethod
    def _remove(tag: int) -> None:
        gmsh.view.remove(tag)
    
    def remove(self) -> None:
        """
        Remove view from Gmsh
        """
        self._remove(self.tag)

    def show_only(self) -> None:
        """
        Set all views invisible except this one.
        """
        hide_all()
        gmsh.view.option.setNumber(self.tag, "Visible", 1)        

class Field(GmshView):
    """
    Put a value of function at each point in model mesh

    Parameters
    ----------
    fun : callable
        Function of coordinates that defines the field value at each node. 
        ``fun`` must have signature ``fun(coords)``, where ``coords`` is an
        ndarray. If ``vectorized = False``, ``coords`` is shape ``(3,)``. If
        ``vectorized = True``, ``coords`` has shape ``(n, 3)`` for some 
        value of ``n``.
    vectorized : bool
        If ``vectorized`` is True, ``fun`` is assumed to be implemented in
        vectorized fashion: i.e. it can be called with signature 
        ``fun(coords)``, where ``coords`` has shape ``(n, 3)`` for some 
        value of ``n``. Default is False.
    """
    def __init__(
            self, 
            fun: Callable[[NDArray], NDArray], 
            vectorized: bool = False) -> None:
        self._fun = fun
        super().__init__(gmsh.view.add("Mesh view"))

        node_tags, coords, parametricCoord = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1,3)
        
        field_values = np.empty((0,3))
        if vectorized:
            field_values = self._fun(coords)
        else:
            field_values = [self._fun(c) for c in coords]
        
        gmsh.view.addModelData(
            tag=self.tag,
            step=0,
            modelName=gmsh.model.getCurrent(),
            dataType="NodeData",
            tags=node_tags.tolist(),
            data=field_values
        )
        
class Integral(GmshView):
    """
    Wrapper class for Gmsh's integrate plugin

    Parameters
    ----------
    field : Field
        Field to integrate over mesh.
    dim : int
        Dimension of integral (i.e. 2 for surface, 3 for volume)
    remove : bool
        If `True`, automatically remove from Gmsh after integrating

    Attributes
    ----------
    field
    dim
    result
    """
    def __init__(self, field: Field, dim: int, remove=True):
        self._field = field
        self._dim = dim
        self._result = None
        self.field.show_only() 
        gmsh.plugin.set_number("Integrate", "View", self.field.index)
        gmsh.plugin.set_number("Integrate", "Dimension", dim)
        tag = gmsh.plugin.run("Integrate")
        super().__init__(self, tag)
        self._parse_data()
        if remove: self.remove()
    
    @property
    def field(self):
        """
        Field to integrate over mesh.
        
        Returns
        -------
        field : Field
            Field to integrate over mesh.
        """
        return self._field

    @property
    def dim(self):
        """
        Dimension of integral
        
        Returns
        -------
        dim : int
            Dimension of integral
        """
        return self._dim
    
    @property
    def result(self) -> float | NDArray:
        """
        Result of integral
        
        Returns
        -------
        result : float | NDArray
            Result of integral
        """
        return self._result
    
    def _parse_data(self):
        data_type, num_elements, integrals = gmsh.view.getListData(self.tag)
        self._result = integrals[0][-1]
        
class Split(GmshView):
    """
    Split a mesh with field along a plane and keep the region below. Cutting
    plane is defined by the equation ``normal.dot(x) = d``, where ``normal``
    is a vector normal to the plane and ``d`` is a scalar.

    Parameters
    ----------
    field : Field
        Field to split.
    normal : array_like, shape (3,)
        Normal vector to cutting plane. Will be normalized.
    d : float
        Constant in plane equation.

    Attributes
    ----------
    field
    dim
    result
    """
    def __init__(self, field: Field, normal: ArrayLike, d: float) -> None:
        self._field = field
        self._normal = _utils.normalized(np.asarray(normal))
        self._d = d
        super().__init__(self._cut())
        self._set_cob_volume()

    @property
    def field(self) -> Field:
        """
        Field to split.
        
        Returns
        -------
        field : Field
            Field to split.
        """
        return self._field
    
    @property
    def normal(self) -> NDArray:
        """
        Normal vector to cutting plane. Is normalized.
        
        Returns
        -------
        normal : array_like, shape (3,)
            Normal vector to cutting plane.
        """
        return self._normal

    @property
    def d(self) -> float:
        """
        Constant term in plane equation.
        
        Returns
        -------
        d : float
            Constant term in plane equation.
        """
        return self._d
    
    @property
    def volume(self):
        return self._volume
    
    @property
    def center_of_buoyancy(self):
        return self._center_of_buoyancy

    def _cut(self) -> int:
        gmsh.plugin.setNumber("CutPlane", "View", self.field.index)
        a, b, c = self.normal
        gmsh.plugin.setNumber("CutPlane", "A", a)
        gmsh.plugin.setNumber("CutPlane", "B", b)
        gmsh.plugin.setNumber("CutPlane", "C", c)
        gmsh.plugin.setNumber("CutPlane", "D", self.d)
        gmsh.plugin.setNumber("CutPlane", "ExtractVolume", -1)
        tag = gmsh.plugin.run("CutPlane")
        return tag

    @staticmethod
    def _tet_coords(data_type, tet_data):
        if data_type == 'SS':
            return tet_data.reshape(-1,16)[:,:12].reshape(-1,3,4)
        if data_type == 'SI':
            prism_coords = tet_data.reshape(-1,24)[:,:18].reshape(-1,3,6)
            tet_inds = ((0,1,2,4),(0,2,3,4),(2,3,4,5))
            return np.moveaxis(prism_coords[...,tet_inds],-2,-3)
        
    @classmethod
    def _tet_volumes(cls, coords: NDArray) -> NDArray:
        det_array = coords[...,:-1] - coords[...,-1,None]
        return 1./6. * np.abs(np.linalg.det(det_array))
    
    def _set_cob_volume(self, tol=1.e-13) -> None:
        dataTypes, numElements, data = gmsh.view.getListData(self.tag)
        cob = 0.0
        volume = 0.0
        for dT, dat in zip(dataTypes, data):
            if dT in ('SS', 'SI'):
                tet_coords = self._tet_coords(dT, dat)
                tet_volumes = self._tet_volumes(tet_coords)
                volume += np.sum(tet_volumes)
                tet_centroids = np.mean(tet_coords, axis=-1)
                cob += np.sum(
                    tet_volumes[...,None]*tet_centroids, 
                    axis=tuple(range(0,tet_volumes.ndim)))
        if volume <= tol:
            self._center_of_buoyancy = np.zeros(3)
        else:
            self._center_of_buoyancy = cob/volume
        self._volume = volume
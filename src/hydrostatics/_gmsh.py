import atexit

import gmsh
import numpy as np

from ._utils import vec3D

class GmshObject:
    _initialized = False
    _finalized = False
    
    def __init__(self):
        self._check_initialize()
    
    def _check_initialize(self) -> None:
        if not GmshObject._initialized:
            gmsh.initialize()
            GmshObject._initialized = True
            atexit.register(self.finalize)
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)

    @staticmethod
    def finalize() -> None:
        gmsh.finalize()
        print("Gmsh exited safely.")
    
class GmshModel(GmshObject):
    _all_model_names = set()

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name
        self._check_add_model_name()

    @property
    def model_name(self):
        """
        Name of model created in Gmsh.

        Returns
        -------
        model_name : str
            Name of Gmsh model.
        """
        return self._model_name

    def _bbox(self) -> (vec3D, vec3D):
        self._check_load()
        bounds = gmsh.model.getBoundingBox(-1, -1)
        lower, upper = (bounds[:-3], bounds[3:])
        return (np.asarray(lower), np.asarray(upper))
    
    def _check_add_model_name(self) -> None:
        if self.model_name in self._all_model_names:
            raise ValueError("model_name supplied to GmshObject must be unique.")
        else:
            gmsh.model.add(self.model_name)
            self.__class__._all_model_names.add(self._model_name)
    
    def _check_load(self) -> None:
        if not (gmsh.model.getCurrent() == self.model_name):
            gmsh.model.setCurrent(self.model_name)

    
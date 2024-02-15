# ================================================================================
#
# Pipeline Abstraction
#
# ================================================================================
import logging
from abc import ABC, abstractmethod
from typing import Any, Union


# ========================================
# Pipeline State
# ========================================
class PipelineState(dict):
    """
    A Map like object responsible for holding
    the state of a pipeline.

    Variables can be accessed and added by either
    dictionary like access: state["key"] or
    dot notation: state.key

    Loose extensions of: https://stackoverflow.com/a/32107024

    Attributes
    ----------
    name: str
        name of the state object, passed by kwargs
        default: self.__class__ value
    """

    def __init__(self, *args, **kwargs):
        super(PipelineState, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

        self.name = kwargs["name"] if "name" in kwargs else self.__class__

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(PipelineState, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(PipelineState, self).__delitem__(key)
        del self.__dict__[key]

    def __str__(self):
        return f"[{self.name}] size: {len(self.__dict__)}"


# ========================================
# Pipeline Stage
# ========================================
class PipelineStage(ABC):
    """
    Abstract Base Class for a Pipeline Stage
    """

    def __init__(self, num_workers: int = 1) -> None:
        super().__init__()
        self.num_workers = num_workers

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the `name` of this Pipeline Stage"""
        pass

    @abstractmethod
    def run(self) -> Any:
        """
        Run this Pipeline Stage
        """
        pass

    @abstractmethod
    def runnable(self) -> bool:
        """
        Check if this stage of the pipeline
        is runnable or not
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Check if this stage is already done
        """
        pass

    def __str__(self):
        return f"[{self.name}]"


class StatefulPipelineStage(PipelineStage):
    """
    Abstract Class for a PipelineStage taking in a PipelineState
    """

    def __init__(self, state: PipelineState, num_workers: int = 1) -> None:
        super().__init__(num_workers=num_workers)
        self.state = state


# ========================================
# Pipeline
# ========================================
class PipelineBase(ABC):
    """
    Abstract Base Class for a Preprocessing Pipeline
    """

    def __init__(
        self, stages: list[PipelineStage], num_workers: int = 1, state_name: str = None
    ) -> None:
        super().__init__()

        # Set Pipeline Stages
        self.stages = stages

        # Add State
        name_tmp = state_name if state_name is not None else f"State:{self.__class__.__name__}"
        self.state = PipelineState(name=name_tmp)

        # Parallel Workers
        logging.info(f"[{self.__class__.__name__}] run with {num_workers} workers")
        self.num_workers = num_workers

    def add_stages(self, stages: Union[PipelineStage, list[PipelineStage]]):
        """
        Add a single or a list of stages

        Parameter
        ---------
        stages: Union[PipelineStage, list[PipelineStage]]
            stages to add to this pipeline (appended)
        """
        if isinstance(stages, PipelineStage):
            self.stages.append(stages)
        else:
            assert all(map(lambda x: isinstance(x, PipelineStage), stages))
            self.stages.extend(stages)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the `name` of this Pipeline"""
        pass

    def run(self) -> Any:
        """
        Run all `self.stages` of this pipelien
        """
        logging.info(f"[{self.__class__.__name__}] running pipeline: {self}")
        for stage in self.stages:
            assert stage.runnable(), f"{stage} not runnable"

            if not stage.is_done():
                logging.info(f"[{self.__class__.__name__}] running stage `{stage.name}`")
                stage.run()
                logging.info(f"[{self.__class__.__name__}] completed stage `{stage.name}`")

            else:
                logging.info(
                    f"[{self.__class__.__name__}] stage `{stage.name}` already done, skipped"
                )

    def __str__(self):
        return f"[{self.name}: {len(self.stages)} stages]"


class GenericPipeline(PipelineBase):
    """
    A generic Pipeline
    """

    name = "Generic Pipeline"

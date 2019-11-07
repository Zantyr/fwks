Model building
==============

Approach of fwks to model building is to connect already established components, that might be well-developped
and tested feature transforms, methods of normalization or neural network components. User supplied code
following some basic guidelines should compose seamlessly with the rest of the framework.

Modules
-------

Two main modules are `model` and `stage`. Model define archetypical structure and tasks with which the models should
be built. Stages contain computations that shall constitute the final model: transforms and layers.

.. automodel: fwks.model
	:members:

.. automodel: fwks.stage
	:members:

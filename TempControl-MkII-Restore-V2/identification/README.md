This directory is entirely devoted to system identification/profiling.

 - ``profiling_data``: This is the subdirectory where profiling data will be stored/organized for various systems.

The workflow here is to first run ``profile_generator.py``, a helper script to generate profiling configurations. Then, once you've completed that, run ``profiler.py`` to actually run the system and collect data. Finally, use the ``identifier.py`` script to generate the system coefficients, which will be saved in the ``identified_system`` directory at the root.

Please be aware that the constructor.py file is somewhat retired. It has not been fully migrated to new data saving schemes, nor has it been tested. It it somewhat useless in the face of the Fourier system, so just... Don't use it, okay?

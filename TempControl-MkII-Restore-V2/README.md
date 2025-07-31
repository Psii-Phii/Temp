This is a renewed Git repo for Dr. Dmitry Pushin's temperature control project, mainly developed by Davis Garrad, both of the University of Waterloo.

To start, the ``networking`` and ``docs`` folders are a good place.

A breakdown of the file structure of this repository is as follows:
 - ``control``: Actual engine behind the controller
 - ``data``: Various data (files not included in the Git, to save space and it's not strictly necessarty data)
 - ``gui``: Files to do with the GUI (user interface)
 - ``identification``: Engine for system identification, both Fourier-based (in the ``fourier`` subdirectory) and naive (``naive`` sub-directory)
 - ``monitoring``: Tooling to extract on-the-fly information from stdout and display it nicely. This is for debugging, as the GUi should achieve mostly the same results
 - ``networking``: Mostly tooling to make networking between SBCs and microcontrollers simpler. This is the home of the command system to use the hardware.
 - ``packages``: This is where most of the files for setup are found. Packages to upload to the Raspberry Pi Pico SBC's, PyBoards (deprecated), and SBCs are to be found. This is a good starting point.
 - ``prometheus``: Configuration files and scripts to use Prometheus and Grafana for network-based monitoring.
 - ``simulations``: Mathematica notebooks, Python scripts, etc. to simulate and test profilings.
 - ``identified_system``: Only the ssparams\_\*.out items should live in this folder. This is what the controller reads from to construct the system matrix.

NOTE: All scripts should be designed to run from the parent (AKA: this) directory.

# topology-control
MMIT SGI 2025 topology control project

All of this works at least with python 3.10:

	conda create --name topologycontrol python=3.10

and then activate with 

	conda activate topologycontrol

Some requirement may be missing (pip freeze gave me many local packages that I removed when I generated the file), but it also looks pretty complete just the same.  Probably missing triangle.

	python -m pip install -r requirements.txt

Wishlist here... to figure out what goes in the requjirements with a pip -freeze (i.e., wiht version numbers)

polyscope
meshio
numpy
pytorch <- but install from the web??


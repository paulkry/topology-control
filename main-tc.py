import polyscope as ps
import numpy as np
import meshio as meshio

# init polysope
ps.init()
# load the bunny mesh
mesh = meshio.read("data/bunny.obj")
# add the mesh to polyscope

ps_mesh = ps.register_surface_mesh("bunny", mesh.points, mesh.cells[0].data)

ps.show()
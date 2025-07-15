import igl
import polyscope as ps
import numpy as np
import meshio as meshio

# import igl for computing signed distances

ps.init()

meshes = ["data/bunny.obj", "data/bimba.obj", "data/torus.obj"]

for mesh_file in meshes:
	# load the mesh
	mesh = meshio.read(mesh_file)
	
	V = mesh.points
	F = mesh.cells[0].data

	# rescale vertices to fit in the -1 to 1 cube
	V -= np.mean(V, axis=0)	
	V /= np.max(np.max(np.abs(V),axis=0))

	name = mesh_file.split('/')[-1].split('.')[0]

	# add the mesh to polyscope
	ps_mesh = ps.register_surface_mesh(name, V, F)
	
	# compute 1000 random points in the -1 to 1 cube
	random_points = np.random.uniform(-1, 1, (1000, 3))

	# compute the signed distances at these points
	d = igl.blue_noise( random_points, V, F )[0]

	# make a point cloud of the random points
	point_cloud = ps.register_point_cloud( name + "sample points", random_points)
	# add the signed distance as a scalar quantity to the point cloud
	point_cloud.add_scalar_quantity( "signed_distance", d )

	# sampling points on surface is important for learning.
	# compute a bunch of blue noise points on the surface... something like
	# compute a bunch of gaussian clouds around those surface points to get points inside and outside the mesh
	# ant then perhaps save all these points to an npy file for faster loadding next time!
	
# Next set up a small torch model and learn the 3 shapes with perhaps 2D latent space

ps.show()
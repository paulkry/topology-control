import igl
import polyscope as ps
import numpy as np
import meshio as meshio
import os

# import igl for computing signed distances

ps.init()

meshes = [ "data/raw/bimba.obj", "data/raw/torus.obj"]

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

	# if npy files exist load them!
	if os.path.exists("data/" + name + "_sampled_points.npy"):
		print("Loading existing sampled points and distances for", name)
		sampled_points = np.load("data/" + name + "_sampled_points.npy")
		d = np.load("data/" + name + "_signed_distances.npy")
	else:
		print("Sampling points for", name)

		# compute 1000 random points in the -1 to 1 cube
		N_uniform = 1000
		random_points = np.random.uniform(-1, 1, (N_uniform, 3))

		# sampling points on surface is important for learning.
		# compute a bunch of blue noise points on the surface... something like  
		radius = 0.02
		surface_points = igl.blue_noise(V, F, radius)[2]

		# concatenate surface points with random points
		sampled_points = np.concatenate((random_points, surface_points), axis=0)

		# now compute a gaussian cloud around each of the surface surface
		mu = 0.0
		sigma = radius 
		N = 10
		noise = np.random.normal(mu, sigma, (N, surface_points.shape[0], 3))

		# add the noise to each of thes sampled points		
		for i in range(N):
			new_points = surface_points + noise[i]
			# concatenate these to the sampled points
			sampled_points = np.concatenate((sampled_points, new_points), axis=0)

		# compute the signed distances at these points
		d = igl.signed_distance( sampled_points, V, F )[0]

		# save the sampled_points and distances c into a npy file
		np.save( "data/" + name + "_sampled_points.npy", sampled_points)
		np.save( "data/" + name + "_signed_distances.npy", d)

	# make a point cloud of the random points
	point_cloud = ps.register_point_cloud( name + "sample points", sampled_points)
	# add the signed distance as a scalar quantity to the point cloud
	point_cloud.add_scalar_quantity( "signed_distance", d )






	# compute a bunch of gaussian clouds around those surface points to get points inside and outside the mesh
	# ant then perhaps save all these points to an npy file for faster loadding next time!
	

# # compute meshgrid for evaluating a function 
# x = np.linspace(-1, 1, 100)
# y = np.linspace(-1, 1, 100)			
# z = np.linspace(-1, 1, 100)			
# X, Y, Z = np.meshgrid(x, y, z)
# # compute the function values at these points
# values = f(np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1))
# # do marching cubes
# verts, faces, normals, values = igl.marching_cubes(values.reshape(X.shape), 0.0)

# # Next set up a small torch model and learn the 3 shapes with perhaps 2D latent space

# def f(q, x):
# 	return np.norm(x, axis=1) - 1 


ps.show()
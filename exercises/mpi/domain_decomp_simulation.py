import numpy as np
import math as m

#seconds per element
t_op = 10**-7
#latency is fixed
t_latency = 10**-3
#transfer is due to bandwidth; seconds per element
t_transfer = 10**-10

N = 2**30
n = np.power(N, 1/3)
w = 2
dims = [1, 2, 3]
pn = 10
ps = np.array([2**(i+1) for i in range(pn)], dtype=np.float64)

t0 = N*t_op
print(f"serial: {t0}")

def get_divs(exp, d):
	divs = np.ones(3)
	for i in range(d):
		divs[i] *= 2**(exp // d)
	for i in range(exp % d):
		divs[i] *= 2
	return divs

print("p\t1d\t2d\t3d")
for p in ps:
	exp = round(m.log(p, 2))
	areas = []
	speedups = []
	
	#1d
	divs = get_divs(exp, 1)
	block_dims1 = np.ceil(n / divs)
	oh1 = 2*(t_latency + t_transfer*w*n*n)
	bw1 = w*n*n
	s1 = 1/(1/p + oh1/t0)

	#2d
	divs = get_divs(exp, 2)
	block_dims2 = np.ceil(n / divs)
	oh2 = 2*(t_latency*2 + t_transfer*w*n*n*(1/divs[0] + 1/divs[1]))
	bw1 = w*n*n*(1/divs[0] + 1/divs[1])
	s2 = 1/(1/p + oh2/t0)

	#3d
	divs = get_divs(exp, 3)
	block_dims3 = np.ceil(n / divs)
	oh3 = 2*(t_latency*2 + t_transfer*w*n*n*(1/(divs[0]*divs[1]) + 1/(divs[1]*divs[2]) + 1/(divs[2]*divs[0])))
	bw1 = w*n*n*(1/(divs[0]*divs[1]) + 1/(divs[1]*divs[2]) + 1/(divs[2]*divs[0]))
	s3 = 1/(1/p + oh3/t0)
	"""
	for d in dims:
		#aka blocks per dim
		divs = np.ones(d) * 2**(exp // d)
		for i in range(exp % d):
			divs[i] *= 2
		print(f"p: {round(p)} d: {d} num of blocks: {divs}")
		
		block_dims = np.ceil(n / divs)
		print(f"p: {round(p)} d: {d} dims of block: {block_dims}")
		
		#comm time is 2* transfer/receive of a single block area
		block_areas = []
		for i in range(d):
			block_areas.append(block_dims[i]*block_dims[(i+1)%d])
		block_areas = np.array(block_areas, dtype=np.float64)
		print(f"p: {round(p)} d: {d} areas of block: {np.sum(block_areas)}")
		
		#a = n*n*2*w*np.sum(divs)
		areas.append(n*n*2*w*np.sum(divs))
		
		t = t0/p + 2*(t_latency*d + t_transfer*2*w*n*n*np.sum(1/divs))
		speedups.append(t0/t)
	"""
	#print(f"{round(p)}\t" + "\t".join([str(round(a)) for a in areas]))
	print(f"{round(p)}\t{block_dims1}\t{block_dims2}\t{block_dims3}")	
	print(f"{round(p)}\t{s1:0.2f}\t{s2:0.2f}\t{s3:0.2f}")	
	
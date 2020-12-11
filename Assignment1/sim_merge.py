import sys

P = int(sys.argv[1])

sent = [False] * P

step = 1
d = 1
while d < P:
	print(f"step {step} d {d}")
	for rank in range(P):
		if sent[rank]:
#print(f"rank {rank} does nothing")
			pass
		elif rank % (2 * d):
			dest = rank - d
			print(f"rank {rank} sends to {dest}")
			sent[rank] = True
		else:
			src = rank + d
			if src >= P:
				if rank == 0:
					sent[rank] = True
					break
				else:
					print(f"rank {rank} idles")
			else:
				print(f"rank {rank} receives from {src}")
	print(f"sent is {sent}")
	print("")

	d *= 2
	step += 1
#if d > P:
#		break



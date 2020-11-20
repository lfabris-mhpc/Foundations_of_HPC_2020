// a simple non blocking implementation taken from https://computing.llnl.gov/tutorials/mpi/
#include "mpi.h"
#include "wait_set.h"

#include <stdio.h>

int main(int argc, char *argv[])  {
	int numtasks, rank, next, prev, buf[2], tag1 = 1, tag2 = 2;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	int periodic = 1;
	MPI_Comm   ring_comm;
	MPI_Cart_create( MPI_COMM_WORLD, 1, &numtasks, &periodic, 1, &ring_comm );
	MPI_Cart_shift( ring_comm, 0, 1, &prev, &next );
	MPI_Comm_rank( ring_comm, &rank );
	MPI_Comm_size( ring_comm, &numtasks );

	wait_set* set = new_wait_set(4);
	// post non-blocking receives and sends for neighbors
	size_t slot = enable_least_inactive_slot(set);
	MPI_Irecv(&buf[0], 1, MPI_INT, prev, tag1, MPI_COMM_WORLD, set->requests + slot);
	slot = enable_least_inactive_slot(set);
	MPI_Irecv(&buf[1], 1, MPI_INT, next, tag2, MPI_COMM_WORLD, set->requests + slot);

	slot = enable_least_inactive_slot(set);
	MPI_Isend(&rank, 1, MPI_INT, prev, tag2, MPI_COMM_WORLD, set->requests + slot);
	slot = enable_least_inactive_slot(set);
	MPI_Isend(&rank, 1, MPI_INT, next, tag1, MPI_COMM_WORLD, set->requests + slot);

	// do some work while sends/receives progress in background
	printf("process %d is waiting\n", rank);
	// wait for all non-blocking operations to complete
	//MPI_Waitall(4, reqs, stats);
	//wait out of order
	for (slot = wait_first(set); slot < 4; slot = wait_first(set)) {
		printf("process %d has finished communication %lu\n", rank, slot);
	}
	
	deactivate_all(set);
	destruct_wait_set(set);
	
	// continue - do more work
	printf("process %d can end\n", rank);

	MPI_Finalize();
}

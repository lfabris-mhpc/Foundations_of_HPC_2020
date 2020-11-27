#ifndef __wait_set_h__
#define __wait_set_h__

#include "mpi.h"

typedef struct {
	size_t capacity;
	int* active;
	MPI_Request* requests;
	int* completed;
	MPI_Status* statuses;
} wait_set;

wait_set* new_wait_set(size_t capacity);

void destruct_wait_set(wait_set* set);

size_t active_slots(wait_set* set);

int deactivate_all(wait_set* set);

int deactivate_slot(wait_set* set, size_t i);

size_t wait_first(wait_set* set);

void wait_all(wait_set* set);

size_t enable_least_inactive_slot(wait_set* set);

#endif

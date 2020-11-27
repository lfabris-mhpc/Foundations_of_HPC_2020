#include <stdlib.h>
#include <stdio.h>

#include "wait_set.h"

wait_set* new_wait_set(size_t capacity) {
	wait_set* set = (wait_set*) malloc(sizeof(wait_set));
	
	set->capacity = capacity;
	set->active = (int*) calloc(sizeof(int), capacity);
	set->requests = (MPI_Request*) malloc(sizeof(MPI_Request) * capacity);
	set->completed = (int*) calloc(sizeof(int), capacity);
	set->statuses = (MPI_Status*) malloc(sizeof(MPI_Status) * capacity);
}

void destruct_wait_set(wait_set* set) {
	if (active_slots(set)) {
		fprintf(stderr, "Cannot destruct set with slots still active\n");
		exit(1);
	}

	free(set->active);
	set->active = NULL;
	free(set->requests);
	set->requests = NULL;
	free(set->completed);
	set->completed = NULL;
	free(set->statuses);
	set->statuses = NULL;
}

size_t active_slots(wait_set* set) {
	size_t ret = 0;
	for (size_t i = 0; i < set->capacity; ++i) {
		ret += set->active[i];
	}
	
	return ret;
}

int deactivate_all(wait_set* set) {
	int ret = 0;
	for (size_t i = 0; i < set->capacity; ++i) {
		ret |= deactivate_slot(set, i);
	}
	
	return ret;
}

int deactivate_slot(wait_set* set, size_t i) {
	if (!set->completed[i]) {
		fprintf(stderr, "Cannot deactivate not completed slot %lu\n", i);
		return 1;
	}
	
	set->active[i] = 0;
	set->completed[i] = 0;
	
	return 0;
}

size_t wait_first(wait_set* set) {
	//error when set is empty?
	
	int actives = 0;
	for (size_t i = 0; i < set->capacity; ++i) {
		if (set->active[i] && !set->completed[i]) {
			++actives;
			MPI_Request_get_status(set->requests[i], set->completed + i, set->statuses + i);
			
			if (set->completed[i]) {
				return i;
			}
		}
	}
	
	while (actives) {
		for (size_t i = 0; i < set->capacity; ++i) {
			if (set->active[i] && !set->completed[i]) {
				++actives;
				MPI_Request_get_status(set->requests[i], set->completed + i, set->statuses + i);

				if (set->completed[i]) {
					return i;
				}
			}
		}
	}
	
	return set->capacity;
}

void wait_all(wait_set* set) {
	//error when set is empty?
	
	size_t size = active_slots(set);
	size_t* tmp2set = NULL;
	MPI_Request* tmpRequests = NULL;
	MPI_Status* tmpStatuses = NULL;
	
	if (size == set->capacity) {
		tmpRequests = set->requests;
		tmpStatuses = set->statuses;
	} else {
		tmp2set = (size_t*) malloc(sizeof(size_t) * size);
		tmpRequests = (MPI_Request*) malloc(sizeof(MPI_Request) * size);
		tmpStatuses = (MPI_Status*) malloc(sizeof(MPI_Status) * size);
	
		size_t map = 0;
		for (size_t i = 0; map < size && i < set->capacity; ++i) {
			if (set->active[i]) {
				tmp2set[map] = i;
				tmpRequests[map] = set->requests[i];
				tmpStatuses[map] = set->statuses[i];
				++map;
			}
		}
	}
	
	MPI_Waitall(size, tmpRequests, tmpStatuses);
	
	if (size != set->capacity) {
		for (size_t j = 0; j < size; ++j) {
			size_t i = tmp2set[j];
			set->requests[i] = tmpRequests[j];
			set->statuses[i] = tmpStatuses[j];
		}
		
		free(tmp2set);
		free(tmpRequests);
		free(tmpStatuses);
	}
	
	for (size_t i = 0; i < set->capacity; ++i) {
		set->completed[i] = set->active[i];
	}
}

size_t enable_least_inactive_slot(wait_set* set) {
	if (active_slots(set) == set->capacity) {
		fprintf(stderr, "Cannot activate new request: wait_set full\n");
		exit(1);
	}
	
	for (size_t i = 0; i < set->capacity; ++i) {
		if (!set->active[i]) {
			set->active[i] = 1;
			set->completed[i] = 0;
			
			return i;
		}
	}
	
	return set->capacity;
}
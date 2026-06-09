#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct QueueItem {
    void *data;
    struct QueueItem *next;
} QueueItem;

typedef struct {
    int size;
    int capacity;
    bool thread_safe;
    bool shutdown;
    QueueItem *head, *tail;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Queue;

Queue *new_queue(int capacity, bool thread_safe);
int enqueue(Queue *queue, void *data);

/* timeout_ms < 0: block indefinitely; 0: non-blocking; >0: timed wait. */
void *dequeue(Queue *queue, long timeout_ms);

void shutdown_queue(Queue *queue);

/* Frees the queue structure only — callers must drain items before calling. */
void free_queue(Queue *queue);

#endif  /* QUEUE_H */

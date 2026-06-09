#include "queue.h"
#include "logger.h"

#include <errno.h>
#include <string.h>
#include <time.h>

extern Logger *logger;

static struct timespec get_timeout_time(int timeout_ms);

Queue *new_queue(int capacity, bool thread_safe) {
    Queue *queue = (Queue *)malloc(sizeof(Queue));
    if (queue == NULL) return NULL;

    queue->size = 0;
    queue->capacity = capacity;
    queue->thread_safe = thread_safe;
    queue->shutdown = false;
    queue->head = NULL;
    queue->tail = NULL;

    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);

    int ret = pthread_mutex_init(&queue->mutex, &mutex_attr);
    pthread_mutexattr_destroy(&mutex_attr);
    if (ret != 0) {
        log_error(logger, "Failed to initialize mutex: %s", strerror(ret));
        free(queue);
        return NULL;
    }

    ret = pthread_cond_init(&queue->cond, NULL);
    if (ret != 0) {
        log_error(logger, "Failed to initialize condition variable: %s", strerror(ret));
        pthread_mutex_destroy(&queue->mutex);
        free(queue);
        return NULL;
    }

    return queue;
}

int enqueue(Queue *queue, void *data) {
    int ret = pthread_mutex_lock(&queue->mutex);
    if (ret != 0) {
        log_error(logger, "Mutex lock failed in enqueue: %s", strerror(ret));
        return 1;
    }

    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return 1;
    }

    if (queue->size >= queue->capacity) {
        /* Drop the oldest item — caller is responsible for freeing data. */
        QueueItem *old_head = queue->head;
        queue->head = old_head->next;
        queue->size--;
        if (queue->head == NULL) queue->tail = NULL;
        log_warning(logger, "Queue full — oldest item dropped");
        free(old_head);
    }

    QueueItem *item = (QueueItem *)malloc(sizeof(QueueItem));
    if (item == NULL) {
        pthread_mutex_unlock(&queue->mutex);
        return 1;
    }

    item->data = data;
    item->next = NULL;

    if (queue->size == 0) {
        queue->head = item;
        queue->tail = item;
    } else {
        queue->tail->next = item;
        queue->tail = item;
    }
    queue->size++;

    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

void *dequeue(Queue *queue, long timeout_ms) {
    int ret = pthread_mutex_lock(&queue->mutex);
    if (ret != 0) {
        log_error(logger, "Mutex lock failed in dequeue: %s", strerror(ret));
        return NULL;
    }

    while (queue->size == 0 && !queue->shutdown) {
        if (timeout_ms < 0) {
            ret = pthread_cond_wait(&queue->cond, &queue->mutex);
            if (ret != 0) {
                pthread_mutex_unlock(&queue->mutex);
                return NULL;
            }
        } else if (timeout_ms == 0) {
            /* Non-blocking: return immediately if empty. */
            pthread_mutex_unlock(&queue->mutex);
            return NULL;
        } else {
            struct timespec ts = get_timeout_time((int)timeout_ms);
            ret = pthread_cond_timedwait(&queue->cond, &queue->mutex, &ts);
            if (ret == ETIMEDOUT) {
                pthread_mutex_unlock(&queue->mutex);
                return NULL;
            } else if (ret != 0) {
                pthread_mutex_unlock(&queue->mutex);
                return NULL;
            }
        }
    }

    if (queue->shutdown && queue->size == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }

    void *data = NULL;
    if (queue->size > 0) {
        QueueItem *item = queue->head;
        queue->head = item->next;
        queue->size--;
        if (queue->head == NULL) queue->tail = NULL;
        data = item->data;
        free(item);
    }

    pthread_mutex_unlock(&queue->mutex);
    return data;
}

void shutdown_queue(Queue *queue) {
    int ret = pthread_mutex_lock(&queue->mutex);
    if (ret != 0) {
        log_error(logger, "Mutex lock failed in shutdown_queue: %s", strerror(ret));
        return;
    }
    queue->shutdown = true;
    pthread_cond_broadcast(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
}

void free_queue(Queue *queue) {
    if (queue == NULL) return;

    shutdown_queue(queue);

    int ret = pthread_mutex_lock(&queue->mutex);
    if (ret != 0) {
        log_warning(logger, "Mutex lock failed in free_queue: %s", strerror(ret));
    }

    /* Free QueueItem wrappers only — data pointers are owned by callers. */
    QueueItem *current = queue->head;
    while (current != NULL) {
        QueueItem *next = current->next;
        free(current);
        current = next;
    }
    queue->head = NULL;
    queue->tail = NULL;
    queue->size = 0;

    pthread_mutex_unlock(&queue->mutex);
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->cond);

    free(queue);
}

static struct timespec get_timeout_time(int timeout_ms) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    while (ts.tv_nsec >= 1000000000L) {
        ts.tv_nsec -= 1000000000L;
        ts.tv_sec++;
    }
    return ts;
}

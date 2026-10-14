// Hardware-free test of the internal queue: overflow must release the dropped
// payload via drop_item instead of leaking it. Links queue.c/logger.c directly.
#include <cstdio>
#include <cstdlib>

extern "C" {
#include "logger.h"
#include "queue.h"
}

Logger* logger = nullptr;  // queue.c expects this global

static int g_dropped = 0;
static void count_drop(void* p) {
    g_dropped++;
    free(p);
}

#define CHECK(cond, msg)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            std::printf("  FAIL: %s\n", msg);               \
            return 1;                                       \
        }                                                   \
    } while (0)

int main() {
    logger = create_logger("queue_test", "queue_test.log", LOG_ERROR, LOG_ERROR);
    CHECK(logger, "create_logger");

    Queue* q = new_queue(2, true);
    CHECK(q, "new_queue");
    q->drop_item = count_drop;

    int* a = (int*)malloc(sizeof(int));
    int* b = (int*)malloc(sizeof(int));
    int* c = (int*)malloc(sizeof(int));
    *a = 1; *b = 2; *c = 3;
    CHECK(enqueue(q, a) == 0 && enqueue(q, b) == 0, "fill queue");
    CHECK(enqueue(q, c) == 0, "enqueue on full queue succeeds");
    CHECK(g_dropped == 1, "oldest payload released through drop_item");
    CHECK(q->size == 2, "size stays at capacity");

    int* first = (int*)dequeue(q, 0);
    CHECK(first == b, "oldest (a) was the one dropped");
    free(first);
    free(dequeue(q, 0));
    CHECK(dequeue(q, 0) == nullptr, "queue empty");

    free_queue(q);
    close_logger(logger);
    std::remove("queue_test.log");
    std::printf("  PASS: queue overflow drops and frees oldest payload\n");
    return 0;
}

TARGET = nn
CC = gcc
DEBUGTARGET = debug
TESTTARGET = test
CFLAGS = -Wall -Wextra -pedantic -std=c99 -I include/

src = $(wildcard src/*.c)
lib = $(wildcard include/*.h)
obj = $(src:.c=.o)


test: clean test_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

debug: clean exe_mode debug_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

nn: clean exe_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

debug_mode:
	$(eval CFLAGS += -D DEBUG)

exe_mode:
	$(eval src = $(filter-out src/test.c, $(src)))
	$(eval obj = $(filter-out src/test.o, $(src)))

test_mode:
	$(eval CFLAGS += -rdynamic)
	$(eval CFLAGS += -Wno-unused-command-line-argument)
	$(eval src = $(filter-out src/nn.c, $(src)))
	$(eval obj = $(filter-out src/nn.o, $(src)))

.PHONY: clean
clean:
	rm -f $(obj) $(TARGET) $(DEBUGTARGET) $(TESTTARGET)

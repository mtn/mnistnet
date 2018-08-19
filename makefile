CC = gcc
CFLAGS = -Wall -Wextra -Wno-gnu-statement-expression -pedantic -std=c99 -I include/

TARGET = mnistnet
DEBUGTARGET = debug
TESTTARGET = test

src = $(wildcard src/*.c)
lib = $(wildcard include/*.h)
obj = $(src:.c=.o)


test: clean debug_mode test_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)
	./test

debug: clean exe_mode debug_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

mnistnet: clean exe_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

debug_mode:
	$(eval CFLAGS += -D DEBUG)

exe_mode:
	$(eval src = $(filter-out src/test.c, $(src)))
	$(eval obj = $(filter-out src/test.o, $(src)))

test_mode:
	$(eval CFLAGS += -rdynamic)
	$(eval CFLAGS += -Wno-unused-command-line-argument)
	$(eval src = $(filter-out src/mnistnet.c, $(src)))
	$(eval obj = $(filter-out src/mnistnet.o, $(obj)))

.PHONY: clean
clean:
	rm -rf $(obj) $(TARGET) $(DEBUGTARGET) $(TESTTARGET)

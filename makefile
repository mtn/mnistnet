CC = gcc
CFLAGS = -Wall -Wextra -Wno-gnu-statement-expression -pedantic -std=c99 -I include/

TARGET = mnistnet
DEBUGTARGET = debug
TESTTARGET = test

src = $(wildcard src/*.c)
lib = $(wildcard include/*.h)
obj = $(src:.c=.o)


rund: debug
	./debug

run: mnistnet
	./mnistnet

mnistnet: clean exe_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

valgrind: valgrind_mode mnistnet
	echo "Start of output"
	# valgrind --leak-check=full --show-leak-kinds=all ./mnistnet
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./mnistnet

test: clean debug_mode test_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)
	./test

debug: clean exe_mode debug_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

debug_mode:
	$(eval CFLAGS += -D DEBUG)
	$(eval CFLAGS += -g)

valgrind_mode:
	$(eval CFLAGS += -g)

exe_mode:
	$(eval src = $(filter-out src/test.c, $(src)))
	$(eval obj = $(filter-out src/test.o, $(src)))
	$(eval CFLAGS += -O3)

test_mode:
	$(eval CFLAGS += -rdynamic)
	$(eval CFLAGS += -Wno-unused-command-line-argument)
	$(eval src = $(filter-out src/mnistnet.c, $(src)))
	$(eval obj = $(filter-out src/mnistnet.o, $(obj)))

.PHONY: clean
clean:
	rm -rf $(obj) $(TARGET) $(DEBUGTARGET) $(TESTTARGET)

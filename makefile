CC = gcc
CFLAGS = -Wall -Wextra -Wno-gnu-statement-expression -pedantic -std=c99 -I include/

TARGET = mnistnet
DEBUGTARGET = debug
TESTTARGET = test

src = $(wildcard src/*.c)
lib = $(wildcard include/*.h)
obj = $(src:.c=.o)

exe_obj = $(filter-out src/test.o, $(obj))
test_obj = $(filter-out src/mnistnet.o, $(obj))


run: mnistnet
	./mnistnet

rund: debug
	./debug

lldb: debug_info mnistnet
	lldb mnistnet

mnistnet: clean $(exe_obj) $(lib)
	$(CC) -o $@ $(exe_obj) $(CFLAGS)

valgrind: clean debug_info mnistnet
	echo "Start of output"
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./mnistnet

test: clean debug_mode $(test_obj) $(lib)
	$(CC) -o $@ $(test_obj) $(CFLAGS)
	./test

debug: clean debug_mode $(exe_obj) $(lib)
	$(CC) -o $@ $(exe_obj) $(CFLAGS)

debug_mode:
	$(eval CFLAGS += -D DEBUG)
	$(eval CFLAGS += -g)

debug_info:
	$(eval CFLAGS += -g)
	$(eval CFLAGS = $(filter-out -O3, $(CFLAGS)))

test_mode:
	$(eval CFLAGS += -rdynamic)
	$(eval CFLAGS += -Wno-unused-command-line-argument)

.PHONY: clean
clean:
	rm -rf $(obj) $(TARGET) $(DEBUGTARGET) $(TESTTARGET) *.dSYM

TARGET = nn
CC = gcc
DEBUGTARGET = debug
CFLAGS = -Wall -Wextra -pedantic -std=c99

src = $(wildcard src/*.c)
lib = $(wildcard src/lib/*.h)
obj = $(src:.c=.o)

debug: clean debug_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

nn: clean $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

debug_mode:
	$(eval CFLAGS += -D DEBUG)

.PHONY: clean
clean:
	rm -f $(obj) $(TARGET) $(DEBUGTARGET)

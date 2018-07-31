TARGET = nn
CC = gcc
DEBUGTARGET = debug
CFLAGS = -Wall -Wextra -pedantic -std=c99 -I include/

src = $(wildcard src/*.c)
lib = $(wildcard include/*.h)
obj = $(src:.c=.o)

run: debug
	./debug

debug: clean debug_mode $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

nn: clean $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

debug_mode:
	$(eval CFLAGS += -D DEBUG)

.PHONY: clean
clean:
	rm -f $(obj) $(TARGET) $(DEBUGTARGET)

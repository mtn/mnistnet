TARGET = nn
CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99

src = $(wildcard src/*.c)
lib = $(wildcard src/lib/*.h)
obj = $(src:.c=.o)

nn: $(obj) $(lib)
	$(CC) -o $@ $(obj) $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) $(TARGET)

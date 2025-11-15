# Makefile — OpenMP build (GCC/Clang)
CC      = gcc
CFLAGS  = -O3 -march=native -Wall -std=c11 -fopenmp
LDFLAGS = -lm -fopenmp

SRC = main.c model.c utils.c
OBJ = $(SRC:.c=.o)
BIN = mlp

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)

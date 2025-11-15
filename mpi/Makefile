# Makefile — hybrid MPI + OpenMP build
# Usage: make clean && make

MPICC ?= mpicc
CFLAGS = -O3 -march=native -Wall -std=c11 -fopenmp
LDFLAGS = -lm -fopenmp
DEFS = -DUSE_MPI

SRC = main.c model.c utils.c
OBJ = $(SRC:.c=.o)
BIN = mlp

all: $(BIN)

$(BIN): $(OBJ)
	$(MPICC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(MPICC) $(CFLAGS) $(DEFS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)

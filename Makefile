.PHONY: clean, mrproper

dir_guard=@mkdir -p $(@D)
ODIR=obj
SRCDIR=src
BDIR=bin
CC=mpicc.mpich
CFLAGS= -O3 -DNDEBUG -Wall -DTAKE_TIMES -DPRINT_STAT
LFLAGS= -lm
EXECS=$(BDIR)/dsgdpar
PARSOURCES=lData.c ucrows.c sschedule.c util.c comm.c sgd.c io.c pardsgd.c
PAROBJS=$(patsubst %.c, $(ODIR)/%.o, $(PARSOURCES))
SEQOBJS=$(patsubst %.c, $(ODIR)/%.o, $(SEQSOURCES))
all: par 

par: $(EXECS) 

$(ODIR)/%.o: $(SRCDIR)/%.c
	$(dir_guard)
	$(CC) $(CFLAGS) -c -o $@ $< 

$(BDIR)/dsgdpar: $(PAROBJS)
	$(dir_guard)
	$(CC) $(CFLAGS) -o $@ $+ $(LFLAGS)

clean:
	rm -f $(ODIR)/*.o core.*

mrproper: clean
	rm -f $(BDIR)/dsgdpar

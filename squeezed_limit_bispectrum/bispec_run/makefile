target: FBE_3d_mom  FBE_3d_mom_omp FBE_3d_mom_omp_rec  clean

FFTW_LIB = /usr/local/lib/  # change the library path according to your installation
GSL_LIB = /usr/local/lib/
FFTW_INC = /usr/local/include/  # change the header folder path according to your installation
GSL_INC = /usr/local/include/

CC=gcc
CFLAGS= -g -O2 

FBE_3d_mom:  FBE_3d_mom.c FBE_3d_mom_funcs.c 
	$(CC) $(CFLAGS) -I$(FFTW_INC) -I$(GSL_INC) FBE_3d_mom.c FBE_3d_mom_funcs.c -fopenmp -lm -L$(FFTW_LIB) -lfftw3f -lfftw3f_omp -L$(GSL_LIB) -lgsl -lgslcblas -o FBE_3d_mom
	rm -rf *~ *.o

FBE_3d_mom_omp:  FBE_3d_mom_omp.c FBE_3d_mom_funcs.c 
	$(CC) $(CFLAGS) -I$(FFTW_INC) -I$(GSL_INC) FBE_3d_mom_omp.c FBE_3d_mom_funcs.c -fopenmp -lm -L$(FFTW_LIB) -lfftw3f -lfftw3f_omp -L$(GSL_LIB) -lgsl -lgslcblas -o FBE_3d_mom_omp
	rm -rf *~ *.o

FBE_3d_mom_omp_rec:  FBE_3d_mom_omp_rec.c FBE_3d_mom_funcs_rec.c 
	$(CC) $(CFLAGS) -I$(FFTW_INC) -I$(GSL_INC) FBE_3d_mom_omp_rec.c FBE_3d_mom_funcs_rec.c -fopenmp -lm -L$(FFTW_LIB) -lfftw3f -lfftw3f_omp -L$(GSL_LIB) -lgsl -lgslcblas -o FBE_3d_mom_omp_rec
	rm -rf *~ *.o
	
# avg:	average_bispec_mom.c
# 	$(CC) $(CFLAGS) average_bispec_mom.c -lm -o avg

clean:
	rm -rf FBE_3d_mom FBE_3d_mom_omp_rec FBE_3d_mom_omp avg modes psout* bsout*

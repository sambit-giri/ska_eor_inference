/*=========================================================================
       This code supports FBE_3d_mom.c to compute power spectra and 
       bispectra of an input 3D non-Gaussian field. Bispectrum computation 
       uses FFT method. It also computes multipole moments of the bispectrum.
       Authors : Abinash Kumar Shaw (IIT Kharagpur)
                 Prof. Somnath Bharadwaj (IIT Kharagpur) 
===========================================================================*/

#include<stdlib.h>
#include<stdio.h>
#include<fftw3.h>
#include<math.h>
#include<unistd.h>
#include<omp.h>


extern int N1,N2,N3;
extern int Nthreads;

extern float pi;
extern float LL_x, LL_y, LL_z;

extern float Cx,Cy,Cz,vol;
extern float ***ro;
extern float ****II0, ****II2, ****II4, ****DD0, ****DD2, ****DD4;
extern fftwf_plan p_ro, q_ro;

float ***allocate_3d_float(int N1,int N2,int N3)
{
  int ii,jj;
  long asize,index;
  float ***phia, *phi;

  phia=(float ***)calloc(N1,sizeof(float **));

  for(ii=0;ii<N1;++ii)
      phia[ii]=(float **)calloc(N2,sizeof(float *));

  asize = N1*N2;
  asize = asize*N3;

  if(!(phi = (float *) calloc(asize,sizeof(float))))
    {
      printf("error in allocate_3d_float.\n");
      exit(0);
    }

  for(ii=0;ii<N1;++ii)
    for(jj=0;jj<N2;++jj)
      {
	index = N2*N3;
	index = index*ii + N3*jj;
	phia[ii][jj]=phi+ index;
      }
  return(phia);
}

float ****allocate_4d_float(int N1,int N2, int N3, int N4)
{
  long ii,jj,kk;
  long asize,index, index1;
  float ****phia, *phi;

  phia=(float ****)calloc(N1,sizeof(float ***));
  
  for(ii=0;ii<N1;ii++)
    phia[ii]=(float ***) calloc (N2 ,  sizeof(float **));
    
  for(ii=0;ii<N1;ii++)
    for(jj=0;jj<N2;jj++)
       phia[ii][jj]=(float **) calloc (N3 ,  sizeof(float *));
  
  asize = N1*N2;
  asize = asize*N3;
  asize = asize*N4;

  if(!(phi = (float *) calloc(asize,sizeof(float))))
    {
      printf("error in allocate_4d_float.\n");
      exit(0);
    }

  for(ii=0;ii<N1;ii++)
    for(jj=0;jj<N2;jj++)
      for(kk=0;kk<N3;kk++)
       {  
        index1 = N3*N4;
        index = N2*index1;
        index = index*ii + index1*jj + N3*kk;
        phia[ii][jj][kk]=phi+index;
       }
  return(phia);
}
/*----------------------------------------------------------*/
void fft_init(int Nbin, double *kL,double *kH){

  double kk;
  float ***I, ***D;
  int i,j,k,m,index,ia,ja;
  unsigned long int count;
  FILE *pf;

  I=allocate_3d_float(N1,N2,N3+2); 
  D=allocate_3d_float(N1,N2,N3+2);
  
  fftwf_complex *i1,*d1, *comp_ro;
  i1=(fftwf_complex*)&(I[0][0][0]);
  d1=(fftwf_complex*)&(D[0][0][0]);
  
  fftwf_plan p1, pd1;
  p1= fftwf_plan_dft_c2r_3d (N1, N2, N3, (fftwf_complex*)&(I[0][0][0]), &(I[0][0][0]), FFTW_ESTIMATE);
  pd1= fftwf_plan_dft_c2r_3d (N1, N2, N3, (fftwf_complex*)&(D[0][0][0]), &(D[0][0][0]), FFTW_ESTIMATE);

  comp_ro = (fftwf_complex *)&(ro[0][0][0]);
 
  pf=fopen("modes","w"); 
  for(m=0;m<Nbin;++m){
    count=0;
    for(i=0;i<N1/3;++i)
      for(j=0;j<N2;++j)
        for(k=0;k<N3;++k){
  	      ia=(i>N1/2) ? (N1-i) : i ;
  	      ja=(j>N2/2) ? (N2-j) : j ;
          index=i*N2*(N3/2+1) + j*(N3/2+1) + k;
  	      kk=sqrt(1.*(ia*ia+ja*ja+k*k));
  	      if(kk>=kL[m] && kk<kH[m] && kk>0.){
  	        i1[index][0]=1.; i1[index][1]=0.;
  	        d1[index][0]=comp_ro[index][0]; d1[index][1]=comp_ro[index][1];
  	        count++; 
  	      }
        }
    fftwf_execute(p1); 
    fftwf_execute(pd1); 
    
    for(i=0;i<N1;++i)
      for(j=0;j<N2;++j)
        for(k=0;k<N3;++k){
          DD0[m][i][j][k]=D[i][j][k];
          II0[m][i][j][k]=I[i][j][k];
        }
    
    
    for(i=0;i<N1;++i)
      for(j=0;j<N2;++j)
        for(k=0;k<N3+2;++k){
            I[i][j][k]=0.;
            D[i][j][k]=0.;
          }

    fprintf(pf,"%d\t%ld\n",m,count);
  }
 fclose(pf);

 fftwf_destroy_plan(p1);
 fftwf_destroy_plan(pd1);
 free(I);
 free(D);
}
void fft_init_new(int Nbin, double *kL,double *kH){

  double kk;
  float ***I, ***D;
  int i,j,k,m,index,ia,ja;
  unsigned long int count;
  FILE *pf;

  I=allocate_3d_float(N1,N2,N3+2); 
  D=allocate_3d_float(N1,N2,N3+2);
  
  fftwf_complex *i1,*d1, *comp_ro;
  i1=(fftwf_complex*)&(I[0][0][0]);
  d1=(fftwf_complex*)&(D[0][0][0]);
  
  fftwf_plan p1, pd1;
  p1= fftwf_plan_dft_c2r_3d (N1, N2, N3, (fftwf_complex*)&(I[0][0][0]), &(I[0][0][0]), FFTW_ESTIMATE);
  pd1= fftwf_plan_dft_c2r_3d (N1, N2, N3, (fftwf_complex*)&(D[0][0][0]), &(D[0][0][0]), FFTW_ESTIMATE);

  comp_ro = (fftwf_complex *)&(ro[0][0][0]);
 
  pf=fopen("modes","w"); 
  for(m=0;m<Nbin;++m){
    count=0;
    for(i=0;i<N1/3;++i)
      for(j=0;j<N2;++j)
        for(k=0;k<N3;++k){
  	      ia=(i>N1/2) ? (N1-i) : i ;
  	      ja=(j>N2/2) ? (N2-j) : j ;
          index=i*N2*(N3/2+1) + j*(N3/2+1) + k;
  
          kk= sqrt(pow(ia * Cx, 2) + pow(ja * Cy, 2) + pow(k * Cz, 2));

  	      if(kk>=kL[m] && kk<kH[m] && kk>0.){
  
  	        i1[index][0]=1.; i1[index][1]=0.;
  	        d1[index][0]=comp_ro[index][0]; d1[index][1]=comp_ro[index][1];
  	        count++; 
  	      }
        }
    fftwf_execute(p1); 
    fftwf_execute(pd1); 
    
    for(i=0;i<N1;++i)
      for(j=0;j<N2;++j)
        for(k=0;k<N3;++k){
          DD0[m][i][j][k]=D[i][j][k];
          II0[m][i][j][k]=I[i][j][k];
        }
    

    for(i=0;i<N1;++i)
      for(j=0;j<N2;++j)
        for(k=0;k<N3+2;++k){
            I[i][j][k]=0.;
            D[i][j][k]=0.;
          }

    fprintf(pf,"%d\t%ld\n",m,count);
  }
  fclose(pf);

 fftwf_destroy_plan(p1);
 fftwf_destroy_plan(pd1);
 free(I);
 free(D);
}



void Make_Image()
{ 

  int i,j,k,index;
  
 
  fftwf_complex *A;
  A=(fftwf_complex*)&(ro[0][0][0]);
  
  for(i=0;i<N1;i++)
    for(j=0;j<N2;j++)
      for(k=0;k<=N3/2;k++)
	 {
	  index=i*N2*(N3/2+1) + j*(N3/2+1) + k;
	  A[index][0]=A[index][0]/vol;
	  A[index][1]=A[index][1]/vol;
        }
  
  fftwf_execute(q_ro); 
}

void Get_Modes()

{

  int i,j,k;
  double Lcube=LL_x*LL_y*LL_z; 
  
  for(i=0;i<N1;i++)
    for(j=0;j<N2;j++)
      for(k=0;k<N3;k++)
	  ro[i][j][k]=ro[i][j][k]*Lcube;
      
      fftwf_execute(p_ro); 
}


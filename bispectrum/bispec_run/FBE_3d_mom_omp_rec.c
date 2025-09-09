/*=========================================================================
       This code takes a 3D non-Gaussian random field as input and compute
       its power spectra and bispectra. Bispectrum computation uses FFT 
       method. It also computes multipole moments of the bispectrum.
       Authors : Abinash Kumar Shaw (IIT Kharagpur)
                 Prof. Somnath Bharadwaj (IIT Kharagpur) 
===========================================================================*/

#include<stdlib.h>
#include<stdio.h>
#include<fftw3.h>
#include<math.h>
#include<unistd.h>
#include<omp.h>
#include<gsl/gsl_errno.h>
#include<gsl/gsl_spline.h>


int N1,N2,N3;
int Nthreads;
float pi=M_PI;
float LL_x, LL_y, LL_z;
float Cx,Cy,Cz,vol;
float ***ro;
float ****II0, ****II2, ****II4, ****DD0, ****DD2, ****DD4;
fftwf_plan p_ro, q_ro;

int main(int argc, char *argv[])
{
  int Nbin, Nmax, ii, jj, ll, NN1;
  double *kLow,*kHigh,*kavg,dk;
  char  OPT;
  char file[512];
  FILE *fp;
  double t,T=omp_get_wtime();

  void Make_Image();
  void Get_Modes();
  
  void power_spec(int Nbin,double *kmode,double *power,long *no);
  void fft_bispec(double kk1,double kk2,double kk3,double dkk1,double dkk2, double dkk3,double *bispec,double *no);
  void fft_init(int Nbin, double *kL,double *kH);
  void fft_init_new(int Nbin, double *kL,double *kH);
  void fft_multipole(int nn, double *kL, double *kH, int nn1);
  float ***allocate_3d_float(int N1,int N2,int N3);
  float ****allocate_4d_float(int N1,int N2, int N3, int N4);
  

  FILE *fp_input_file;
  fp_input_file = fopen(argv[1], "r");
  fscanf(fp_input_file, "%f", &LL_x);
  fscanf(fp_input_file, "%f", &LL_y);
  fscanf(fp_input_file, "%f", &LL_z);
  fscanf(fp_input_file, "%d", &Nmax);
  fscanf(fp_input_file, "%d", &Nbin);
  fscanf(fp_input_file, "%d", &NN1);
  fscanf(fp_input_file, "%d", &Nthreads);
  fclose(fp_input_file);
  
 fp = fopen(argv[2], "r");
 fread(&N1, sizeof(int), 1, fp);
 fread(&N2, sizeof(int), 1, fp);
 fread(&N3, sizeof(int), 1, fp);
 
 fprintf(stdout,"N1= %d\tN2= %d\tN3= %d\n",N1, N2, N3);
 
 if(Nmax>N1/3 || Nmax>N2/3 || Nmax>N3/3)
  {
   printf("k_max grid is out of the bound (N1/3, N2/3, N3/3).\n");
   return 1; 
  }
 
 

  ro= allocate_3d_float(N1,N2,N3+2);
  
  for(ii=0;ii<N1;ii++)
    for(jj=0;jj<N2;jj++)
      for(ll=0;ll<N3;ll++)
         fread(&ro[ii][jj][ll], sizeof(float), 1, fp);
  
  fclose(fp);
 /*------------------------------------------------------*/

 /*------------------------------------------------------*/
 printf("%f %f %f \n", ro[0][0][0], ro[0][0][1], ro[0][0][2]);
 
  Cx=2.*pi/(N1*LL_x);
  Cy=2.*pi/(N2*LL_y);  
  Cz=2.*pi/(N3*LL_z);
  vol=(N1*LL_x)*(N2*LL_y)*(N3*LL_z);
  
  fftwf_init_threads();
  fftwf_plan_with_nthreads(Nthreads);
  omp_set_num_threads(Nthreads);
  
  p_ro = fftwf_plan_dft_r2c_3d(N1, N2, N3, &(ro[0][0][0]), (fftwf_complex*)&(ro[0][0][0]), FFTW_ESTIMATE);  
  q_ro = fftwf_plan_dft_c2r_3d(N1, N2, N3, (fftwf_complex*)&(ro[0][0][0]), &(ro[0][0][0]), FFTW_ESTIMATE);
 /*----------------------------------------------------------*/
 
 Get_Modes(); 
 
 
  int i,j,l,NB=15;
  long *num;
  double *kmean,*pkmean;

 
  num=(long*)calloc(NB,sizeof(long));
  kmean=(double*)calloc(NB,sizeof(double));
  pkmean=(double*)calloc(NB,sizeof(double));

  
  t=omp_get_wtime();

  t=omp_get_wtime();
  double Bispec, Bispec2, Bispec4, Num, Num2, Num4, tt, mu, Biana, Bierr;
  double k1,k2,k3;

  DD0=allocate_4d_float(Nbin,N1,N2,N3);


  II0=allocate_4d_float(Nbin,N1,N2,N3);

  
  kLow=(double*)calloc(Nbin,sizeof(double));
  kHigh=(double*)calloc(Nbin,sizeof(double));
  kavg=(double*)calloc(Nbin,sizeof(double));
  
  dk=(Nmax-1)/(1.*Nbin); 
  
  double k_min = fmin(fmin(Cx, Cy), Cz); 

  dk = dk * k_min; 
  
  for(i=0;i<Nbin;++i)
  {
 
    kLow[i]=k_min + i*dk;
    kHigh[i]=k_min + (i+1)*dk;
    kavg[i]=(3./4.)*(pow(kHigh[i],4.)-pow(kLow[i],4.))/(pow(kHigh[i],3.)-pow(kLow[i],3.)); 
   fft_init_new(Nbin, kLow, kHigh); 

  for(i=Nbin-NN1;i<Nbin;++i){
    k1 = kavg[i]; 

    snprintf(file, 512 * sizeof(char),"%s/bsout_k%d_%.3lf",argv[3],i,k1);
    fp=fopen(file,"w");
    for(j=0;j<=i;++j){
      k2 = kavg[j];
      tt=kavg[j]/kavg[i];
      if(tt>=0.5 && tt<=1.){
        for(l=0;l<=j;++l){
          k3 = kavg[l];
          mu=0.5*(tt+1./tt-(kavg[l]/kavg[i])*(kavg[l]/kavg[j]));
          if(mu>=0.5/tt && mu<=1.){ 
            Bispec=0.;
  	        Num=0.;
            #pragma omp parallel for collapse(3) private(ii,jj,ll) reduction(+:Bispec,Num) 
  	        for(ii=0;ii<N1;++ii)
    	        for(jj=0;jj<N2;++jj)
      	       for(ll=0;ll<N3;++ll){
         	        Bispec+=DD0[i][ii][jj][ll]*DD0[j][ii][jj][ll]*DD0[l][ii][jj][ll];
         	        Num+=II0[i][ii][jj][ll]*II0[j][ii][jj][ll]*II0[l][ii][jj][ll];
                }
        	
            Bispec/=(vol*Num); 
            Num/=(N1*N2*N3*1.);
            fprintf(fp,"%e\t%e\t%e\t%.0lf\n",tt,mu,Bispec,Num); 
      	  } 
        }
      }
    }
   fclose(fp);
  }

 
  fprintf(stdout,"Bispectra computation time = %e sec\n",omp_get_wtime()-t);

  free(num);
  free(kmean);
  free(pkmean);
  free(II0);
  free(DD0);
  fftwf_destroy_plan(p_ro);
  fftwf_destroy_plan(q_ro);
  
 fprintf(stdout,"Total time taken = %d hr %d min %d sec\n",(int)((omp_get_wtime()-T)/3600), (int)((omp_get_wtime()-T)/60)%60, (int)(omp_get_wtime()-T)%60);
}

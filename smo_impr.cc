/*
* This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/license
s/>.
*
 * smo_impr.cc
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#define EPS 1e-20 // required for takestep
#define ZERO 1e-12 

using namespace std;

const double pone = 1.0, mone = -1.0;
const double TOL = 1e-3;
double numkerneleval = 0.0;
double (*kernelchoice)(int i1, int i2, double **inputvecmat, 
                const double *precompdotprod, const int dim, const double sigmasqr);

double (*testkernelchoice)(const int i1, const int i2, double **inputvecmat, 
                double **testvecmat, const int dim, const double *precompdotprod, 
                const double *precompdotprod_test, const double sigmasqr);


/*
 * kernel caching for only a point with itself, 
 * can expand this to all points
 */
void precomputedp (double **inputvecmat, const int numtrsamp, double 
                *precompdotprod, const int dim)
{
        int i, j;
        double sum  = 0.0;
        for (j  = 0; j < numtrsamp; j++)
        {
                sum = 0.0;
                for(i = 0; i <dim; i++)
                {
                        if ( fabs(inputvecmat[j][i]) < ZERO )
                                continue;
                        sum += inputvecmat[j][i] * inputvecmat[j][i];
                }
                precompdotprod[j] = sum;
        }
        numkerneleval = numkerneleval + (double)j;
}

/*
 * kernel caching for only a test point with itself
 */
void precomputedp_test (double **testvecmat, const int numtestsamp, double *precompdotprod_test,
                const int dim)
{
        int i, j;
        double sum  = 0.0;
        for (j  = 0; j < numtestsamp; j++)
        {
                sum = 0.0;
                for(i = 0; i <dim; i++)
                {
                        if ( fabs(testvecmat[j][i]) < ZERO )
                                continue;
                        sum += testvecmat[j][i] * testvecmat[j][i];
                }
                precompdotprod_test[j] = sum;
        }
}


/*
 * Linear Kernel function between training and test point
 */
double testdp (const int i1, const int i2, double **inputvecmat, 
                double **testvecmat, const int dim, const double *precompdotprod,
                const double *precompdotprod_test, const double sigmasqr )
{
        int i;
        double xy = 0.0;
        for (i = 0; i < dim; i++)
        {
                if ( fabs(inputvecmat[i1][i]) < ZERO ||
                                fabs(testvecmat[i2][i]) < ZERO )
                        continue;
                xy += inputvecmat[i1][i] * testvecmat[i2][i];
        }
        return xy;
}
/*
 * Polynomial test kernel function 
 */
double testpolykernel(const int i1, const int i2, double **inputvecmat, 
                double **testvecmat, const int dim, const double *precompdotprod, 
                const double *precomptdotprod_test, const double sigmasqr)
{
        int i;
        double xy, prod = 1.0;
        xy = testdp(i1, i2, inputvecmat, testvecmat, dim, NULL, NULL, 0 ) + 1.0;
        for (i = 1; i <= (int)sigmasqr; i++)
                prod = prod * xy;
        return prod;

}
/*
 * RBF kernel function between training and test point.
 * Called by testevali
 */
double testRBF (const int i1, const int i2, double **inputvecmat, double **testvecmat,
                const int dim, const double *precompdotprod, 
                const double *precompdotprod_test, const double sigmasqr)
{
        double twosigmasqr, xx = 0.0, yy = 0.0, normsqr;
        twosigmasqr = 2 * sigmasqr;
        double xy = 0.0;
        xx = precompdotprod[i1];
        yy = precompdotprod_test[i2];
        xy = testdp(i1, i2, inputvecmat, testvecmat, dim, NULL, NULL, 0); 
        normsqr = (2 * xy -xx -yy) / twosigmasqr; 
        return( exp(normsqr) );
        return xy;
}

double testevali(int idx, double **inputvecmat, double **testvecmat, 
                const double *alpharr, const double *yarr, const int numtrsamp, 
                const int dim, const double sigmasqr, const double *precompdotprod,
                const double *precompdotprod_test)
{
        double scalarsum = 0.0, scalar;
        int i;
        for ( i = 0; i < numtrsamp; i++)
        {
                if (alpharr[i] < ZERO)
                        continue;
                scalar = alpharr[i] * yarr[i];
                scalar *= testkernelchoice(i, idx, inputvecmat, testvecmat, dim, 
                                precompdotprod, precompdotprod_test,sigmasqr);
                scalarsum += scalar;
        }
        return scalarsum;
}

/* Function to calculate the number of misclassified examples from 
 * test set; requires alpharr, thresh and dotproduct calculation
 * between support vectors and test vectors 
 */
int testsamples(double **inputvecmat, double **testvecmat, 
                const double *alpharr, const double *yarr,
                const int numtrsamp, const int dim, const double sigmasqr, 
                const double thresh, const int numtestsamp, const double *testy,
                const double *precompdotprod, const double *precompdotprod_test)
{
        double testval;
        int mclass = 0, i;
        if (dim == 2)
                fprintf(stdout,"test results\n");
        for ( i = 0; i < numtestsamp; i++)
        {
                testval = testevali(i, inputvecmat, testvecmat,alpharr, yarr,
                                numtrsamp, dim, sigmasqr,precompdotprod,
                                precompdotprod_test) - thresh;
                if (testval  < ZERO)
                {
                        if(dim == 2)
                                fprintf(stdout, "%d\n", -1);
                        if (testy[i] > ZERO) 
                                mclass++;
                } 
                if (testval > ZERO )
                {
                        if(dim == 2)
                                fprintf(stdout, "%d\n", 1);
                        if (testy[i] < ZERO)
                                mclass ++;
                }
        }
        return mclass;
}
/*
 * add the fcache values of tr samples with current alpha 0<alpha<C
 */
double addFcache(const double *alpharr, const double *fcache, const double b_up, 
                const double b_low, const int numtrsamp, const double C)
{
        double sum = 0.0;
        int cnt = 0, i;
        for ( i = 0; i < numtrsamp; i++)
        {
                if ( alpharr[i] > ZERO && fabs(alpharr[i] - C) > ZERO )
                {
                        sum += fcache[i];
                        cnt ++;
                }
        }
        if (cnt != 0)
                sum /= cnt;
        else
                sum += (b_up + b_low) * 0.5;
        return sum;
}

/*
 * finds the dot product of the input sample with index idx with every other 
 * input sample, scales the dot product scalar val with alpha and y value 
 */
double evali(const int idx, double **inputvecmat, const double *alpharr, 
                const double *yarr, const int numtrsamp, const double *precompdotprod,
                const double sigmasqr, const int dim)
{
        /*double RBF (int i1, int i2);
          double dp (int i1, int i2);*/
        double scalarsum = 0.0, scalar;
        int i;
        for ( i = 0; i < numtrsamp; i++)
        {
                if (alpharr[i] < ZERO)
                        continue;
                scalar = alpharr[i] * yarr[i];
                scalar *= kernelchoice(i, idx, inputvecmat, precompdotprod, dim,
                                sigmasqr); 
                scalarsum += scalar;
        }
        return scalarsum;
}
/*
 * from the primal problem constraints- variable for soft margin
 * classification
 */
void setslackarr(double *slackarr, double **inputvecmat, const double *alpharr, 
                const double *yarr, const int numtrsamp, const double *precompdotprod, 
                const double sigmasqr, const double thresh, const int dim)
{
        double arg;
        int i;
        for ( i = 0; i < numtrsamp; i++)
        {
                arg = (evali(i, inputvecmat, alpharr, yarr, numtrsamp,
                                        precompdotprod, sigmasqr, dim) - 
                                thresh) * yarr[i];
                arg = pone - arg;
                slackarr[i] = (arg > ZERO) ? arg : 0.0;
        }
}

/*dense dotproduct - tuned for sparse non binary datasets */
double dp(int i1, int i2, double **inputvecmat, const double *precompdotprod, 
                const int dim, const double sigmasqr)
{
        int i;
        double sum = 0.0;
        if (i1 == i2)
                return precompdotprod[i1];
        for (i = 0; i < dim; i++)
        {
                if ( fabs(inputvecmat[i1][i]) < ZERO ||
                                fabs(inputvecmat[i2][i]) < ZERO )
                        continue;
                sum += inputvecmat[i1][i] * inputvecmat[i2][i];
        }
        numkerneleval = numkerneleval + 1.0;
        return sum;
}
/*
 * Gaussian kernel function, using precomputed dot products
 */
double RBF (int i1, int i2, double **inputvecmat, const double *precompdotprod, 
                const int dim, const double sigmasqr)
{
        double twosigmasqr, xx, xy, yy, normsqr;
        twosigmasqr = 2 * sigmasqr;
        xx = precompdotprod[i1];
        yy = precompdotprod[i2];
        xy = dp (i1, i2, inputvecmat, precompdotprod, dim, 0);
        normsqr = (2 * xy -xx -yy) / twosigmasqr; 
        return( exp(normsqr) );
}
/*
 * Polynomial kernel
 */
double polykernel(int i1, int i2, double **inputvecmat, const double *precompdotprod,
                const int dim, const double sigmasqr)
{
        int i;
        double prod = 1.0, xy ;
        xy = dp(i1, i2, inputvecmat, precompdotprod, dim, 0) + 1.0;
        for (i = 1; i <= (int)sigmasqr; i++)
                prod = prod * xy;
        return prod;
}

/*
 * function to update the fcache of all points in the set I_O based
 * on the current two alpha updates
 */
void updateFI(double *fcache, double delalph1, double delalph2, int i1, int i2, 
                const int numtrsamp, const double *yarr, const int *setinfo, 
                double **inputvecmat, const double *precompdotprod, 
                const int dim, const double sigmasqr )
{
        double withone, withtwo;
        int i;
        for ( i = 0; i < numtrsamp; i++)
        {
                if(setinfo[i] == 0)
                {
                        withone = kernelchoice(i1, i, inputvecmat, precompdotprod, 
                                        dim, sigmasqr);
                        withtwo = kernelchoice(i2, i, inputvecmat, precompdotprod, dim
                                        ,sigmasqr);
                        fcache[i] += yarr[i1] * delalph1 * withone + 
                                yarr[i2] * delalph2 * withtwo;
                }
        }
}

/*
 * initialize the set information based on alpha value and yvalue
 * value is one of the 5 sets: refer paper
 */
void setset(int idx, int *setinfo, const double *alpharr, const double *yarr, 
                const double C)
{
        double alphval, yval;
        alphval = alpharr[idx];
        yval = yarr[idx];
        if(alphval > ZERO && fabs(alphval - C ) > ZERO)
                setinfo[idx] = 0;
        //alpha is zero
        if( alphval < ZERO )
        {
                //points on the correct side of hyperplane
                if ( fabs(yval - pone) < ZERO )
                        setinfo[idx] = 1;//y_i=+1
                else
                        setinfo[idx] = 4;//y_i=-1
        }
        //alpha is C
        if ( fabs(alphval - C) < ZERO)
        {
                //points on the wrong side of the hyperplane
                if ( fabs(yval - pone) < ZERO)
                        setinfo[idx] = 3;//y_i=-1
                else
                        setinfo[idx] = 2;//y_i=+1

        }
}

/*
 * compute the thresholds for each iteration
 */
void computeb_i_up(int i1, int i2, const int numtrsamp, const int *setinfo,
                const double *fcache, double *b_up, int *i_up)
{
        double upmin = 1e12;
        int i, sinfo;
        for ( i = 0; i < numtrsamp; i++)
        {
                sinfo = setinfo[i];
                if (sinfo == 0 || i == i1 || i == i2 )
                {
                        //find the minimum most fcache val
                        if ( fcache[i] < upmin)
                        {
                                upmin = fcache[i];
                                *b_up = upmin;
                                *i_up = i;
                        }
                }
        }
}

void computeb_i_low(int i1, int i2, const int numtrsamp, const int *setinfo,
                const double *fcache, double *b_low, int *i_low)
{
        double lowmax = -1e12;
        int i, sinfo;
        for ( i = 0; i < numtrsamp; i++)
        {
                sinfo = setinfo[i];
                if (sinfo == 0 || i == i1 || i == i2 )
                {
                        //find the max most fcache val
                        if ( fcache[i] > lowmax)
                        {
                                lowmax = fcache[i];
                                *b_low = lowmax;
                                *i_low = i;
                        }
                }
        }
}

/*
 * function takestep to optimize the constraints involving i1 and i2
 */
int takestep(int i1, int i2, double *alpharr, int *setinfo, double *fcache,double *b_low,
                double *b_up, int *i_low, int *i_up, double **inputvecmat, 
                const double *precompdotprod, const double *yarr, const int numtrsamp,
                const double sigmasqr, const int dim, const double C)
{

        double s, alph1, alph2, F1, F2, L, H, k11, k22, k12, y1, y2 ;
        double eta, a2, a1, delalph1, delalph2, compart, Lobj, Hobj;
        double t;
        if ( i1 == i2)
                return 0;
        alph1 = alpharr[i1];
        alph2 = alpharr[i2];
        y1 = yarr[i1];
        y2 = yarr[i2];
        F1 = fcache[i1];
        F2 = fcache[i2];
        s = y1 * y2; 
        //from Platt's paper
        if ( fabs(y1 - y2) > ZERO)
        {
                L = ( (alph2 - alph1) > ZERO )? alph2 - alph1 : 0.0; 
                H = ( (C + alph2 - alph1) < C) ? C + alph2 - alph1 : C ;
        }
        else
        {
                L = ( (alph2 + alph1 - C) > ZERO )? alph2 + alph1 - C : 0.0; 
                H = ( (alph2 + alph1) < C) ? alph2 + alph1 : C ;
        }
        if ( fabs(L - H) < ZERO )
        {
                /*fprintf(stderr, "L:%lf H:%lf\n", L, H);*/
                return 0;
        }

        k11 = kernelchoice(i1, i1, inputvecmat, precompdotprod, dim, sigmasqr); 
        k22 = kernelchoice(i2, i2, inputvecmat, precompdotprod, dim, sigmasqr);
        k12 = kernelchoice(i1, i2, inputvecmat, precompdotprod, dim, sigmasqr);
        eta = 2 * k12 - k11 - k22;
        if ( eta < ZERO)
        {
                a2 = alph2 - y2 * (F1 - F2) / eta;
                /*fprintf(stderr, "%lf %lf %lf %lf %lf\n",
                  a2, alph2, y2, F1-F2, eta);*/
                if (a2 < L)
                {
                        /*fprintf(stderr, "a2 less than L\n");*/
                        a2 = L;
                }
                else if (a2 > H)
                        a2 = H;
        }
        else
        {
                compart = y2 * (F1 - F2) - eta * alph2;
                Lobj = 0.5 * eta * L * L + compart * L;
                Hobj = 0.5 * eta * H * H + compart * H;
                if (Lobj > Hobj + EPS)
                        a2 = L;
                else if (Lobj < Hobj - EPS)
                        a2 = H;
                else
                        a2 = alph2;
        }
        if ( a2 < ZERO)
        {
                /*fprintf(stderr, "a2 less ZERO \n");*/
                a2 = 0.0;
        }
        else if (a2 > (C -ZERO))
                a2 = C;
        delalph2 = a2 - alph2;
        /*fprintf(stderr, "updated:a2 %lf L:%lf H:%lf\n", a2, L, H);*/
        if ( fabs(delalph2) < EPS * (a2 + alph2 + EPS ))
        {
                fprintf(stderr,"fabs quit\n");
                fprintf(stderr, "delal:%lf a2+alph2%lf L:%lf H:%lf ",
                                delalph2,
                                a2 + alph2, L, H);
                return 0;
        }
        a1 = alph1 + s * (alph2 - a2);
        if ( a1 < ZERO)
        {
                a1 = 0.0;
        }
        else if( a1 > (C-ZERO) )
        {
                t = a1 - C;
                a2 += s * t;
                a1 = C;
        }

        delalph1 = a1 - alph1;
        alpharr[i1] = a1;
        alpharr[i2] = a2;
        setset(i1, setinfo, alpharr, yarr, C);
        setset(i2, setinfo, alpharr, yarr, C);
        updateFI(fcache, delalph1, delalph2, i1, i2, numtrsamp, yarr, setinfo,
                        inputvecmat, precompdotprod, dim, sigmasqr);
        computeb_i_low(i1, i2, numtrsamp, setinfo, fcache, b_low, i_low);
        computeb_i_up(i1, i2, numtrsamp, setinfo, fcache, b_up, i_up);
        return 1;
}

/*
 * function examineexample to optimize single sample at a time based on 
 * current b_low and b_up
 */
int examineexample(int i2, double *alpharr, int *setinfo, double *fcache, double *b_low,
                double *b_up, int *i_low, int *i_up, double **inputvecmat, 
                const double*precompdotprod, const double *yarr, const int numtrsamp,
                const double sigmasqr, const int dim, const double C )
{
        double fcalc(int, const double*, const double *, double **, const double*,
                        const int, const double, const int );
        double y2, alph2, f2;
        int i1, sinfo, optimality;
        y2 = yarr[i2];
        alph2 = alpharr[i2];
        sinfo = setinfo[i2];
        if (sinfo == 0)
                f2 = fcache[i2];
        else
        {
                f2 = fcalc(i2, alpharr, yarr, inputvecmat, precompdotprod, 
                                dim, sigmasqr, numtrsamp);
                fcache[i2] = f2;
                if ((sinfo == 1 || sinfo == 2)&&(f2 < *b_up))
                {
                        *b_up = f2;
                        *i_up = i2;
                }
                else if((sinfo == 3 || sinfo == 4)&&(f2 > *b_low))
                {
                        *b_low = f2;
                        *i_low = i2;
                }

        }//end else
        optimality = 1;
        if (setinfo == 0 || sinfo == 1 || sinfo == 2)
        {
                if((*b_low - f2) > 2*TOL)
                {
                        optimality = 0;
                        i1 = *i_low;
                }

        }
        if (setinfo == 0 || sinfo == 3 || sinfo == 4)
        {
                if ((f2 - *b_up) > 2*TOL)
                {
                        optimality = 0;
                        i1 = *i_up;
                }
        }
        if (optimality  == 1)
                return 0;
        /*
         * fine tune the second index selection
         */
        if (sinfo == 0)
        {
                if ( (*b_low - f2) > (f2- *b_up))
                        i1 = *i_low;
                else
                        i1 = *i_up;
        }

        if (takestep(i1, i2, alpharr, setinfo, fcache, b_low, b_up, i_low, i_up,
                                inputvecmat, precompdotprod, yarr, numtrsamp,
                                sigmasqr, dim, C))
                return 1;
        else 
                return 0;
}

/*
 * SMO main
 */
void smo_main(double **inputvecmat, const int numtrsamp, const double *yarr, 
                double *alpharr, double *thresh, const double C, 
                const double sigmasqr, const double * precompdotprod, const int dim)
{
        double b_up, b_low; 
        int i_up, i_low, i;
        int iupsetflag, ilowsetflag, numchanged, examineall;
        int *setinfo; 
        double *fcache; 

        setinfo = (int *) malloc(sizeof(int) * numtrsamp);
        fcache= (double*)malloc(sizeof(double) * numtrsamp);
        for ( i = 0; i < numtrsamp; i++)
        {
                setset(i, setinfo, alpharr, yarr, C);
                //initially all alphas are zero
                fcache[i] = mone * yarr[i];
        }
        b_up = mone; b_low = pone;
        iupsetflag = ilowsetflag = 0;
        /*
         * for two training samples of different classes, 
         * initialize the set info of those points
         * to 1 and 4. 
         */
        for ( i = 0; i < numtrsamp; i++)
        {
                if ( !iupsetflag && fabs(yarr[i] - pone) < ZERO)
                {
                        i_up = i;
                        iupsetflag = 1;
                        setinfo[i] = 1;
                }
                if ( !ilowsetflag && fabs(yarr[i] - mone) < ZERO)
                {
                        i_low = i;
                        ilowsetflag = 1;
                        setinfo[i] = 4;
                }
                if ( iupsetflag && ilowsetflag)
                        break;
        }
        fcache[i_low] = pone;
        fcache[i_up] = mone;
        numchanged = 0;
        examineall = 1;

        while (numchanged > 0 || examineall)
        {
                numchanged = 0;
                if (examineall)
                {
                        //loop over all training samples
                        for (i = 0; i < numtrsamp; i++)
                        {
                                numchanged += examineexample(i, alpharr, setinfo, fcache,
                                                &b_low, &b_up, &i_low, &i_up, inputvecmat,
                                                precompdotprod, yarr, numtrsamp, sigmasqr,
                                                dim, C);
                        }

                }
                else
                {
                        for (i=0; i < numtrsamp; i++)
                        {
                                if (setinfo[i] == 0)
                                {
                                        numchanged += examineexample(i, alpharr, setinfo,
                                                        fcache, &b_low, &b_up, &i_low, &i_up,
                                                        inputvecmat, precompdotprod, yarr,
                                                        numtrsamp, sigmasqr, dim, C);
                                        ;
                                        //check for only these I
                                        if ( b_up > b_low - 2 * TOL)
                                        {
                                                numchanged = 0;
                                                break;
                                        }
                                }
                        }
                }
                if (examineall)
                        examineall = 0;
                else if(!numchanged)
                        examineall = 1;
        }
        *thresh = addFcache(alpharr, fcache, b_up, b_low, numtrsamp, C);  
        free(fcache);
        free(setinfo);
}
/*
 * main routine - calls modification 1 in the paper
 */
int main(int argc, char *argv[])
{
        int i, kc, spde;
        unsigned int svcnt = 0, bsvcnt = 0, zsvcnt = 0;
        double secelap, testout;
        int (*filereadingfunc)(char*, double**, double*, const int, const int);
        int rfile(char*, double**, double*, const int, const int);
        int sparserfile(char*, double**, double*, const int, const int);
        struct timeval t[2];
        /*
         * global to local declarations change
         */
        double *precompdotprod, *precompdotprod_test;
        double **inputvecmat, *yarr, *alpharr, **testvecmat, *testy;
        double C, sigmasqr, thresh;
        int numtrsamp, dim, numtestsamp;

        if (argc != 10)
        {
                fprintf(stderr, "Usage:\n");
                fprintf(stderr, "# training_samples, tr_file\n");
                fprintf(stderr, "dim, C, sigmasqr/degree\n");
                fprintf(stderr, "# test_samples, Test_file\n");
                fprintf(stderr,"kernelchoice(1:RBF,2:dp 3:poly)\n");
                fprintf(stderr, "file format:sparse(1)/dense(0)\n");
                exit(1);
        }
        numtrsamp = atoi(argv[1]);
        dim = atoi(argv[3]);
        C = atof (argv[4]);
        sigmasqr = atof(argv[5]);
        numtestsamp = atoi(argv[6]);
        kc = atoi(argv[8]);
        spde = atoi(argv[9]);

        precompdotprod = (double *) malloc(sizeof(double) * numtrsamp);
        precompdotprod_test = (double *) malloc(sizeof(double) * numtestsamp);
        inputvecmat = (double **) malloc( sizeof(double*) * numtrsamp);
        inputvecmat[0] = (double *)calloc ((unsigned int) (dim * numtrsamp), sizeof(double));
        for ( i = 1; i < numtrsamp; i++)
                inputvecmat[i] = inputvecmat[i-1] + dim;
        yarr = (double *)malloc(sizeof(double) * numtrsamp);
        if (spde == 1)
        {
                filereadingfunc = sparserfile;
        }
        else if(spde == 0)
        {
                filereadingfunc = rfile;
        }
        else
        {
                fprintf(stderr, "Unknown file format...exiting\n");
                exit(0);
        }
        if(!filereadingfunc(argv[2], inputvecmat, yarr, numtrsamp, dim))
        {
                fprintf(stderr, "File not opened\n");
                free(inputvecmat[0]);
                free(inputvecmat);
                free(yarr);
                exit(0);
        }
        testvecmat = (double **)malloc(sizeof(double*) * numtestsamp);
        testvecmat[0] = (double *)calloc ((unsigned int)(dim * numtestsamp), sizeof(double));
        /*  memset ( testvecmat[0], 0, sizeof(double) * 
                        (unsigned int)(dim * numtestsamp));*/
        for ( i = 1; i < numtestsamp; i++)
                testvecmat[i] = testvecmat[i-1] + dim;
        testy = (double *)malloc(sizeof(double) * numtestsamp);
        alpharr = (double*) malloc(sizeof(double) * numtrsamp);
        memset(alpharr, 0, sizeof(double) * numtrsamp);
        /*
         * initialize the function pointer for kernel function call
         */
        if (kc == 1)
        {
                kernelchoice = RBF;
                testkernelchoice = testRBF;
        }
        else if (kc == 2)
        {
                kernelchoice = dp;
                testkernelchoice = testdp;
        }
        else if (kc == 3)
        {
                kernelchoice = polykernel;
                testkernelchoice = testpolykernel;

        }
        else
        {
                fprintf(stderr, "unknown kernel choice, using RBF instead\n");
                kernelchoice = RBF;
                testkernelchoice = testRBF;
        }

        precomputedp(inputvecmat, numtrsamp, precompdotprod, dim);
        gettimeofday(t + 0, NULL);
        smo_main(inputvecmat, numtrsamp, yarr, alpharr, &thresh, C, sigmasqr,
                        precompdotprod, dim);
        gettimeofday( t + 1, NULL);

        secelap = (t[1].tv_usec - t[0].tv_usec) * 1e-6 +
                (t[1].tv_sec - t[0].tv_sec);
        fprintf(stdout, "Time in seconds:%.5lf\n", secelap);
        fprintf(stdout, "Thresh:%.4lf\n", thresh);
        fflush(stdout);	
        if (dim == 2)
                fprintf(stdout, "Alpharr\n");
        for (i = 0; i < numtrsamp; i++)
        {
                if (dim == 2)
                        fprintf(stdout, "%.5lf\n", alpharr[i]);
                if (alpharr[i] > ZERO && fabs(alpharr[i]- C) > ZERO)
                        svcnt ++;
                else
                        if( fabs(alpharr[i] - C) < ZERO)
                                bsvcnt ++;
                        else
                                zsvcnt ++;
        }
        fprintf(stdout,"The nBsvs:%d Bsvs:%d Zsvs:%d\n", 
                        svcnt, bsvcnt, zsvcnt);
        fprintf(stdout, "# of kernel eval * 1e-6:%lf\n", 
                        numkerneleval*1e-6);
        /*
         * Read the testing set and calculate the generalization error 
         */
        if (!filereadingfunc (argv[7], testvecmat, testy, numtestsamp, dim))
        {
                fprintf(stderr,"testfile not opened\n");
                exit(0);
        }
        precomputedp_test(testvecmat, numtestsamp, precompdotprod_test, dim);
        testout = testsamples(inputvecmat, testvecmat, alpharr, yarr, numtrsamp,
                        dim, sigmasqr, thresh, numtestsamp, testy, precompdotprod,
                        precompdotprod_test);
        fprintf(stdout,"The error rate:%.2lf\n",
                        ((double)testout * 100.0) 
                        / (double)numtestsamp);
        fflush(stdout);

        free (yarr);
        free(alpharr);
        free(precompdotprod);
        free(precompdotprod_test);
        free(inputvecmat[0]);
        free(inputvecmat);
        free(testvecmat[0]);
        free(testvecmat);
        free (testy);
        return 1;
}

/* examineAll loop calls this function, for the
 * condition when passed id is not in I_0
 */
double fcalc(int idx, const double *alpharr, const double *yarr, 
                double **inputvecmat, const double *precompdotprod, 
                const int dim, const double sigmasqr, const int numtrsamp)
{
        double retf = 0.0, tempscalar;
        int i;
        for ( i = 0; i < numtrsamp; i++)
        {
                tempscalar = alpharr[i] * yarr[i];
                /*tempscalar *= RBF(i, idx);
                  tempscalar *= dp(i, idx);*/
                tempscalar *= kernelchoice(i, idx, inputvecmat, precompdotprod, dim, 
                                sigmasqr);
                retf += tempscalar;
        }
        retf = retf - yarr[idx];	
        return retf;
}

/*-----------------------------------------------------------------------------
  routine to tokenize the line and set the return array. Takes the to-be parsed
  string and output vector to contain tokenized words as inputs. Returns 0 if the
  input string is empty one and returns 1 otherwise.
  Calls remove_space to remove the leading and trailing spaces
 *-----------------------------------------------------------------------------*/
int tokenize (string token, vector<string>&retarray)
{
        const string SPACES= " \t\r\n\f";
        void remove_space(string&, const string);
        string tempstr;
        const string singlespace=" ";
        bool flag = false;
        size_t tab_place, char_place;
        //remove the leading and trailing spaces
        remove_space(token, SPACES);
        if (token.empty())
                return 0;
        tab_place = token.find_first_of(SPACES);
        char_place = 0;
        while(tab_place != string::npos)
        {
                flag = true;
                tempstr = token.substr(char_place,tab_place-char_place);
                retarray.push_back(tempstr);
                char_place = token.find_first_not_of(SPACES, tab_place);
                tab_place = token.find_first_of(SPACES, char_place);

        }
        //there is single word in the file
        if(!flag)
                retarray.push_back(token);
        else//insert the last word of the line into the vector
                retarray.push_back(token.substr(char_place));
        return 1;
}

//function to remove leading and trailing spaces around
//a string
void remove_space(string &node, const string spaces)
{
        size_t spacestart, spaceend;

        spacestart = node.find_first_not_of(spaces);
        spaceend = node.find_last_not_of(spaces);
        if (spacestart == string::npos)
                node="";
        else
                node = node.substr(spacestart, spaceend+1-spacestart);
}
/*
 * file input for training/test - sparse format
 */
int sparserfile(char *fname, double **mat, double *acty, const int samplesize, 
                const int dim)
{
        ifstream ifs;
        vector <string> retarray;
        string token, ttoken;
        int i = 0, j, icol, max = -10;
        double val;
        size_t colonplace;
        int tokenize (string, vector<string>&);

        ifs.open(fname);
        while(!ifs.eof())
        {
                getline(ifs, token);
                if(!tokenize(token, retarray))
                {
                        if(ifs.eof())
                                break;
                        else
                        {
                                cerr<<"empty line in file...exiting"<<endl;
                                return 0;
                        }
                }
                //first entity is always class label
                *(acty + i) = atof(retarray[0].c_str());
                for(j = 1; j < (int)retarray.size(); j++)
                {
                        ttoken =  retarray[j];
                        //find the first colon (:)
                        colonplace = ttoken.find_first_of(':');
                        if (colonplace == string::npos)
                        {
                                cerr<<"File not in sparse format"<<endl;
                                return 0;
                        }
                        icol = atoi((ttoken.substr(0, colonplace)).c_str()); 
                        val = atof((ttoken.substr(colonplace+1, 
                                                        string::npos)).c_str()); 
                        mat[i][icol] = val;
                }
                if (icol > max) 
                        max = icol;
                retarray.clear();
                i++;
        }
        if (i != samplesize)
        {
                cerr<<"The number of lines: "<<i<<endl;
                cerr<<"insufficient lines in the file"<<endl;
                return 0;
        }
        return 1;
}
/*
 * file input for training/test set - dense format
 */
int rfile(char *fname, double **mat, double *acty, const int samplesize, const int dim)
{
        // FILE *fp;
        ifstream ifs;
        vector <string>retarray;
        string token;
        int i = 0, icol, j;
        int tokenize(string, vector<string>&);

        ifs.open(fname);

        while(!ifs.eof())
        {
                getline(ifs, token);
                if(!tokenize(token, retarray))
                {
                        if(ifs.eof())
                                break;
                        else
                        {
                                cerr<<"empty line in file...exiting"<<endl;
                                return 0;
                        }
                }
                //first entity is always class label
                *(acty + i) = atof(retarray[0].c_str());
                for (j = 1, icol = 0; j < (int)retarray.size();j++, icol++)
                {
                        mat[i][icol] = atof(retarray[j].c_str());
                }
                if (icol != dim)
                {
                        cerr<<"insufficient columns in line "<<i+1<<endl;
                        return 0;
                }
                retarray.clear();
                i++;
        }
        if (i != samplesize)
        {
                cerr<<"insufficient lines in the file"<<endl;
                return 0;
        }
        return 1;
}

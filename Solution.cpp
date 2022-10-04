#include <stdio.h>
#include <math.h>
#include <armadillo>

using namespace arma;

double mu = 1.;
double tau = 0.1;
double h_1 = 0.1;
double h_2 = 0.1;
double pi = M_PI;

unsigned long int N = static_cast<unsigned long int>(1 / tau);
unsigned long int M_1 = static_cast<unsigned long int>(1 / h_1);
unsigned long int M_2 = static_cast<unsigned long int>(1 / h_2);

double p(double rho)                                            //rho = exp(t)
{
    return pow(rho, 1.4);
}

double u_1(double x, double y, double t)
{
    return sin(pi * x) * sin(pi * y) * t;
}

double u_2(double x, double y, double t)
{
    return sin(pi * x * x) * sin(pi * y * y) * t;
}

double norm(int n, mat matr, int ind)
{
    double sum = 0.;
    for(unsigned long int i = 0; i < M_1+1; i++)
    {
        for(unsigned long int j = 0; j < M_2 + 1; j++)
        {
            if(ind == 1)
                matr(i, j) -= u_1(static_cast<double>(i) * h_1, static_cast<double>(j) * h_2, static_cast<double>(n) * tau);
            else if(ind == 2)
                matr(i, j) -= u_2(static_cast<double>(i) * h_1, static_cast<double>(j) * h_2, static_cast<double>(n) * tau);
            else
                matr(i, j) -= exp(static_cast<double>(n) * tau);
            sum += matr(i, j)*matr(i, j);
        }
    }
    sum = pow(sum*h_1*h_2, 0.5);
    return sum;
}

mat create_f0(int n)
{
    mat f0(static_cast<arma::uword>(M_1) + 1, static_cast<arma::uword>(M_2) + 1, fill::ones);
    double x;
    double y;
    double t;
//     double f;
    for(unsigned long int i = 0; i < M_1+1; i++)
    {
        for(unsigned long int j=0; j < M_2+1; j++)
        {
            x = static_cast<double>(i) * h_1;
            y = static_cast<double>(j) * h_2;
            t = static_cast<double>(n) * tau;
            
//             f = exp(t)*(1+pi*cos(pi*x)*sin(pi*y)*t+pi*2*y*sin(pi*x*x)*cos(pi*y*y)*t);
            if (i!=0 && i!=M_1 && j!=0 && j!=M_2)
                f0(i, j) = (exp(t+tau)-exp(t))/tau + 0.5*((exp(t+tau)+exp(t))*(u_1(x+h_1,y,t)-u_1(x-h_1,y,t))/(2*h_1)+(exp(t+tau)+exp(t))*(u_2(x,y+h_2,t)-u_2(x,y-h_2,t))/(2*h_2));
            else if (i==0)
                f0(i, j) = (exp(t+tau)-exp(t))/tau + 0.5*u_1(x+h_1,y,t)*(exp(t+tau)+exp(t))/h_1 - exp(t)*0.5/h_1 * ((u_1(x+2*h_1,y,t)-2*u_1(x+h_1,y,t))-0.5*(u_1(x+3*h_1,y,t)-2*u_1(x+2*h_1,y,t)+u_1(x+h_1,y,t))+(u_1(x+2*h_1,y,t)-2*u_1(x+h_1,y,t))-0.5*(u_1(x+3*h_1,y,t)-2*u_1(x+2*h_1,y,t)+u_1(x+h_1,y,t)));
            else if (j==0)
                f0(i, j) = (exp(t+tau)-exp(t))/tau + 0.5*u_2(x,y+h_2,t)*(exp(t+1)+exp(t))/h_2 - exp(t)*0.5/h_2 * ((u_2(x,y+2*h_2,t)-2*u_2(x,y+h_2,t))-0.5*(u_2(x,y+3*h_2,t)-2*u_2(x,y+2*h_2,t)+u_2(x,y+h_2,t))+(u_2(x,y+2*h_2,t)-2*u_2(x,y+h_2,t))-0.5*(u_2(x,y+3*h_2,t)-2*u_2(x,y+2*h_2,t)+u_2(x,y+h_2,t)));
            else if (i==M_1)
                f0(i, j) = (exp(t+tau)-exp(t))/tau - 0.5*u_1(x-h_1,y,t)*(exp(t+tau)+exp(t))/h_1 + exp(t)*0.5/h_1 * ((u_1(x-2*h_1,y,t)-2*u_1(x-h_1,y,t))-0.5*(u_1(x-3*h_1,y,t)-2*u_1(x-2*h_1,y,t)+u_1(x-h_1,y,t))+(u_1(x-2*h_1,y,t)-2*u_1(x-h_1,y,t))-0.5*(u_1(x-3*h_1,y,t)-2*u_1(x-2*h_1,y,t)+u_1(x-h_1,y,t)));
            else if (j==M_1)
                f0(i, j) = (exp(t+tau)-exp(t))/tau - 0.5*u_2(x,y-h_2,t)*(exp(t+tau)+exp(t))/h_2 + exp(t)*0.5/h_2 * ((u_2(x,y-2*h_2,t)-2*u_2(x,y-h_2,t))-0.5*(u_2(x,y-3*h_2,t)-2*u_2(x,y-2*h_2,t)+u_2(x,y-h_2,t))+(u_2(x,y-2*h_2,t)-2*u_2(x,y-h_2,t))-0.5*(u_2(x,y-3*h_2,t)-2*u_2(x,y-2*h_2,t)+u_2(x,y-h_2,t)));
//             if (i==1 && j==1)
//                 printf("f0 - %lf\n", f-f0(i,j));
        }
    }
    return f0;
}

mat create_f1(int n)
{
    mat f1(static_cast<arma::uword>(M_1) - 1, static_cast<arma::uword>(M_2) - 1, fill::ones);
    double x;
    double y;
    double t;
//     double f;
    double val_0, val_1, val_2, val_3, val_4;//, val_5, val_6;
    for(unsigned long int i = 1; i < M_1; i++)
    {
        for(unsigned long int j=1; j < M_2; j++)
        {
            x = static_cast<double>(i) * h_1;
            y = static_cast<double>(j) * h_2;
            t = static_cast<double>(n) * tau;
            
//             val_0 = sin(pi*x)*sin(pi*y)*t;
//             val_1 = sin(pi*x)*sin(pi*y);
//             val_2 = 2*pi*sin(pi*x)*sin(pi*y)*t*cos(pi*x)*sin(pi*y)*t;
//             val_3 = pi*sin(pi*x)*cos(pi*y)*t*sin(pi*x*x)*sin(pi*y*y)*t + pi*2*y*sin(pi*x)*sin(pi*y)*t*sin(pi*x*x)*cos(pi*y*y)*t;
//             val_4 = -pi*pi*sin(pi*x)*sin(pi*y)*t;
//             val_5 = -pi*pi*sin(pi*x)*sin(pi*y)*t;
//             val_6 = 4*pi*pi*x*y*cos(pi*x*x)*cos(pi*y*y)*t;
            
//             f = val_0 + val_1 + val_2 + val_3 - mu / exp(t) * (4./3.*val_4 + val_5 + val_6/3.);
            val_0 = exp(t+tau)/tau*(u_1(x,y,t+tau)-u_1(x,y,t));
            val_1 = exp(t+tau)*u_1(x,y,t)/2/h_1*(u_1(x+h_1,y,t+tau)-u_1(x-h_1,y,t+tau))+ exp(t+tau)*(u_1(x+h_1,y,t)*u_1(x+h_1,y,t+tau)-u_1(x-h_1,y,t)*u_1(x-h_1,y,t+tau))/2/h_1;
            val_2 = exp(t+tau)*(u_2(x,y,t)*(u_1(x,y+h_2,t+tau)-u_1(x,y-h_2,t+tau)) + (u_2(x,y+h_2,t)*u_1(x,y+h_2,t)-u_2(x,y-h_2,t)*u_1(x,y-h_2,t)) - u_1(x,y,t)*(u_2(x,y+h_2,t)-u_2(x,y-h_2,t)))/2/h_2;
            val_3 = (u_1(x+h_1,y,t+tau)-2*u_1(x,y,t+tau)+u_1(x-h_1,y,t+tau))/h_1/h_1;
            val_4 = (u_1(x,y+h_2,t+tau)-2*u_1(x,y,t+tau)+u_1(x,y-h_2,t+tau))/h_2/h_2 +1/12.*(u_2(x+h_1,y+h_2,t)-u_2(x+h_1,y-h_2,t)-u_2(x-h_1,y+h_2,t)+u_2(x-h_1,y-h_2,t))/h_1/h_2;
            
            f1(i-1, j-1) = ((val_0 + 1/3.*val_1 + 0.5*val_2) - mu*(4/3.*val_3 + val_4))/exp(t+tau);
//             if (i==1 && j==1)
//                 printf("f1 - %lf\n", f-f1(i-1, j-1));
        }
    }
    return f1;
}

mat create_f2(int n)
{
    mat f2(static_cast<arma::uword>(M_1) - 1, static_cast<arma::uword>(M_2) - 1, fill::ones);
    double x;
    double y;
    double t;
//     double f;
    double val_0, val_1, val_2, val_3, val_4;//, val_5, val_6;
    for(unsigned long int i = 1; i < M_1; i++)
    {
        for(unsigned long int j=1; j < M_2; j++)
        {
            x = static_cast<double>(i) * h_1;
            y = static_cast<double>(j) * h_2;
            t = static_cast<double>(n) * tau;
            
//             val_0 = sin(pi*x*x)*sin(pi*y*y)*t;
//             val_1 = sin(pi*x*x)*sin(pi*y*y);
//             val_2 = 4*pi*y*sin(pi*x*x)*sin(pi*y*y)*t*sin(pi*x*x)*cos(pi*y*y)*t;
//             val_3 = pi*cos(pi*x)*sin(pi*y)*t*sin(pi*x*x)*sin(pi*y*y)*t + pi*2*x*sin(pi*x)*sin(pi*y)*t*cos(pi*x*x)*sin(pi*y*y)*t;
//             val_4 = pi*2*sin(pi*x*x)*cos(pi*y*y)*t-pi*pi*4*y*y*sin(pi*x*x)*sin(pi*y*y)*t;
//             val_5 = pi*2*cos(pi*x*x)*sin(pi*y*y)*t-pi*pi*4*x*x*sin(pi*x*x)*sin(pi*y*y)*t;
//             val_6 = pi*pi*cos(pi*x)*cos(pi*y)*t;
            
//             f = val_0 + val_1 + val_2 + val_3 - mu / exp(t) * (4./3.*val_4 + val_5 + val_6/3.);
            
            val_0 = exp(t+tau)/tau*(u_2(x,y,t+tau)-u_2(x,y,t));
            val_1 = exp(t+tau)*u_2(x,y,t)/2/h_2*(u_2(x,y+h_2,t+tau)-u_2(x,y-h_2,t+tau))+ exp(t+tau)*(u_2(x,y+h_2,t)*u_2(x,y+h_2,t+tau)-u_2(x,y-h_2,t)*u_2(x,y-h_2,t+tau))/2/h_2;
            val_2 = exp(t+tau)*(u_1(x,y,t)*(u_2(x+h_1,y,t+tau)-u_2(x-h_1,y,t+tau)) + (u_1(x+h_1,y,t)*u_2(x+h_1,y,t)-u_1(x-h_1,y,t)*u_2(x-h_1,y,t)) - u_2(x,y,t)*(u_1(x+h_1,y,t)-u_1(x-h_1,y,t)))/2/h_1;
            val_3 = (u_2(x,y+h_2,t+tau)-2*u_2(x,y,t+tau)+u_2(x,y-h_2,t+tau))/h_2/h_2;
            val_4 = (u_2(x+h_1,y,t+tau)-2*u_2(x,y,t+tau)+u_2(x-h_1,y,t+tau))/h_1/h_1 +1/12.*(u_1(x+h_1,y+h_2,t)-u_1(x+h_1,y-h_2,t)-u_1(x-h_1,y+h_2,t)+u_1(x-h_1,y-h_2,t))/h_1/h_2;
            
            f2(i-1, j-1) = ((val_0 + 1/3.*val_1 + 0.5*val_2) - mu*(4/3.*val_3 + val_4))/exp(t+tau);
//             if (i==1 && j==1)
//                 printf("f2 - %lf\n", f-f2(i-1, j-1));
        }
    }
    return f2;
}

mat create_A1(mat V_1, mat V_2)
{
    mat A1((static_cast<arma::uword>(M_1)+1)*(static_cast<arma::uword>(M_2)+1), (static_cast<arma::uword>(M_1)+1)*(static_cast<arma::uword>(M_2)+1), fill::zeros);
    for(unsigned long int block_num=0; block_num < M_1+1; block_num++)
    {
        if(block_num == 0)
        {
            for(unsigned long int i=0; i < M_2+1; i++)
            {
                A1(i, i) = 1 / tau;
                if (i!=0 && i!=M_2)
                    A1(i, (M_2+1)+i) = 0.5 * V_1(1,i) / h_1;
            }
        }
        else if(block_num == M_1)
        {
            for(unsigned long int i=0; i < M_2+1; i++)
            {
                unsigned long int I = M_1 * (M_2 + 1) + i;
                A1(I, I) = 1 / tau;
                if (i!=0 && i!=M_2)
                    A1(I, (M_1-1)*(M_2+1)+i) = -0.5 * V_1(M_1-1,i) / h_1;
            }
        }
        else
        {
            for(unsigned long int i=0; i < M_2+1; i++)
            {
                unsigned long int I = block_num * (M_2 + 1) + i;
                A1(I, I) = 1 / tau;
                if(i==0)
                    A1(I, I+1) = 0.5 * V_2(block_num, i+1) / h_2;
                else if(i==M_2)
                    A1(I, I-1) = -0.5 * V_2(block_num,i-1) / h_2;
                else
                {
                    A1(I, I - 1) = -0.25 * (V_2(block_num,i) + V_2(block_num,i-1)) / h_2;
                    A1(I, I + 1) = 0.25 * (V_2(block_num, i) + V_2(block_num, i+1)) / h_2;
                    A1(I,I - (M_2 + 1)) = -0.25 * (V_1(block_num, i) + V_1(block_num-1, i)) / h_1;
                    A1(I,I + (M_2 + 1)) = 0.25 * (V_1(block_num, i) + V_1(block_num+1, i)) / h_1;
                }
            }
        }
    }
    return A1;
}

vec create_b1(mat f0, mat V_1, mat V_2, mat H)
{
    vec b1((static_cast<arma::uword>(M_1)+1)*(static_cast<arma::uword>(M_2)+1), fill::zeros);
    double val_1, val_2, val_3, val_4, val_5;
    for(unsigned long int block_num=0; block_num < M_1+1; block_num++)
    {
        if(block_num == 0)
        {
            for(unsigned long int i=0; i < M_2+1; i++)
            {
                val_1 = - H(block_num,i) / tau;
                val_2 = 0.5 * H(block_num,i) * V_1(block_num+1,i) / h_1;
                val_3 = -0.5 * (H(block_num+2,i) * V_1(block_num+2,i) - 2 * H(block_num+1,i) * V_1(block_num+1,i)) / h_1;
                val_4 = 0.25 * (H(block_num+3,i) * V_1(block_num+3,i) - 2 * H(block_num+2,i) * V_1(block_num+2,i) + H(block_num+1,i) * V_1(block_num+1,i)) / h_1;
                val_5 = -0.5*H(block_num,i) * (V_1(block_num+2,i) - 2*V_1(block_num+1,i) - 0.5*(V_1(block_num+3,i) - 2*V_1(block_num+2,i) + V_1(block_num+1,i))) / h_1;
                b1(block_num*(M_2+1)+i) = f0(block_num,i) - val_1 - val_2 - val_3 - val_4 - val_5;
            }
        }
        else if(block_num == M_1)
        {
            for(unsigned long int i=0; i < M_2+1; i++)
            {
                val_1 = - H(block_num,i) / tau;
                val_2 = -0.5 * H(block_num,i) * V_1(block_num-1, i) / h_1;
                val_3 = 0.5 * (-2 * H(block_num-1,i) * V_1(block_num-1,i) + H(block_num-2,i) * V_1(block_num-2,i)) / h_1;
                val_4 = -0.25 * (H(block_num-1,i) * V_1(block_num-1,i) - 2 * H(block_num-2,i) * V_1(block_num-2,i) + H(block_num-3,i) * V_1(block_num-3,i)) / h_1;
                val_5 = 0.5 * H(block_num,i) * (-2*V_1(block_num-1,i) + V_1(block_num-2,i) - 0.5*(V_1(block_num-1,i) - 2*V_1(block_num-2,i) + V_1(block_num-3,i))) / h_1;
                b1(block_num*(M_2+1)+i) = f0(block_num,i) - val_1 - val_2 - val_3 - val_4 - val_5;
            }
            
        }
        else
        {
            for(unsigned long int i=0; i < M_2+1; i++)
            {
                if(i==0)
                {
                    val_1 = - H(block_num,i) / tau;
                    val_2 = 0.5 * H(block_num,i) * V_2(block_num,i+1) / h_2;
                    val_3 = -0.5 * (H(block_num,i+2) * V_2(block_num,i+2) - 2 * H(block_num,i+1) * V_2(block_num,i+1)) / h_2;
                    val_4 = 0.25 * (H(block_num,i+3) * V_2(block_num,i+3) - 2 * H(block_num,i+2) * V_2(block_num,i+2) + H(block_num,i+1) * V_2(block_num,i+1)) / h_2;
                    val_5 = -0.5*H(block_num,i) * (V_2(block_num,i+2) - 2*V_2(block_num,i+1) - 0.5*(V_2(block_num,i+3) - 2*V_2(block_num,i+2) + V_2(block_num,i+1))) / h_2;
                    b1(block_num*(M_2+1)+i) = f0(block_num,i) - val_1 - val_2 - val_3 - val_4 - val_5;
                }
                else if(i==M_2)
                {
                    val_1 = - H(block_num,i) / tau;
                    val_2 = -0.5 * H(block_num,i) * V_2(block_num,i-1) / h_2;
                    val_3 = 0.5 * (- 2 * H(block_num,i-1) * V_2(block_num,i-1) + H(block_num,i-2) * V_2(block_num,i-2)) / h_2;
                    val_4 = -0.25 * (H(block_num,i-1) * V_2(block_num,i-1) - 2 * H(block_num,i-2) * V_2(block_num,i-2) + H(block_num,i-3) * V_2(block_num,i-3)) / h_2;
                    val_5 = 0.5*H(block_num,i) * (-2*V_2(block_num,i-1) + V_2(block_num,i-2) - 0.5*(V_2(block_num,i-1) - 2*V_2(block_num,i-2) + V_2(block_num,i-3))) / h_2;
                    b1(block_num*(M_2+1)+i) = f0(block_num,i) - val_1 - val_2 - val_3 - val_4 - val_5;
                }
                else
                {
                    val_1 = - H(block_num,i) / tau;
                    val_2 = 0.25 * H(block_num,i) * (V_1(block_num+1,i) - V_1(block_num-1,i)) / h_1;
                    val_3 = 0.25 * H(block_num,i) * (V_2(block_num,i+1) - V_2(block_num,i-1)) / h_2;
                    b1(block_num*(M_2+1)+i) = f0(block_num,i) - val_1 - val_2 - val_3;
                }
            }
        }
    }
    return b1;
}

mat create_A2(mat V_1, mat V_2, mat H)
{
    mat A2((static_cast<arma::uword>(M_1)-1)*(static_cast<arma::uword>(M_2)-1), (static_cast<arma::uword>(M_1)-1)*(static_cast<arma::uword>(M_2)-1), fill::zeros);
    for(unsigned long int block_num=0; block_num<M_1-1; block_num++)
    {
        for(unsigned long int i=0; i < M_2-1; i++)
        {
            unsigned long int I = block_num * (M_2 - 1) + i;

            A2(I,I) = H(block_num+1,i+1)/tau+8*mu/(3*h_1*h_1)+2*mu/(h_2*h_2);
            if (i!=0)
                A2(I,I-1) = -0.5*H(block_num+1,i+1) * V_2(block_num+1,i+1) / (2*h_2) -0.5*H(block_num+1,i-1+1) * V_2(block_num+1,i-1+1) / (2*h_2) - mu / (h_2 * h_2);
            if (i!=M_2-2)
                A2(I,I+1) = 0.5*H(block_num+1,i+1) * V_2(block_num+1,i+1) / (2*h_2) + 0.5*H(block_num+1,i+1+1) * V_2(block_num+1,i+1+1) / (2*h_2) - mu / (h_2 * h_2);
            if (block_num!=0)
                A2(I,(block_num-1)*(M_2-1)+i) = -1./3*H(block_num+1,i+1) * V_1(block_num+1,i+1) / (2*h_1) -1./3*H(block_num-1+1,i+1) * V_1(block_num-1+1,i+1) / (2*h_1) -4*mu/(3*h_1 * h_1);
            if (block_num!=M_1-2)
                A2(I,(block_num+1)*(M_2-1)+i) = 1./3*H(block_num+1,i+1) * V_1(block_num+1,i+1) / (2*h_1) +1./3*H(block_num+1+1,i+1) * V_1(block_num+1+1,i+1) / (2*h_1) -4*mu/(3*h_1 * h_1);
        }
    }
    return A2;
}    

vec create_b2(mat f1, mat V_1, mat V_2, mat H)
{
    vec b2((static_cast<arma::uword>(M_1) - 1) * (static_cast<arma::uword>(M_2) - 1), fill::zeros);
    double val_1, val_2, val_3, val_4, val_5, val_6;
    for(unsigned long int block_num=0; block_num<M_1-1; block_num++)
    {
        for(unsigned long int i=0; i<M_2-1; i++)
        {
            val_1 = -H(block_num+1,i+1) * V_1(block_num+1,i+1) / tau;
            val_2 = - 1 / 3 * V_1(block_num+1,i+1) * V_1(block_num+1,i+1) * (H(block_num+1+1,i+1) - H(block_num-1+1,i+1)) / (2 * h_1);
            val_3 = -0.5 * V_1(block_num+1,i+1) * (H(block_num+1,i+1+1) * V_2(block_num+1,i+1+1) - H(block_num+1,i-1+1) * V_2(block_num+1,i-1+1)) / (2 * h_2);
            val_4 = (p(H(block_num+1+1,i+1)) - p(H(block_num-1+1,i+1))) / (2 * h_1);
            
            val_5 = mu / 3 * (V_2(block_num+1+1,i+1+1) - V_2(block_num+1+1,i-1+1) - V_2(block_num-1+1,i+1+1) + V_2(block_num-1+1,i-1+1)) / (4 * h_1 * h_2);
            val_6 = H(block_num+1,i+1) * f1(block_num,i);
            
            b2(block_num * (M_2-1) + i) = val_5 + val_6 - val_1 - val_2 - val_3 - val_4;
        }
    }
    return b2;
}

mat create_A3(mat V_1, mat V_2, mat H)
{
    mat A3((static_cast<arma::uword>(M_1) - 1) * (static_cast<arma::uword>(M_2) - 1), (static_cast<arma::uword>(M_1) - 1) * (static_cast<arma::uword>(M_2) - 1), fill::zeros);
    for(unsigned long int block_num=0; block_num<M_1-1; block_num++)
    {
        for(unsigned long int i=0; i < M_2-1; i++)
        {
            unsigned long int I = block_num * (M_2 - 1) + i;
            
            A3(I,I) = H(block_num+1,i+1)/tau+8*mu/(3*h_2*h_2)+2*mu/(h_1*h_1);
            if (i!=0)
                A3(I,I-1) = -1./3*H(block_num+1,i+1) * V_2(block_num+1,i+1) / (2*h_2) -1./3*H(block_num+1,i-1+1) * V_2(block_num+1,i-1+1) / (2*h_2) -4*mu/(3*h_2 * h_2);
            if (i!=M_2-2)
                A3(I,I+1) = 1./3*H(block_num+1,i+1) * V_2(block_num+1,i+1) / (2*h_2) +1./3*H(block_num+1,i+1+1) * V_2(block_num+1,i+1+1) / (2*h_2) -4*mu/(3*h_2 * h_2);
            if (block_num!=0)
                A3(I,(block_num-1)*(M_2-1)+i) = -0.5*H(block_num+1,i+1) * V_1(block_num+1,i+1) / (2*h_1) -0.5*H(block_num-1+1,i+1) * V_1(block_num-1+1,i+1) / (2*h_1) -mu/(h_1 * h_1);
            if (block_num!=M_1-2)
                A3(I,(block_num+1)*(M_2-1)+i) = 0.5*H(block_num+1,i+1) * V_1(block_num+1,i+1) / (2*h_1) +0.5*H(block_num+1+1,i+1) * V_1(block_num+1+1,i+1) / (2*h_1) -mu/(h_1 * h_1);
        }
    }
    return A3;
} 

vec create_b3(mat f2, mat V_1, mat V_2, mat H)
{
    vec b3((static_cast<arma::uword>(M_1) - 1) * (static_cast<arma::uword>(M_2) - 1), fill::zeros);
    double val_1, val_2, val_3, val_4, val_5, val_6;
    for(unsigned long int block_num=0; block_num<M_1-1; block_num++)
    {
        for(unsigned long int i=0; i<M_2-1; i++)
        {
            val_1 = -H(block_num+1,i+1) * V_2(block_num+1,i+1) / tau;
            val_2 = - 1 / 3 * V_2(block_num+1,i+1) * V_2(block_num+1,i+1) * (H(block_num+1,i+1+1) - H(block_num+1,i-1+1)) / (2 * h_2);
            val_3 = -0.5 * V_2(block_num+1,i+1) * (H(block_num+1+1,i+1) * V_1(block_num+1+1,i+1) - H(block_num-1+1,i+1) * V_1(block_num-1+1,i+1)) / (2 * h_1);
            val_4 = (p(H(block_num+1,i+1+1)) - p(H(block_num+1,i-1+1))) / (2 * h_2);
            
            val_5 = mu / 3 * (V_1(block_num+1+1,i+1+1) - V_1(block_num-1+1,i+1+1) - V_1(block_num+1+1,i-1+1) + V_1(block_num-1+1,i-1+1)) / (4 * h_1 * h_2);
            val_6 = H(block_num+1,i+1) * f2(block_num,i);
            
            b3(block_num * (M_2-1) + i) = val_5 + val_6 - val_1 - val_2 - val_3 - val_4;
        }
    }
    return b3;
}

mat print(mat matrix, vec vector, FILE* f, int ind, int k)
{
    for(unsigned long int i=0; i<M_1+1+static_cast<unsigned long int>(k-k);i++)
    {
        for(unsigned long int j=0; j<M_2+1;j++)
        {
            if (ind == 0)
                matrix(i, j) = vector(i * (M_2 + 1) + j);
//                 matrix(i, j) =  exp(static_cast<double>(k+1)*tau);
            else
            {
//                 double x = static_cast<double>(i)*h_1;
//                 double y = static_cast<double>(j)*h_2;
//                 double t = static_cast<double>(k+1)*tau;
                if(i==0 || j==0 || i==M_1 || j==M_2)
                    matrix(i, j) = 0.;
                else
                {
//                     if (ind == 1)
//                         matrix(i, j) = u_1(x,y,t);
//                     if (ind == 2)
//                         matrix(i, j) = u_2(x,y,t);
                        matrix(i, j) = vector((i - 1) * (M_2 - 1) + j - 1);
                }
            }
            fprintf(f, "%lf\n", matrix(i, j));
        }
    }
    return matrix;
}

void algorithm (mat H, mat V_1, mat V_2, mat V_1_temp, FILE* v_1_file, FILE* v_2_file, FILE* h_file)
{
    mat f0, f1, f2;
    mat A1, A2, A3;
    sp_mat C1, C2, C3;
    vec b1, b2, b3;
    vec sol1, sol2, sol3;
    double norm_V1, norm_V2, norm_H;
    int n = static_cast<int>(N);
    for(int k=0; k<n; k++)
    {
        A1 = create_A1(V_1, V_2);
        C1 = sp_mat(A1);
        f0 = create_f0(k);
        b1 = create_b1(f0, V_1, V_2, H);
        sol1 = spsolve(C1, b1);
        H = print(H, sol1, h_file, 0, k);
        
        A2 = create_A2(V_1, V_2, H);
        C2 = sp_mat(A2);
        f1 = create_f1(k);
        b2 = create_b2(f1, V_1, V_2, H);
        sol2 = spsolve(C2, b2);
        V_1_temp = print(V_1_temp, sol2, v_1_file, 1, k);
        
        A3 = create_A3(V_1, V_2, H);
        C3 = sp_mat(A3);
        f2 = create_f2(k);
        b3 = create_b3(f2, V_1, V_2, H);
        sol3 = spsolve(C3, b3);
        V_2 = print(V_2, sol3, v_2_file, 2, k);
        
        for(unsigned long int i=0; i<M_1+1; i++)
        {
            for(unsigned long int j=0; j<M_2+1;j++)
                V_1(i, j) = V_1_temp(i, j);
        }
        
        norm_V1 = norm(k+1, V_1, 1);
        norm_V2 = norm(k+1, V_2, 2);
        norm_H = norm(k+1, H, 3);
        printf("norm_V1 = %le (t = %.2lf)\n", norm_V1, static_cast<double>(k+1)*tau);
        printf("norm_V2 = %le (t = %.2lf)\n", norm_V2, static_cast<double>(k+1)*tau);
        printf("norm_H = %le (t = %.2lf)\n", norm_H, static_cast<double>(k+1)*tau);
    }
}


int main()
{
    mat H(static_cast<arma::uword>(M_1) + 1, static_cast<arma::uword>(M_2) + 1, fill::ones);
    mat V_1(static_cast<arma::uword>(M_1) + 1, static_cast<arma::uword>(M_2) + 1, fill::zeros);
    mat V_1_temp(static_cast<arma::uword>(M_1) + 1, static_cast<arma::uword>(M_2) + 1, fill::zeros);
    mat V_2(static_cast<arma::uword>(M_1) + 1, static_cast<arma::uword>(M_2) + 1, fill::zeros);
    FILE* v_1_file;
    FILE* v_2_file;
    FILE* h_file;
       
    v_1_file = fopen("V_1.txt", "w");
    v_2_file = fopen("V_2.txt", "w");
    h_file = fopen("H.txt", "w");
    
    for(long unsigned int i = 0; i < M_1 + 1; i++)
    {
        for(long unsigned int j=0; j < M_2 + 1; j++)
        {
            fprintf(h_file, "%lf\n", H(i, j));
            fprintf(v_1_file, "%lf\n", V_1(i, j));
            fprintf(v_2_file, "%lf\n", V_2(i, j));
        }
    }
    
    algorithm(H, V_1, V_2, V_1_temp, v_1_file, v_2_file, h_file);
        
    fclose(h_file);
    fclose(v_1_file);
    fclose(v_2_file);
    return 0;
}
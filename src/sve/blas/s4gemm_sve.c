#include <stddef.h>
#include <stdint.h>
#include<nnpack.h>
#include <nnpack/arm_neon.h>
#include <nnpack/macros.h>
#include <nnpack/hwinfo.h>

//complete VLA based
void nnp_s4gemm_only_3x3__sve(
        size_t k, size_t update,
        const float a[restrict static 1],
        const float b[restrict static 1],
        float c[restrict static 1],
        size_t row_stride_c)
{
//printf("%d", row_stride_c);
const  int simd_width = nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
int index_host[simd_width];

   for(int i1=0;i1<64;i1+=svcntw())
    {
                const float* a1 = &a[0];
             svbool_t pg = svwhilelt_b32(i1,64);
               svfloat32_t acc00 = svdup_f32(0.0f), acc10 = svdup_f32(0.0f), acc20 = svdup_f32(0.0f), acc30=svdup_f32(0.0f);
                svfloat32_t acc40 = svdup_f32(0.0f), acc50 = svdup_f32(0.0f), acc60 = svdup_f32(0.0f), acc70=svdup_f32(0.0f);
                svfloat32_t acc80 = svdup_f32(0.0f), acc90 = svdup_f32(0.0f), acc100 = svdup_f32(0.0f), acc110=svdup_f32(0.0f);
                svfloat32_t acc120 = svdup_f32(0.0f), acc130 = svdup_f32(0.0f), acc140 = svdup_f32(0.0f), acc150=svdup_f32(0.0f);
                svfloat32_t acc01 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc31=svdup_f32(0.0f);
                svfloat32_t acc02 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc32=svdup_f32(0.0f);
                svfloat32_t acc03 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f), acc33=svdup_f32(0.0f);
                svfloat32_t acc41 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc71=svdup_f32(0.0f);
                svfloat32_t acc42 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc72=svdup_f32(0.0f);
                svfloat32_t acc43 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f), acc73=svdup_f32(0.0f);

            for(int j=0;j<k;j++){
                svfloat32_t b0 = svld1( pg, &b[i1+(j*64)]);
                const svfloat32_t a0 = svld1rq(pg, &a[0+(j*64)]);
                acc00 = svmla_m(pg, acc00, a0, b0);
                const svfloat32_t a1 = svld1rq(pg, &a[4+(j*64)]);
                acc10 = svmla_m(pg, acc10, a1, b0);
                const svfloat32_t a2 = svld1rq(pg, &a[8+(j*64)]);
                acc20 = svmla_m(pg, acc20, a2, b0);
                const svfloat32_t a3 = svld1rq(pg, &a[12+(j*64)]);
                acc30 = svmla_m(pg, acc30, a3, b0);
                const svfloat32_t a4 = svld1rq(pg, &a[16+(j*64)]);
                acc40 = svmla_m(pg, acc40, a4, b0);
                const svfloat32_t a5 = svld1rq(pg, &a[20+(j*64)]);
                acc50 = svmla_m(pg, acc50, a5, b0);
                const svfloat32_t a6 = svld1rq(pg, &a[24+(j*64)]);
                acc60 = svmla_m(pg, acc60, a6, b0);
                const svfloat32_t a7 = svld1rq(pg, &a[28+(j*64)]);
                acc70 = svmla_m(pg, acc70, a7, b0);
                const svfloat32_t a8 = svld1rq(pg, &a[32+(j*64)]);
                acc80 = svmla_m(pg, acc80, a8, b0);
                const svfloat32_t a9 = svld1rq(pg, &a[36+(j*64)]);
                acc90 = svmla_m(pg, acc90, a9, b0);
                const svfloat32_t a10 = svld1rq(pg, &a[40+(j*64)]);
                acc100 = svmla_m(pg, acc100, a10, b0);
                const svfloat32_t a11 = svld1rq(pg,&a[44+(j*64)]);
                acc110 = svmla_m(pg, acc110, a11, b0);
                const svfloat32_t a12 = svld1rq(pg, &a[48+(j*64)]);
                acc120 = svmla_m(pg, acc120, a12, b0);
                const svfloat32_t a13 = svld1rq(pg, &a[52+(j*64)]);
                acc130 = svmla_m(pg, acc130, a13, b0);
                const svfloat32_t a14 = svld1rq(pg, &a[56+(j*64)]);
                acc140 = svmla_m(pg, acc140, a14, b0);
                const svfloat32_t a15 = svld1rq(pg, &a[60+(j*64)]);
                acc150 = svmla_m(pg, acc150, a15, b0);


               // a1+=32;
                }
                if (update != 0)
                {       
                        svst1(pg, &c[i1],  svadd_m(pg, svld1(pg, &c[i1]), acc00));
                        svst1(pg, &c[i1+row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+row_stride_c]), acc10));
                        svst1(pg, &c[i1+2*row_stride_c], svadd_m(pg, svld1(pg, &c[i1+2*row_stride_c]), acc20));
                        svst1(pg, &c[i1+3*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+3*row_stride_c]), acc30));
                        svst1(pg, &c[i1+4*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+4*row_stride_c]), acc40));
                        svst1(pg, &c[i1+5*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+5*row_stride_c]), acc50));
                        svst1(pg, &c[i1+6*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+6*row_stride_c]), acc60));
                        svst1(pg, &c[i1+7*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+7*row_stride_c]), acc70));
                        svst1(pg, &c[i1+8*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+8*row_stride_c]), acc80));
                        svst1(pg, &c[i1+9*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+9*row_stride_c]), acc90));
                        svst1(pg, &c[i1+10*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+10*row_stride_c]), acc100));
                        svst1(pg, &c[i1+11*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+11*row_stride_c]), acc110));
                        svst1(pg, &c[i1+12*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+12*row_stride_c]), acc120));
                        svst1(pg, &c[i1+13*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+13*row_stride_c]), acc130));
                        svst1(pg, &c[i1+14*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+14*row_stride_c]), acc140));
                        svst1(pg, &c[i1+15*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+15*row_stride_c]), acc150));
                
                }
               else
                {
                        svst1(pg, &c[i1], acc00);
                        svst1(pg, &c[i1+row_stride_c],  acc10);
                        svst1(pg, &c[i1+2*row_stride_c],  acc20);
                        svst1(pg, &c[i1+3*row_stride_c],  acc30);
                        svst1(pg, &c[i1+4*row_stride_c],  acc40);
                        svst1(pg, &c[i1+5*row_stride_c],  acc50);
                        svst1(pg, &c[i1+6*row_stride_c],  acc60);
                        svst1(pg, &c[i1+7*row_stride_c],  acc70);
                        svst1(pg, &c[i1+8*row_stride_c],  acc80);
                        svst1(pg, &c[i1+9*row_stride_c],  acc90);
                        svst1(pg, &c[i1+10*row_stride_c],  acc100);
                        svst1(pg, &c[i1+11*row_stride_c],  acc110);
                        svst1(pg, &c[i1+12*row_stride_c],  acc120);
                        svst1(pg, &c[i1+13*row_stride_c],  acc130);
                        svst1(pg, &c[i1+14*row_stride_c],  acc140);
                        svst1(pg, &c[i1+15*row_stride_c],  acc150);
                }
        }
}
/*void nnp_s4gemm_only_3x3__neon(
        size_t k, size_t update,
        const float a[restrict static 1],
        const float b[restrict static 1],
        float c[restrict static 1],
        size_t row_stride_c)
{
const  int simd_width = nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
int index_host[simd_width];
int vl = svcntw();
 int rem =  64 / svcntw();
if(rem == 1)
{
	for(int i1=0;i1<64;i1+=(svcntw()))
	{
		svbool_t pg = svwhilelt_b32(i1,64);
               svfloat32_t acc00 = svdup_f32(0.0f), acc10 = svdup_f32(0.0f), acc20 = svdup_f32(0.0f), acc30=svdup_f32(0.0f);
                svfloat32_t acc40 = svdup_f32(0.0f), acc50 = svdup_f32(0.0f), acc60 = svdup_f32(0.0f), acc70=svdup_f32(0.0f);
                svfloat32_t acc80 = svdup_f32(0.0f), acc90 = svdup_f32(0.0f), acc100 = svdup_f32(0.0f), acc110=svdup_f32(0.0f);
                svfloat32_t acc120 = svdup_f32(0.0f), acc130 = svdup_f32(0.0f), acc140 = svdup_f32(0.0f), acc150=svdup_f32(0.0f);
                svfloat32_t acc01 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc31=svdup_f32(0.0f);
                svfloat32_t acc02 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc32=svdup_f32(0.0f);
                svfloat32_t acc03 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f), acc33=svdup_f32(0.0f);
                svfloat32_t acc41 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc71=svdup_f32(0.0f);
                svfloat32_t acc42 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc72=svdup_f32(0.0f);
                svfloat32_t acc43 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f), acc73=svdup_f32(0.0f);

                do{
                          //   printf("I am going in sve");
                svbool_t pg = svwhilelt_b32(i1,simd_width);
                const svfloat32_t a0 = svld1rq(pg, a + 0);
        //      const svfloat32_t a0 = svld1(pg, a + 0);
                const svfloat32_t b0 = svld1(pg, b +  0);
                acc00 = svmla_m(pg, acc00, a0, b0);
                const svfloat32_t a1 = svld1rq(pg, a + 4);
                acc10 = svmla_m(pg, acc10, a1, b0);
                const svfloat32_t a2 = svld1rq(pg, a + 8);
                acc20 = svmla_m(pg, acc20, a2, b0);
                const svfloat32_t a3 = svld1rq(pg, a + 12);
                acc30 = svmla_m(pg, acc30, a3, b0);
                const svfloat32_t a4 = svld1rq(pg, a + 16);
                acc40 = svmla_m(pg, acc40, a4, b0);
               const svfloat32_t a5 = svld1rq(pg, a + 20);
                acc50 = svmla_m(pg, acc50, a5, b0);
                const svfloat32_t a6 = svld1rq(pg, a + 24);
                acc60 = svmla_m(pg, acc60, a6, b0);
                const svfloat32_t a7 = svld1rq(pg, a + 28);
                acc70 = svmla_m(pg, acc70, a7, b0);
                const svfloat32_t a8 = svld1rq(pg, a +  32);
                acc80 = svmla_m(pg, acc80, a8, b0);
                const svfloat32_t a9 = svld1rq(pg, a + 36);
                acc90 = svmla_m(pg, acc90, a9, b0);
                const svfloat32_t a10 = svld1rq(pg, a + 40);
                acc100 = svmla_m(pg, acc100, a10, b0);
                const svfloat32_t a11 = svld1rq(pg, a + 44);
                acc110 = svmla_m(pg, acc110, a11, b0);
                const svfloat32_t a12 = svld1rq(pg, a + 48);
                acc120 = svmla_m(pg, acc120, a12, b0);
                const svfloat32_t a13 = svld1rq(pg, a + 52);
                acc130 = svmla_m(pg, acc130, a13, b0);
                const svfloat32_t a14 = svld1rq(pg, a + 56);
                acc140 = svmla_m(pg, acc140, a14, b0);
                const svfloat32_t a15 = svld1rq(pg, a + 60);
                acc150 = svmla_m(pg, acc150, a15, b0);

                a+=64;
                b+=64;
                }while (--k);
                 if (update != 0)
                {
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc00));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc10));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc20));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc30));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc40));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc50));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc60));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc70));

                             c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc80));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc90));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc100));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc110));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc120));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc130));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc140));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc150));
                }
                else
                {
                        svst1(pg, c+0, acc00);
                        c += row_stride_c;
                        svst1(pg, c+0, acc10);
                        c += row_stride_c;
                        svst1(pg, c+0, acc20);
                        c += row_stride_c;
                        svst1(pg, c+0, acc30);
                        c += row_stride_c;
                        svst1(pg, c+0, acc40);
                        c += row_stride_c;
                        svst1(pg, c+0, acc50);
                        c += row_stride_c;
                        svst1(pg, c+0, acc60);
                        c += row_stride_c;
                        svst1(pg, c+0, acc70);
                         c += row_stride_c;
                        svst1(pg, c+0, acc80);
                        c += row_stride_c;
                        svst1(pg, c+0, acc90);
			                        c += row_stride_c;
                        svst1(pg, c+0, acc100);
                        c += row_stride_c;
                        svst1(pg, c+0, acc110);
                        c += row_stride_c;
                        svst1(pg, c+0, acc120);
                        c += row_stride_c;
                        svst1(pg, c+0, acc130);
                        c += row_stride_c;
                        svst1(pg, c+0, acc140);
                        c += row_stride_c;
                        svst1(pg, c+0, acc150);

                }
	}
}
else
{
   for(int i1=0;i1<64;i1+=(2*svcntw()))
    {
                const float* a1 = &a[0];
             svbool_t pg = svwhilelt_b32(i1,64);
               svfloat32_t acc00 = svdup_f32(0.0f), acc10 = svdup_f32(0.0f), acc20 = svdup_f32(0.0f), acc30=svdup_f32(0.0f);
                svfloat32_t acc40 = svdup_f32(0.0f), acc50 = svdup_f32(0.0f), acc60 = svdup_f32(0.0f), acc70=svdup_f32(0.0f);
                svfloat32_t acc80 = svdup_f32(0.0f), acc90 = svdup_f32(0.0f), acc100 = svdup_f32(0.0f), acc110=svdup_f32(0.0f);
                svfloat32_t acc81 = svdup_f32(0.0f), acc91 = svdup_f32(0.0f), acc101 = svdup_f32(0.0f), acc111=svdup_f32(0.0f);
                svfloat32_t acc120 = svdup_f32(0.0f), acc130 = svdup_f32(0.0f), acc140 = svdup_f32(0.0f), acc150=svdup_f32(0.0f);
                svfloat32_t acc01 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc31=svdup_f32(0.0f);
                svfloat32_t acc121 = svdup_f32(0.0f), acc131 = svdup_f32(0.0f), acc141 = svdup_f32(0.0f), acc151=svdup_f32(0.0f);
                svfloat32_t acc02 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc32=svdup_f32(0.0f);
                svfloat32_t acc03 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f), acc33=svdup_f32(0.0f);
                svfloat32_t acc41 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc71=svdup_f32(0.0f);
                svfloat32_t acc42 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc72=svdup_f32(0.0f);
                svfloat32_t acc43 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f), acc73=svdup_f32(0.0f);
            for(int j=0;j<k;j++){
                svfloat32_t b0 = svld1( pg, &b[i1+(j*64)]);
                svfloat32_t b1 = svld1( pg, &b[i1+vl+(j*64)]);
                const svfloat32_t a0 = svld1rq(pg, &a[0+(j*64)]);
                acc00 = svmla_m(pg, acc00, a0, b0);
                acc01 = svmla_m(pg, acc01, a0, b1);
                const svfloat32_t a1 = svld1rq(pg, &a[4+(j*64)]);
                acc10 = svmla_m(pg, acc10, a1, b0);
                acc11 = svmla_m(pg, acc11, a1, b1);
                const svfloat32_t a2 = svld1rq(pg, &a[8+(j*64)]);
                acc20 = svmla_m(pg, acc20, a2, b0);
                acc21 = svmla_m(pg, acc21, a2, b1);
                const svfloat32_t a3 = svld1rq(pg, &a[12+(j*64)]);
                acc30 = svmla_m(pg, acc30, a3, b0);
                acc31 = svmla_m(pg, acc31, a3, b1);
                const svfloat32_t a4 = svld1rq(pg, &a[16+(j*64)]);
                acc40 = svmla_m(pg, acc40, a4, b0);
                acc41 = svmla_m(pg, acc41, a4, b1);
                const svfloat32_t a5 = svld1rq(pg, &a[20+(j*64)]);
                acc50 = svmla_m(pg, acc50, a5, b0);
                acc51 = svmla_m(pg, acc51, a5, b1);
                const svfloat32_t a6 = svld1rq(pg, &a[24+(j*64)]);
                acc60 = svmla_m(pg, acc60, a6, b0);
                acc61 = svmla_m(pg, acc61, a6, b1);
                const svfloat32_t a7 = svld1rq(pg, &a[28+(j*64)]);
                acc70 = svmla_m(pg, acc70, a7, b0);
                acc71 = svmla_m(pg, acc71, a7, b1);
		const svfloat32_t a8 = svld1rq(pg, &a[32+(j*64)]);
                acc80 = svmla_m(pg, acc80, a8, b0);
                acc81 = svmla_m(pg, acc81, a8, b1);
                const svfloat32_t a9 = svld1rq(pg, &a[36+(j*64)]);
                acc90 = svmla_m(pg, acc90, a9, b0);
                acc91 = svmla_m(pg, acc91, a9, b1);
                const svfloat32_t a10 = svld1rq(pg, &a[40+(j*64)]);
                acc100 = svmla_m(pg, acc100, a10, b0);
                acc101 = svmla_m(pg, acc101, a10, b1);
                const svfloat32_t a11 = svld1rq(pg,&a[44+(j*64)]);
                acc110 = svmla_m(pg, acc110, a11, b0);
                acc111 = svmla_m(pg, acc111, a11, b1);
                const svfloat32_t a12 = svld1rq(pg, &a[48+(j*64)]);
                acc120 = svmla_m(pg, acc120, a12, b0);
                acc121 = svmla_m(pg, acc121, a12, b1);
                const svfloat32_t a13 = svld1rq(pg, &a[52+(j*64)]);
                acc130 = svmla_m(pg, acc130, a13, b0);
                acc131 = svmla_m(pg, acc131, a13, b1);
                const svfloat32_t a14 = svld1rq(pg, &a[56+(j*64)]);
                acc140 = svmla_m(pg, acc140, a14, b0);
                acc141 = svmla_m(pg, acc141, a14, b1);
                const svfloat32_t a15 = svld1rq(pg, &a[60+(j*64)]);
                acc150 = svmla_m(pg, acc150, a15, b0);
                acc151 = svmla_m(pg, acc151, a15, b1);
		

               // a1+=32;
                }

                if (update != 0)
                {
                        svst1(pg, &c[i1],  svadd_m(pg, svld1(pg, &c[i1]), acc00));
                        svst1(pg, &c[i1+row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+row_stride_c]), acc10));
                        svst1(pg, &c[i1+2*row_stride_c], svadd_m(pg, svld1(pg, &c[i1+2*row_stride_c]), acc20));
                        svst1(pg, &c[i1+3*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+3*row_stride_c]), acc30));
                        svst1(pg, &c[i1+4*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+4*row_stride_c]), acc40));
                        svst1(pg, &c[i1+5*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+5*row_stride_c]), acc50));
                        svst1(pg, &c[i1+6*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+6*row_stride_c]), acc60));
                        svst1(pg, &c[i1+7*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+7*row_stride_c]), acc70));
                        svst1(pg, &c[i1+8*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+8*row_stride_c]), acc80));
                        svst1(pg, &c[i1+9*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+9*row_stride_c]), acc90));
                        svst1(pg, &c[i1+10*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+10*row_stride_c]), acc100));
                        svst1(pg, &c[i1+11*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+11*row_stride_c]), acc110));
                        svst1(pg, &c[i1+12*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+12*row_stride_c]), acc120));
                        svst1(pg, &c[i1+13*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+13*row_stride_c]), acc130));
                        svst1(pg, &c[i1+14*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+14*row_stride_c]), acc140));
                        svst1(pg, &c[i1+15*row_stride_c],  svadd_m(pg, svld1(pg, &c[i1+15*row_stride_c]), acc150));
                        svst1(pg, &c[i1+vl],  svadd_m(pg, svld1(pg, &c[i1+vl]), acc01));
                        svst1(pg, &c[i1+row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+row_stride_c+vl]), acc11));
                        svst1(pg, &c[i1+2*row_stride_c+vl], svadd_m(pg, svld1(pg, &c[i1+2*row_stride_c+vl]), acc21));
                        svst1(pg, &c[i1+3*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+3*row_stride_c+vl]), acc31));
                        svst1(pg, &c[i1+4*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+4*row_stride_c+vl]), acc41));
                        svst1(pg, &c[i1+5*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+5*row_stride_c+vl]), acc51));
                        svst1(pg, &c[i1+6*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+6*row_stride_c+vl]), acc61));
                        svst1(pg, &c[i1+7*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+7*row_stride_c+vl]), acc71));
                        svst1(pg, &c[i1+8*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+8*row_stride_c+vl]), acc81));
                        svst1(pg, &c[i1+9*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+9*row_stride_c+vl]), acc91));
                        svst1(pg, &c[i1+10*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+10*row_stride_c+vl]), acc101));
                        svst1(pg, &c[i1+11*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+11*row_stride_c+vl]), acc111));
                        svst1(pg, &c[i1+12*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+12*row_stride_c+vl]), acc121));
                        svst1(pg, &c[i1+13*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+13*row_stride_c+vl]), acc131));
                        svst1(pg, &c[i1+14*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+14*row_stride_c+vl]), acc141));
                        svst1(pg, &c[i1+15*row_stride_c+vl],  svadd_m(pg, svld1(pg, &c[i1+15*row_stride_c+vl]), acc151));

                }
               else
                {
                        svst1(pg, &c[i1], acc00);
                        svst1(pg, &c[i1+row_stride_c],  acc10);
                        svst1(pg, &c[i1+2*row_stride_c],  acc20);
                        svst1(pg, &c[i1+3*row_stride_c],  acc30);
                        svst1(pg, &c[i1+4*row_stride_c],  acc40);
                        svst1(pg, &c[i1+5*row_stride_c],  acc50);
                        svst1(pg, &c[i1+6*row_stride_c],  acc60);
                        svst1(pg, &c[i1+7*row_stride_c],  acc70);
                        svst1(pg, &c[i1+8*row_stride_c],  acc80);
                        svst1(pg, &c[i1+9*row_stride_c],  acc90);
                        svst1(pg, &c[i1+10*row_stride_c],  acc100);
                        svst1(pg, &c[i1+11*row_stride_c],  acc110);
                        svst1(pg, &c[i1+12*row_stride_c],  acc120);
                        svst1(pg, &c[i1+13*row_stride_c],  acc130);
                        svst1(pg, &c[i1+14*row_stride_c],  acc140);
                        svst1(pg, &c[i1+15*row_stride_c],  acc150);
                        svst1(pg, &c[i1+vl], acc01);
                        svst1(pg, &c[i1+row_stride_c+vl],  acc11);
                        svst1(pg, &c[i1+2*row_stride_c+vl],  acc21);
                        svst1(pg, &c[i1+3*row_stride_c+vl],  acc31);
                        svst1(pg, &c[i1+4*row_stride_c+vl],  acc41);
                        svst1(pg, &c[i1+5*row_stride_c+vl],  acc51);
                        svst1(pg, &c[i1+6*row_stride_c+vl],  acc61);
                        svst1(pg, &c[i1+7*row_stride_c+vl],  acc71);
                        svst1(pg, &c[i1+8*row_stride_c+vl],  acc81);
                        svst1(pg, &c[i1+9*row_stride_c+vl],  acc91);
                        svst1(pg, &c[i1+10*row_stride_c+vl],  acc101);
                        svst1(pg, &c[i1+11*row_stride_c+vl],  acc111);
                        svst1(pg, &c[i1+12*row_stride_c+vl],  acc121);
                        svst1(pg, &c[i1+13*row_stride_c+vl],  acc131);
                        svst1(pg, &c[i1+14*row_stride_c+vl],  acc141);
                        svst1(pg, &c[i1+15*row_stride_c+vl],  acc151);
                }
        }
	}
}*/
/*
void nnp_s4gemm_only_3x3__neon(
        size_t k, size_t update,
        const float a[restrict static 1],
        const float b[restrict static 1],
        float c[restrict static 1],
        size_t row_stride_c)
{
const  int simd_width = nnp_hwinfo.sve_simd_width;//nnp_hwinfo.simd_width;
int index_host[simd_width];

    int rem =  64 / svcntw();
   for(int i1=0;i1<64;i1+= (rem*svcntw()))
    {

        svbool_t pg = svwhilelt_b32(i1,64);
        if(rem == 2)
        {
                 svfloat32_t acc00 = svdup_f32(0.0f), acc10 = svdup_f32(0.0f), acc20 = svdup_f32(0.0f), acc30=svdup_f32(0.0f);
                svfloat32_t acc40 = svdup_f32(0.0f), acc50 = svdup_f32(0.0f), acc60 = svdup_f32(0.0f), acc70=svdup_f32(0.0f);
                svfloat32_t acc80 = svdup_f32(0.0f), acc90 = svdup_f32(0.0f), acc100 = svdup_f32(0.0f), acc110=svdup_f32(0.0f);
                svfloat32_t acc81 = svdup_f32(0.0f), acc91 = svdup_f32(0.0f), acc101 = svdup_f32(0.0f), acc111=svdup_f32(0.0f);
                svfloat32_t acc120 = svdup_f32(0.0f), acc130 = svdup_f32(0.0f), acc140 = svdup_f32(0.0f), acc150=svdup_f32(0.0f);
                svfloat32_t acc121 = svdup_f32(0.0f), acc131 = svdup_f32(0.0f), acc141 = svdup_f32(0.0f), acc151=svdup_f32(0.0f);
                svfloat32_t acc01 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc31=svdup_f32(0.0f);
                svfloat32_t acc02 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc32=svdup_f32(0.0f);
                svfloat32_t acc03 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f), acc33=svdup_f32(0.0f);
                svfloat32_t acc41 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc71=svdup_f32(0.0f);
                svfloat32_t acc42 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc72=svdup_f32(0.0f);
                svfloat32_t acc43 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f), acc73=svdup_f32(0.0f);

                do{
                const svfloat32_t a0 = svld1rq(pg, a + 0);
        //      const svfloat32_t a0 = svld1(pg, a + 0);
                const svfloat32_t b0 = svld1(pg, b +  0);
                const svfloat32_t b1 = svld1(pg, b +  32);
                acc00 = svmla_m(pg, acc00, a0, b0);
                acc01 = svmla_m(pg, acc01, a0, b1);
                const svfloat32_t a1 = svld1rq(pg, a + 4);
                acc10 = svmla_m(pg, acc10, a1, b0);
                acc11 = svmla_m(pg, acc11, a1, b1);
                const svfloat32_t a2 = svld1rq(pg, a + 8);
                acc20 = svmla_m(pg, acc20, a2, b0);
                acc21 = svmla_m(pg, acc21, a2, b1);
                const svfloat32_t a3 = svld1rq(pg, a + 12);
                acc30 = svmla_m(pg, acc30, a3, b0);
                acc31 = svmla_m(pg, acc31, a3, b1);
                const svfloat32_t a4 = svld1rq(pg, a + 16);
                acc40 = svmla_m(pg, acc40, a4, b0);
                acc41= svmla_m(pg, acc41, a4, b1);
                const svfloat32_t a5 = svld1rq(pg, a + 20);
                acc50 = svmla_m(pg, acc50, a5, b0);
                acc51 = svmla_m(pg, acc51, a5, b1);
                const svfloat32_t a6 = svld1rq(pg, a + 24);
                acc60 = svmla_m(pg, acc60, a6, b0);
                acc61 = svmla_m(pg, acc61, a6, b1);
                const svfloat32_t a7 = svld1rq(pg, a + 28);
                acc70 = svmla_m(pg, acc70, a7, b0);
                acc71 = svmla_m(pg, acc71, a7, b1);
                const svfloat32_t a8 = svld1rq(pg, a + 32);
                acc80 = svmla_m(pg, acc80, a8, b0);
                acc81 = svmla_m(pg, acc81, a8, b1);
                const svfloat32_t a9 = svld1rq(pg, a + 36);
                acc90 = svmla_m(pg, acc90, a9, b0);
                acc91 = svmla_m(pg, acc91, a9, b1);
                const svfloat32_t a10 = svld1rq(pg, a + 40);
                acc100 = svmla_m(pg, acc100, a10, b0);
                acc101 = svmla_m(pg, acc101, a10, b1);
                const svfloat32_t a11 = svld1rq(pg, a + 44);
                acc110 = svmla_m(pg, acc110, a11, b0);
                acc111 = svmla_m(pg, acc111, a11, b1);
                const svfloat32_t a12 = svld1rq(pg, a + 48);
                acc120 = svmla_m(pg, acc120, a12, b0);
                acc121 = svmla_m(pg, acc121, a12, b1);
                const svfloat32_t a13 = svld1rq(pg, a + 52);
                acc130 = svmla_m(pg, acc130, a13, b0);
                acc131 = svmla_m(pg, acc131, a13, b1);
                const svfloat32_t a14 = svld1rq(pg, a + 56);
                acc140 = svmla_m(pg, acc140, a14, b0);
                acc141 = svmla_m(pg, acc141, a14, b1);
                const svfloat32_t a15 = svld1rq(pg, a + 60);
                acc150 = svmla_m(pg, acc150, a15, b0);
                acc151 = svmla_m(pg, acc151, a15, b1);

                  a+=64;
                b+=64;
                }while (--k);
                if (update != 0)
                {
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc00));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc01));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc10));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc11));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc20));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc21));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc30));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc31));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc40));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc41));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc50));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc51));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc60));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc61));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc70));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc71));

                             c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc80));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc81));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc90));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc91));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc100));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc101));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc110));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc111));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc120));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc121));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc130));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc131));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc140));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc141));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc150));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc151));

                }
                else
                {
                        svst1(pg, c+0, acc00);
                        svst1(pg, c+32, acc01);
                        c += row_stride_c;
                        svst1(pg, c+0, acc10);
                        svst1(pg, c+32, acc11);
                        c += row_stride_c;
                        svst1(pg, c+0, acc20);
                        svst1(pg, c+32, acc21);
                        c += row_stride_c;
                        svst1(pg, c+0, acc30);
                        svst1(pg, c+32, acc31);
                        c += row_stride_c;
                        svst1(pg, c+0, acc40);
                        svst1(pg, c+32, acc41);
                        c += row_stride_c;
                        svst1(pg, c+0, acc50);
                        svst1(pg, c+32, acc51);
                        c += row_stride_c;
                        svst1(pg, c+0, acc60);
                        svst1(pg, c+32, acc61);
                        c += row_stride_c;
                        svst1(pg, c+0, acc70);
                        svst1(pg, c+32, acc71);
                        c += row_stride_c;

                        svst1(pg, c+0, acc80);
                        svst1(pg, c+32, acc81);
                        c += row_stride_c;
                        svst1(pg, c+0, acc90);
                        svst1(pg, c+32, acc91);
                        c += row_stride_c;
                        svst1(pg, c+0, acc100);
                        svst1(pg, c+32, acc101);
                        c += row_stride_c;
                        svst1(pg, c+0, acc110);
                        svst1(pg, c+32, acc111);
                        c += row_stride_c;
                        svst1(pg, c+0, acc120);
                        svst1(pg, c+32, acc121);
                        c += row_stride_c;
                        svst1(pg, c+0, acc130);
                        svst1(pg, c+32, acc131);
                        c += row_stride_c;
                        svst1(pg, c+0, acc140);
                        svst1(pg, c+32, acc141);
                        c += row_stride_c;
                        svst1(pg, c+0, acc150);
                        svst1(pg, c+32, acc151);
                }
        }
        else if(rem == 4)
        {
                 svfloat32_t acc00 = svdup_f32(0.0f), acc10 = svdup_f32(0.0f), acc20 = svdup_f32(0.0f), acc30=svdup_f32(0.0f);
                svfloat32_t acc40 = svdup_f32(0.0f), acc50 = svdup_f32(0.0f), acc60 = svdup_f32(0.0f), acc70=svdup_f32(0.0f);
                svfloat32_t acc80 = svdup_f32(0.0f), acc90 = svdup_f32(0.0f), acc100 = svdup_f32(0.0f), acc110=svdup_f32(0.0f);
                svfloat32_t acc81 = svdup_f32(0.0f), acc91 = svdup_f32(0.0f), acc101 = svdup_f32(0.0f), acc111=svdup_f32(0.0f);
                svfloat32_t acc120 = svdup_f32(0.0f), acc130 = svdup_f32(0.0f), acc140 = svdup_f32(0.0f), acc150=svdup_f32(0.0f);
                svfloat32_t acc121 = svdup_f32(0.0f), acc131 = svdup_f32(0.0f), acc141 = svdup_f32(0.0f), acc151=svdup_f32(0.0f);
                svfloat32_t acc01 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc31=svdup_f32(0.0f);
                svfloat32_t acc02 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc32=svdup_f32(0.0f);
                svfloat32_t acc03 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f), acc33=svdup_f32(0.0f);
                svfloat32_t acc41 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc71=svdup_f32(0.0f);
                svfloat32_t acc42 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc72=svdup_f32(0.0f);
                svfloat32_t acc43 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f), acc73=svdup_f32(0.0f);
                svfloat32_t acc82 = svdup_f32(0.0f), acc92 = svdup_f32(0.0f), acc102 = svdup_f32(0.0f), acc112=svdup_f32(0.0f);
                svfloat32_t acc83 = svdup_f32(0.0f), acc93 = svdup_f32(0.0f), acc103 = svdup_f32(0.0f), acc113=svdup_f32(0.0f);
                svfloat32_t acc122 = svdup_f32(0.0f), acc132 = svdup_f32(0.0f), acc142 = svdup_f32(0.0f), acc152=svdup_f32(0.0f);
                svfloat32_t acc123 = svdup_f32(0.0f), acc133 = svdup_f32(0.0f), acc143 = svdup_f32(0.0f), acc153=svdup_f32(0.0f);

                do{
                const svfloat32_t a0 = svld1rq(pg, a + 0);
        //      const svfloat32_t a0 = svld1(pg, a + 0);
                const svfloat32_t b0 = svld1(pg, b +  0);
                const svfloat32_t b1 = svld1(pg, b +  16);
                const svfloat32_t b2 = svld1(pg, b + 32);
                const svfloat32_t b3 = svld1(pg, b +  48);
                acc00 = svmla_m(pg, acc00, a0, b0);
                acc01 = svmla_m(pg, acc01, a0, b1);
                acc02 = svmla_m(pg, acc02, a0, b2);
                acc03 = svmla_m(pg, acc03, a0, b3);
                const svfloat32_t a1 = svld1rq(pg, a + 4);
                acc10 = svmla_m(pg, acc10, a1, b0);
                acc11 = svmla_m(pg, acc11, a1, b1);
                acc12 = svmla_m(pg, acc12, a1, b2);
                acc13 = svmla_m(pg, acc13, a1, b3);
                const svfloat32_t a2 = svld1rq(pg, a + 8);
                acc20 = svmla_m(pg, acc20, a2, b0);
                acc21 = svmla_m(pg, acc21, a2, b1);
                acc22 = svmla_m(pg, acc22, a2, b2);
                acc23 = svmla_m(pg, acc23, a2, b3);
                const svfloat32_t a3 = svld1rq(pg, a + 12);
                acc30 = svmla_m(pg, acc30, a3, b0);
                acc31 = svmla_m(pg, acc31, a3, b1);
                acc32 = svmla_m(pg, acc32, a3, b2);
                acc33 = svmla_m(pg, acc33, a3, b3);
                const svfloat32_t a4 = svld1rq(pg, a + 16);
                acc40 = svmla_m(pg, acc40, a4, b0);
                acc41= svmla_m(pg, acc41, a4, b1);
                acc42 = svmla_m(pg, acc42, a4, b2);
                acc43= svmla_m(pg, acc43, a4, b3);
                const svfloat32_t a5 = svld1rq(pg, a + 20);
                acc50 = svmla_m(pg, acc50, a5, b0);
                acc51 = svmla_m(pg, acc51, a5, b1);
                acc52 = svmla_m(pg, acc52, a5, b2);
                acc53 = svmla_m(pg, acc53, a5, b3);
                const svfloat32_t a6 = svld1rq(pg, a + 24);
                acc60 = svmla_m(pg, acc60, a6, b0);
                acc61 = svmla_m(pg, acc61, a6, b1);
                acc62 = svmla_m(pg, acc62, a6, b2);
                acc63 = svmla_m(pg, acc63, a6, b3);
                const svfloat32_t a7 = svld1rq(pg, a + 28);
                acc70 = svmla_m(pg, acc70, a7, b0);
                acc71 = svmla_m(pg, acc71, a7, b1);
                acc72 = svmla_m(pg, acc72, a7, b2);
                acc73 = svmla_m(pg, acc73, a7, b3);
                const svfloat32_t a8 = svld1rq(pg, a + 32);
                acc80 = svmla_m(pg, acc80, a8, b0);
                acc81 = svmla_m(pg, acc81, a8, b1);
                acc82 = svmla_m(pg, acc82, a8, b2);
                acc83 = svmla_m(pg, acc83, a8, b3);
                const svfloat32_t a9 = svld1rq(pg, a + 36);
                acc90 = svmla_m(pg, acc90, a9, b0);
                acc91 = svmla_m(pg, acc91, a9, b1);
                acc92 = svmla_m(pg, acc92, a9, b2);
                acc93 = svmla_m(pg, acc93, a9, b3);
                const svfloat32_t a10 = svld1rq(pg, a + 40);
                acc100 = svmla_m(pg, acc100, a10, b0);
                acc101 = svmla_m(pg, acc101, a10, b1);
                acc102 = svmla_m(pg, acc102, a10, b2);
                acc103 = svmla_m(pg, acc103, a10, b3);
                const svfloat32_t a11 = svld1rq(pg, a + 44);
                acc110 = svmla_m(pg, acc110, a11, b0);
                acc111 = svmla_m(pg, acc111, a11, b1);
                acc112 = svmla_m(pg, acc112, a11, b2);
                acc113 = svmla_m(pg, acc113, a11, b3);
                const svfloat32_t a12 = svld1rq(pg, a + 48);
                acc120 = svmla_m(pg, acc120, a12, b0);
                acc121 = svmla_m(pg, acc121, a12, b1);
                acc122 = svmla_m(pg, acc122, a12, b2);
                acc123 = svmla_m(pg, acc123, a12, b3);
                const svfloat32_t a13 = svld1rq(pg, a + 52);
                acc130 = svmla_m(pg, acc130, a13, b0);
                acc131 = svmla_m(pg, acc131, a13, b1);
                acc132 = svmla_m(pg, acc132, a13, b2);
                acc133 = svmla_m(pg, acc133, a13, b3);
                const svfloat32_t a14 = svld1rq(pg, a + 56);
                acc140 = svmla_m(pg, acc140, a14, b0);
                acc141 = svmla_m(pg, acc141, a14, b1);
                acc142 = svmla_m(pg, acc142, a14, b2);
                acc143 = svmla_m(pg, acc143, a14, b3);
                const svfloat32_t a15 = svld1rq(pg, a + 60);
                acc150 = svmla_m(pg, acc150, a15, b0);
                acc151 = svmla_m(pg, acc151, a15, b1);
                acc152 = svmla_m(pg, acc152, a15, b2);
                acc153 = svmla_m(pg, acc153, a15, b3);

                  a+=64;
                b+=64;
                }while (--k);
                if (update != 0)
                {
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc00));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc01));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc02));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc03));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc10));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc11));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc12));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc13));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc20));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc21));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc22));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc23));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc30));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc31));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc32));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc33));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc40));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc41));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc42));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc43));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc50));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc51));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc52));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc53));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc60));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc61));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc62));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc63));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc70));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc71));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc72));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc73));

                             c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc80));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc81));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc82));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc83));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc90));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc91));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc92));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc93));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc100));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc101));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc102));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc103));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc110));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc111));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc112));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc113));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc120));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc121));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc122));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc123));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc130));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc131));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc132));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc133));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc140));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc141));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc142));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc143));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc150));
                        svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc151));
                        svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc152));
                        svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc153));

                }
                else
                {
                        svst1(pg, c+0, acc00);
                        svst1(pg, c+16, acc01);
                        svst1(pg, c+32, acc02);
                        svst1(pg, c+48, acc03);
                        c += row_stride_c;
                        svst1(pg, c+0, acc10);
                        svst1(pg, c+16, acc11);
                        svst1(pg, c+32, acc12);
                        svst1(pg, c+48, acc13);
                        c += row_stride_c;
                        svst1(pg, c+0, acc20);
                        svst1(pg, c+16, acc21);
                        svst1(pg, c+32, acc22);
                        svst1(pg, c+48, acc23);
                        c += row_stride_c;
                        svst1(pg, c+0, acc30);
                        svst1(pg, c+16, acc31);
                        svst1(pg, c+32, acc32);
                        svst1(pg, c+48, acc33);
                        c += row_stride_c;
                        svst1(pg, c+0, acc40);
                        svst1(pg, c+16, acc41);
                        svst1(pg, c+32, acc42);
                        svst1(pg, c+48, acc43);
                        c += row_stride_c;
                        svst1(pg, c+0, acc50);
                        svst1(pg, c+16, acc51);
                        svst1(pg, c+32, acc52);
                        svst1(pg, c+48, acc53);
                        c += row_stride_c;
                        svst1(pg, c+0, acc60);
                        svst1(pg, c+16, acc61);
                        svst1(pg, c+32, acc62);
                        svst1(pg, c+48, acc63);
                        c += row_stride_c;
                        svst1(pg, c+0, acc70);
                        svst1(pg, c+16, acc71);
                        svst1(pg, c+32, acc72);
                        svst1(pg, c+48, acc73);
                        c += row_stride_c;

                        svst1(pg, c+0, acc80);
                        svst1(pg, c+16, acc81);
                        svst1(pg, c+32, acc82);
                        svst1(pg, c+48, acc83);
                        c += row_stride_c;
                        svst1(pg, c+0, acc90);
                        svst1(pg, c+16, acc91);
                        svst1(pg, c+32, acc92);
                        svst1(pg, c+48, acc93);
                        c += row_stride_c;
                        svst1(pg, c+0, acc100);
                        svst1(pg, c+16, acc101);
                        svst1(pg, c+32, acc102);
                        svst1(pg, c+48, acc103);
                        c += row_stride_c;
                        svst1(pg, c+0, acc110);
                        svst1(pg, c+16, acc111);
                        svst1(pg, c+32, acc112);
                        svst1(pg, c+48, acc113);
                        c += row_stride_c;
                        svst1(pg, c+0, acc120);
                        svst1(pg, c+16, acc121);
                        svst1(pg, c+32, acc122);
                        svst1(pg, c+48, acc123);
                        c += row_stride_c;
                        svst1(pg, c+0, acc130);
                        svst1(pg, c+16, acc131);
                        svst1(pg, c+32, acc132);
                        svst1(pg, c+48, acc133);
                        c += row_stride_c;
                        svst1(pg, c+0, acc140);
                        svst1(pg, c+16, acc141);
                        svst1(pg, c+32, acc142);
                        svst1(pg, c+48, acc143);
                        c += row_stride_c;
                        svst1(pg, c+0, acc150);
                        svst1(pg, c+16, acc151);
                        svst1(pg, c+32, acc152);
                        svst1(pg, c+48, acc153);
                }

        }
        else if(rem ==1)
        {

               svfloat32_t acc00 = svdup_f32(0.0f), acc10 = svdup_f32(0.0f), acc20 = svdup_f32(0.0f), acc30=svdup_f32(0.0f);
                svfloat32_t acc40 = svdup_f32(0.0f), acc50 = svdup_f32(0.0f), acc60 = svdup_f32(0.0f), acc70=svdup_f32(0.0f);
                svfloat32_t acc80 = svdup_f32(0.0f), acc90 = svdup_f32(0.0f), acc100 = svdup_f32(0.0f), acc110=svdup_f32(0.0f);
                svfloat32_t acc120 = svdup_f32(0.0f), acc130 = svdup_f32(0.0f), acc140 = svdup_f32(0.0f), acc150=svdup_f32(0.0f);
                svfloat32_t acc01 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc31=svdup_f32(0.0f);
                svfloat32_t acc02 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc32=svdup_f32(0.0f);
                svfloat32_t acc03 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f), acc33=svdup_f32(0.0f);
                svfloat32_t acc41 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc71=svdup_f32(0.0f);
                svfloat32_t acc42 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc72=svdup_f32(0.0f);
                svfloat32_t acc43 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f), acc73=svdup_f32(0.0f);

                do{
                          //   printf("I am going in sve");
                svbool_t pg = svwhilelt_b32(i1,simd_width);
                const svfloat32_t a0 = svld1rq(pg, a + 0);
        //      const svfloat32_t a0 = svld1(pg, a + 0);
                const svfloat32_t b0 = svld1(pg, b +  0);
                acc00 = svmla_m(pg, acc00, a0, b0);
                const svfloat32_t a1 = svld1rq(pg, a + 4);
                acc10 = svmla_m(pg, acc10, a1, b0);
                const svfloat32_t a2 = svld1rq(pg, a + 8);
                acc20 = svmla_m(pg, acc20, a2, b0);
                const svfloat32_t a3 = svld1rq(pg, a + 12);
                acc30 = svmla_m(pg, acc30, a3, b0);
                const svfloat32_t a4 = svld1rq(pg, a + 16);
                acc40 = svmla_m(pg, acc40, a4, b0);
               const svfloat32_t a5 = svld1rq(pg, a + 20);
                acc50 = svmla_m(pg, acc50, a5, b0);
                const svfloat32_t a6 = svld1rq(pg, a + 24);
                acc60 = svmla_m(pg, acc60, a6, b0);
                const svfloat32_t a7 = svld1rq(pg, a + 28);
                acc70 = svmla_m(pg, acc70, a7, b0);
                const svfloat32_t a8 = svld1rq(pg, a +  32);
                acc80 = svmla_m(pg, acc80, a8, b0);
                const svfloat32_t a9 = svld1rq(pg, a + 36);
                acc90 = svmla_m(pg, acc90, a9, b0);
                const svfloat32_t a10 = svld1rq(pg, a + 40);
                acc100 = svmla_m(pg, acc100, a10, b0);
                const svfloat32_t a11 = svld1rq(pg, a + 44);
                acc110 = svmla_m(pg, acc110, a11, b0);
                const svfloat32_t a12 = svld1rq(pg, a + 48);
                acc120 = svmla_m(pg, acc120, a12, b0);
                const svfloat32_t a13 = svld1rq(pg, a + 52);
                acc130 = svmla_m(pg, acc130, a13, b0);
                const svfloat32_t a14 = svld1rq(pg, a + 56);
                acc140 = svmla_m(pg, acc140, a14, b0);
                const svfloat32_t a15 = svld1rq(pg, a + 60);
                acc150 = svmla_m(pg, acc150, a15, b0);

                a+=64;
                b+=64;
                }while (--k);
                 if (update != 0)
                {
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc00));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc10));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc20));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc30));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc40));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc50));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc60));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc70));

                             c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc80));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc90));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc100));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc110));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc120));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc130));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc140));
                        c += row_stride_c;
                        svst1(pg, c+0, svadd_m(pg, svld1(pg, c+0), acc150));
                }
                else
                {
                        svst1(pg, c+0, acc00);
                        c += row_stride_c;
                        svst1(pg, c+0, acc10);
                        c += row_stride_c;
                        svst1(pg, c+0, acc20);
                        c += row_stride_c;
                        svst1(pg, c+0, acc30);
                        c += row_stride_c;
                        svst1(pg, c+0, acc40);
                        c += row_stride_c;
                        svst1(pg, c+0, acc50);
                        c += row_stride_c;
                        svst1(pg, c+0, acc60);
                        c += row_stride_c;
                        svst1(pg, c+0, acc70);
                         c += row_stride_c;
                        svst1(pg, c+0, acc80);
                        c += row_stride_c;
                        svst1(pg, c+0, acc90);
			                        c += row_stride_c;
                        svst1(pg, c+0, acc100);
                        c += row_stride_c;
                        svst1(pg, c+0, acc110);
                        c += row_stride_c;
                        svst1(pg, c+0, acc120);
                        c += row_stride_c;
                        svst1(pg, c+0, acc130);
                        c += row_stride_c;
                        svst1(pg, c+0, acc140);
                        c += row_stride_c;
                        svst1(pg, c+0, acc150);

                }

        }



    }
}*/
/*
void nnp_s4gemm_upto_3x3__neon(
        uint32_t mr, uint32_t nr,
        size_t k, size_t update,
        const float a[restrict static 1],
        const float b[restrict static 1],
        float c[restrict static 1],
        size_t row_stride_c)
{
        int simd_width=nnp_hwinfo.simd_width;
        svfloat32_t acc00 = svdup_f32(0.0f), acc01 = svdup_f32(0.0f), acc02 = svdup_f32(0.0f), acc03 = svdup_f32(0.0f);
        svfloat32_t acc04 = svdup_f32(0.0f), acc05 = svdup_f32(0.0f), acc06 = svdup_f32(0.0f), acc07 = svdup_f32(0.0f);
        svfloat32_t acc08 = svdup_f32(0.0f), acc09 = svdup_f32(0.0f), acc010 = svdup_f32(0.0f), acc011 = svdup_f32(0.0f);
        svfloat32_t acc012 = svdup_f32(0.0f), acc013 = svdup_f32(0.0f), acc014 = svdup_f32(0.0f), acc015 = svdup_f32(0.0f);
        svfloat32_t acc10 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f);
        svfloat32_t acc14 = svdup_f32(0.0f), acc15 = svdup_f32(0.0f), acc16 = svdup_f32(0.0f), acc17 = svdup_f32(0.0f);
        svfloat32_t acc18 = svdup_f32(0.0f), acc19 = svdup_f32(0.0f), acc110 = svdup_f32(0.0f), acc111 = svdup_f32(0.0f);
        svfloat32_t acc112 = svdup_f32(0.0f), acc113 = svdup_f32(0.0f), acc114 = svdup_f32(0.0f), acc115 = svdup_f32(0.0f);
        svfloat32_t acc20 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f);
        svfloat32_t acc24 = svdup_f32(0.0f), acc25 = svdup_f32(0.0f), acc26 = svdup_f32(0.0f), acc27 = svdup_f32(0.0f);
        svfloat32_t acc28 = svdup_f32(0.0f), acc29 = svdup_f32(0.0f), acc210 = svdup_f32(0.0f), acc211 = svdup_f32(0.0f);
        svfloat32_t acc212 = svdup_f32(0.0f), acc213 = svdup_f32(0.0f), acc214 = svdup_f32(0.0f), acc215 = svdup_f32(0.0f);
        svfloat32_t acc30 = svdup_f32(0.0f), acc31 = svdup_f32(0.0f), acc32 = svdup_f32(0.0f), acc33 = svdup_f32(0.0f);
        svfloat32_t acc34 = svdup_f32(0.0f), acc35 = svdup_f32(0.0f), acc36 = svdup_f32(0.0f), acc37 = svdup_f32(0.0f);
        svfloat32_t acc38 = svdup_f32(0.0f), acc39 = svdup_f32(0.0f), acc310 = svdup_f32(0.0f), acc311 = svdup_f32(0.0f);
        svfloat32_t acc312 = svdup_f32(0.0f), acc313 = svdup_f32(0.0f), acc314 = svdup_f32(0.0f), acc315 = svdup_f32(0.0f);
        svfloat32_t acc40 = svdup_f32(0.0f), acc41 = svdup_f32(0.0f), acc42 = svdup_f32(0.0f), acc43 = svdup_f32(0.0f);
        svfloat32_t acc44 = svdup_f32(0.0f), acc45 = svdup_f32(0.0f), acc46 = svdup_f32(0.0f), acc47 = svdup_f32(0.0f);
        svfloat32_t acc48 = svdup_f32(0.0f), acc49 = svdup_f32(0.0f), acc410 = svdup_f32(0.0f), acc411 = svdup_f32(0.0f);
        svfloat32_t acc412 = svdup_f32(0.0f), acc413 = svdup_f32(0.0f), acc414 = svdup_f32(0.0f), acc415 = svdup_f32(0.0f);
        svfloat32_t acc50 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f);
        svfloat32_t acc54 = svdup_f32(0.0f), acc55 = svdup_f32(0.0f), acc56 = svdup_f32(0.0f), acc57 = svdup_f32(0.0f);
        svfloat32_t acc58 = svdup_f32(0.0f), acc59 = svdup_f32(0.0f), acc510 = svdup_f32(0.0f), acc511 = svdup_f32(0.0f);
        svfloat32_t acc512 = svdup_f32(0.0f), acc513 = svdup_f32(0.0f), acc514 = svdup_f32(0.0f), acc515 = svdup_f32(0.0f);
        svfloat32_t acc60 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f);
        svfloat32_t acc64 = svdup_f32(0.0f), acc65 = svdup_f32(0.0f), acc66 = svdup_f32(0.0f), acc67 = svdup_f32(0.0f);
        svfloat32_t acc68 = svdup_f32(0.0f), acc69 = svdup_f32(0.0f), acc610 = svdup_f32(0.0f), acc611 = svdup_f32(0.0f);
        svfloat32_t acc612 = svdup_f32(0.0f), acc613 = svdup_f32(0.0f), acc614 = svdup_f32(0.0f), acc615 = svdup_f32(0.0f);
        svfloat32_t acc70 = svdup_f32(0.0f), acc71 = svdup_f32(0.0f), acc72 = svdup_f32(0.0f), acc73 = svdup_f32(0.0f);
        svfloat32_t acc74 = svdup_f32(0.0f), acc75 = svdup_f32(0.0f), acc76 = svdup_f32(0.0f), acc77 = svdup_f32(0.0f);
        svfloat32_t acc78 = svdup_f32(0.0f), acc79 = svdup_f32(0.0f), acc710 = svdup_f32(0.0f), acc711 = svdup_f32(0.0f);
        svfloat32_t acc712 = svdup_f32(0.0f), acc713 = svdup_f32(0.0f), acc714 = svdup_f32(0.0f), acc715 = svdup_f32(0.0f);
        svfloat32_t acc80 = svdup_f32(0.0f), acc81 = svdup_f32(0.0f), acc82 = svdup_f32(0.0f), acc83 = svdup_f32(0.0f);
        svfloat32_t acc84 = svdup_f32(0.0f), acc85 = svdup_f32(0.0f), acc86 = svdup_f32(0.0f), acc87 = svdup_f32(0.0f);
        svfloat32_t acc88 = svdup_f32(0.0f), acc89 = svdup_f32(0.0f), acc810 = svdup_f32(0.0f), acc811 = svdup_f32(0.0f);
        svfloat32_t acc812 = svdup_f32(0.0f), acc813 = svdup_f32(0.0f), acc814 = svdup_f32(0.0f), acc815 = svdup_f32(0.0f);
        const float* b0_ptr = b;
        const float* b1_ptr = (nr >= 2) ? b + simd_width : b;
        const float* b2_ptr = (nr >= 3) ? b + 2*simd_width : b;
        const float* b3_ptr = (nr >= 4) ? b + 3*simd_width : b;
        const float* b4_ptr = (nr >= 5) ? b + 4*simd_width : b;
        const float* b5_ptr = (nr >= 6) ? b + 5*simd_width : b;
        const float* b6_ptr = (nr >= 7) ? b + 6*simd_width : b;
        const float* b7_ptr = (nr >= 8) ? b + 7*simd_width : b;
        const float* b8_ptr = (nr >= 9) ? b + 8*simd_width : b;
        const float* b9_ptr = (nr >= 10) ? b + 9*simd_width : b;
        const float* b10_ptr = (nr >= 11) ? b + 10*simd_width : b;
        const float* b11_ptr = (nr >= 12) ? b + 11*simd_width : b;
        const float* b12_ptr = (nr >= 13) ? b + 12*simd_width : b;
        const float* b13_ptr = (nr >= 14) ? b + 13*simd_width : b;
        const float* b14_ptr = (nr >= 15) ? b + 14*simd_width : b;
        const float* b15_ptr = (nr >= 16) ? b + 15*simd_width : b;
        const size_t b_increment = nr * simd_width;
        switch (mr) {
                case 1:
                {
                        do {
                                for(int i1=0;i1<simd_width;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                svbool_t pg = svwhilelt_b32(i1,simd_width);

                                const svfloat32_t a0 = svld1rq(pg, a); a += simd_width;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);

                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc01 = svmla_m(pg, acc01, a0, b1);

                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                const svfloat32_t b4 = svld1(pg, b4_ptr); b4_ptr += b_increment;
                                acc04 = svmla_m(pg, acc04, a0, b4);
                                const svfloat32_t b5 = svld1(pg, b5_ptr); b5_ptr += b_increment;
                                acc05 = svmla_m(pg, acc05, a0, b5);
                                const svfloat32_t b6 = svld1(pg, b6_ptr); b6_ptr += b_increment;
                                acc06 = svmla_m(pg, acc06, a0, b6);
                                const svfloat32_t b7 = svld1(pg, b7_ptr); b7_ptr += b_increment;
                                acc07 = svmla_m(pg, acc07, a0, b7);
                                ////////
                                //
                                const svfloat32_t b8 = svld1(pg, b8_ptr); b8_ptr += b_increment;
                                acc08 = svmla_m(pg, acc08, a0, b8);
                                const svfloat32_t b9 = svld1(pg, b9_ptr); b9_ptr += b_increment;
                                acc09 = svmla_m(pg, acc09, a0, b9);
                                const svfloat32_t b10 = svld1(pg, b10_ptr); b10_ptr += b_increment;
                                acc010 = svmla_m(pg, acc010, a0, b10);
                                const svfloat32_t b11 = svld1(pg, b11_ptr); b11_ptr += b_increment;
                                acc011 = svmla_m(pg, acc011, a0, b11);
                                const svfloat32_t b12 = svld1(pg, b12_ptr); b12_ptr += b_increment;
                                acc012 = svmla_m(pg, acc012, a0, b12);
                                const svfloat32_t b13 = svld1(pg, b13_ptr); b13_ptr += b_increment;
                                acc013 = svmla_m(pg, acc013, a0, b13);
                                const svfloat32_t b14 = svld1(pg, b14_ptr); b14_ptr += b_increment;
                                acc014 = svmla_m(pg, acc014, a0, b14);
                                const svfloat32_t b15 = svld1(pg, b15_ptr); b15_ptr += b_increment;
                                acc015 = svmla_m(pg, acc015, a0, b15);

                                }
                        } while (--k);
                        for(int i1=0;i1<simd_width;i1+=svcntw())
                        {
                       // printf("I am going in sve");
                        svbool_t pg = svwhilelt_b32(i1,simd_width);

                        if (update != 0) {
                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc00)); c += simd_width;
                                if (nr >= 2) {
                                        svst1(pg, c, svadd_m(pg, svld1(pg, c), acc01)); c += simd_width;
                                        if (nr >= 3) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc02)); c += simd_width;
                                        if (nr >= 4) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc03)); c += simd_width;
                                        if (nr >= 5) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc04)); c += simd_width;
                                        if (nr >= 6) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc05)); c += simd_width;
                                        if (nr >= 7) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc06)); c += simd_width;
                                        if (nr >= 8) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc07)); c += simd_width;
                                        if (nr >= 9) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc08)); c += simd_width;
                                        if (nr >= 10) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc09)); c += simd_width;
                                        if (nr >= 11) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc010)); c += simd_width;
                                        if (nr >= 12) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc011)); c += simd_width;
                                        if (nr >= 13) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc012)); c += simd_width;
                                        if (nr >= 14) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc013)); c += simd_width;
                                        if (nr >= 15) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc014)); c += simd_width;
                                        if (nr >= 16) {
                                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc015));


                                       }}}}}}}}}}}}}
                                }
                        }} else {
                                svst1(pg, c, acc00); c += simd_width;
                                 if (nr >= 2) {
                                        svst1(pg, c, acc01); c += simd_width;
                                        if (nr >= 3) {
                                                svst1(pg, c, acc02);  c += simd_width;
                                        if (nr >= 4) {
                                                svst1(pg, c, acc03); c += simd_width;
                                        if (nr >= 5) {
                                                svst1(pg, c, acc04); c += simd_width;
                                        if (nr >= 6) {
                                                svst1(pg, c, acc05); c += simd_width;
                                        if (nr >= 7) {
                                                svst1(pg, c, acc06); c += simd_width;
                                        if (nr >= 8) {
                                                svst1(pg, c, acc07); c += simd_width;
                                        if (nr >= 9) {
                                                svst1(pg, c, acc08); c += simd_width;
                                        if (nr >= 10) {
                                                svst1(pg, c, acc09); c += simd_width;
                                        if (nr >= 11) {
                                                svst1(pg, c, acc010); c += simd_width;
                                        if (nr >= 12) {
                                                svst1(pg, c, acc011); c += simd_width;
                                        if (nr >= 13) {
                                                svst1(pg, c, acc012); c += simd_width;
                                        if (nr >= 14) {
                                                svst1(pg, c, acc013); c += simd_width;
                                        if (nr >= 15) {
                                                svst1(pg, c, acc014);  c += simd_width;
                                        if (nr >= 16) {
                                                svst1(pg, c, acc015);
                                        }} } } }}}}}}}}}
                                }
                        }
                        }}
                         break;
                }
                case 2:
                {
                        do {
                                for(int i1=0;i1<simd_width;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,simd_width);

                                const svfloat32_t a0 = svld1(pg, a); a += simd_width;
                                const svfloat32_t a1 = svld1(pg, a); a += simd_width;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);

                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);

                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);
                                const svfloat32_t b4 = svld1(pg, b4_ptr); b4_ptr += b_increment;
                                acc04 = svmla_m(pg, acc04, a0, b4);
                                acc14 = svmla_m(pg, acc14, a1, b4);
                                const svfloat32_t b5 = svld1(pg, b5_ptr); b5_ptr += b_increment;
                                acc05 = svmla_m(pg, acc05, a0, b5);
                                acc15 = svmla_m(pg, acc15, a1, b5);
                                const svfloat32_t b6 = svld1(pg, b6_ptr); b6_ptr += b_increment;
                                acc06 = svmla_m(pg, acc06, a0, b6);
                                acc16 = svmla_m(pg, acc16, a1, b6);
                                const svfloat32_t b7 = svld1(pg, b7_ptr); b7_ptr += b_increment;
                                acc07 = svmla_m(pg, acc07, a0, b7);
                                acc17 = svmla_m(pg, acc17, a1, b7);

                               const svfloat32_t b8 = svld1(pg, b8_ptr); b8_ptr += b_increment;
                                acc08 = svmla_m(pg, acc08, a0, b8);
                                acc18 = svmla_m(pg, acc18, a1, b8);
                                const svfloat32_t b9 = svld1(pg, b9_ptr); b9_ptr += b_increment;
                                acc09 = svmla_m(pg, acc09, a0, b9);
                                acc19 = svmla_m(pg, acc19, a1, b9);
                                const svfloat32_t b10 = svld1(pg, b10_ptr); b10_ptr += b_increment;
                                acc010 = svmla_m(pg, acc010, a0, b10);
                                acc110 = svmla_m(pg, acc110, a1, b10);
                                const svfloat32_t b11 = svld1(pg, b11_ptr); b11_ptr += b_increment;
                                acc011 = svmla_m(pg, acc011, a0, b11);
                                acc111 = svmla_m(pg, acc111, a1, b11);
                                const svfloat32_t b12 = svld1(pg, b12_ptr); b12_ptr += b_increment;
                                acc012 = svmla_m(pg, acc012, a0, b12);
                                acc112 = svmla_m(pg, acc112, a1, b12);
                                const svfloat32_t b13 = svld1(pg, b13_ptr); b13_ptr += b_increment;
                                acc013 = svmla_m(pg, acc013, a0, b13);
                                acc113 = svmla_m(pg, acc113, a1, b13);
                                const svfloat32_t b14 = svld1(pg, b14_ptr); b14_ptr += b_increment;
                                acc014 = svmla_m(pg, acc014, a0, b14);
                                acc114 = svmla_m(pg, acc114, a1, b14);
                                const svfloat32_t b15 = svld1(pg, b15_ptr); b15_ptr += b_increment;
                                acc015 = svmla_m(pg, acc015, a0, b15);
                                acc115 = svmla_m(pg, acc115, a1, b15);

                                }
                        } while (--k);
                        for(int i1=0;i1<simd_width;i1+=svcntw())
                        {
                         //       printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,simd_width);
                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); crow0 += simd_width;
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); crow1 += simd_width;
                                if (nr >= 2) {
                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc01)); crow0 += simd_width;
                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc11)); crow1 += simd_width;
                                        if (nr >= 3) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc02)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc12)); crow1 += simd_width;
                                        if (nr >= 4) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc03));  crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc13)); crow1 += simd_width;
                                        if (nr >= 5) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc04));  crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc14)); crow1 += simd_width;
                                        if (nr >= 6) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc05)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc15));crow1 += simd_width;
                                        if (nr >= 7) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc06)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc16));crow1 += simd_width;
                                        if (nr >= 8) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc07)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc17)); crow1 += simd_width;

                                        ////
                                        if (nr >=  9) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc08)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc18)); crow1 += simd_width;

                                        if (nr >=  10) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc09)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc19)); crow1 += simd_width;

                                        if (nr >=  11) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc010)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc110)); crow1 += simd_width;
                                        if (nr >=  12) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc011)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc111)); crow1 += simd_width;
                                        if (nr >=  13) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc012)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc112)); crow1 += simd_width;
                                        if (nr >=  14) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc013)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc113)); crow1 += simd_width;
                                        if (nr >=  15) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc014)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc114)); crow1 += simd_width;
                                        if (nr >=  16) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc015));
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc115));


                                        }}}}}}}}}}}}}}
                                }
                        } else {

                                svst1(pg, crow0, acc00); crow0 += simd_width;
                                svst1(pg, crow1, acc10); crow1 += simd_width;
                                if (nr >= 2) {
                                        svst1(pg, crow0, acc01); crow0 += simd_width;
                                        svst1(pg, crow1, acc11); crow1 += simd_width;
                                        if (nr >= 3) {
                                                svst1(pg, crow0, acc02); crow0 += simd_width;
                                                svst1(pg, crow1, acc12);  crow1 += simd_width;
                                        if (nr >= 4) {
                                                svst1(pg, crow0, acc03); crow0 += simd_width;
                                                svst1(pg, crow1, acc13);crow1 += simd_width;
                                        if (nr >= 5) {
                                                svst1(pg, crow0, acc04);  crow0 += simd_width;
                                                svst1(pg, crow1, acc14);crow1 += simd_width;
                                        if (nr >= 6) {
                                                svst1(pg, crow0, acc05); crow0 += simd_width;
                                                svst1(pg, crow1, acc15); crow1 += simd_width;
                                        if (nr >= 7) {
                                                svst1(pg, crow0, acc06); crow0 += simd_width;
                                                svst1(pg, crow1, acc16);crow1 += simd_width;
                                        if (nr >= 8) {
                                                svst1(pg, crow0, acc07); crow0 += simd_width;
                                                svst1(pg, crow1, acc17); crow1 += simd_width;
                                        //////
                                        if (nr >= 9) {
                                                svst1(pg, crow0, acc08); crow0 += simd_width;
                                                svst1(pg, crow1, acc18); crow1 += simd_width;
                                        if (nr >= 10) {
                                                svst1(pg, crow0, acc09); crow0 += simd_width;
                                                svst1(pg, crow1, acc19); crow1 += simd_width;
                                        if (nr >= 11) {
                                                svst1(pg, crow0, acc010); crow0 += simd_width;
                                                svst1(pg, crow1, acc110);  crow1 += simd_width;
                                        if (nr >= 12) {
                                                svst1(pg, crow0, acc011);  crow0 += simd_width;
                                                svst1(pg, crow1, acc111); crow1 += simd_width;
                                        if (nr >= 13) {
                                                svst1(pg, crow0, acc012); crow0 += simd_width;
                                                svst1(pg, crow1, acc112); crow1 += simd_width;

                                        if (nr >= 14) {
                                                svst1(pg, crow0, acc013); crow0 += simd_width;
                                                svst1(pg, crow1, acc113); crow1 += simd_width;
                                        if (nr >= 15) {
                                                svst1(pg, crow0, acc014); crow0 += simd_width;
                                                svst1(pg, crow1, acc114); crow1 += simd_width;
                                        if (nr >= 16) {
                                                svst1(pg, crow0, acc015);
                                                svst1(pg, crow1, acc115);

                                        }}}}}}}}}}}}}}}
                                }
                        }
                        break;
                }
                case 3:
                {
                        do {
                                for(int i1=0;i1<simd_width;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,simd_width);
                                const svfloat32_t a0 = svld1(pg, a); a += simd_width;
                                const svfloat32_t a1 = svld1(pg, a); a += simd_width;
                                const svfloat32_t a2 = svld1(pg, a); a += simd_width;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);

                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);

                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                acc22 = svmla_m(pg, acc22, a2, b2);
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);
                                acc23 = svmla_m(pg, acc23, a2, b3);
                                const svfloat32_t b4 = svld1(pg, b4_ptr); b4_ptr += b_increment;
                                acc04 = svmla_m(pg, acc04, a0, b4);
                                acc14 = svmla_m(pg, acc14, a1, b4);
                                acc24 = svmla_m(pg, acc24, a2, b4);
                                const svfloat32_t b5 = svld1(pg, b5_ptr); b5_ptr += b_increment;
                                acc05 = svmla_m(pg, acc05, a0, b5);
                                acc15 = svmla_m(pg, acc15, a1, b5);
                                acc25 = svmla_m(pg, acc25, a2, b5);
                                const svfloat32_t b6 = svld1(pg, b6_ptr); b6_ptr += b_increment;
                                acc06 = svmla_m(pg, acc06, a0, b6);
                                acc16 = svmla_m(pg, acc16, a1, b6);
                                acc26 = svmla_m(pg, acc26, a2, b6);
                                const svfloat32_t b7 = svld1(pg, b7_ptr); b7_ptr += b_increment;
                                acc07 = svmla_m(pg, acc07, a0, b7);
                                acc17 = svmla_m(pg, acc17, a1, b7);
                                acc27 = svmla_m(pg, acc27, a2, b7);
                                ///
                                const svfloat32_t b8 = svld1(pg, b8_ptr); b8_ptr += b_increment;
                                acc08 = svmla_m(pg, acc08, a0, b8);
                                acc18 = svmla_m(pg, acc18, a1, b8);
                                acc28 = svmla_m(pg, acc28, a2, b8);
                                const svfloat32_t b9 = svld1(pg, b9_ptr); b9_ptr += b_increment;
                                acc09 = svmla_m(pg, acc09, a0, b9);
                                acc19 = svmla_m(pg, acc19, a1, b9);
                                acc29 = svmla_m(pg, acc29, a2, b9);
                                const svfloat32_t b10 = svld1(pg, b10_ptr); b10_ptr += b_increment;
                                acc010 = svmla_m(pg, acc010, a0, b10);
                                acc110 = svmla_m(pg, acc110, a1, b10);
                                acc210 = svmla_m(pg, acc210, a2, b10);
                                const svfloat32_t b11 = svld1(pg, b11_ptr); b11_ptr += b_increment;
                                acc011 = svmla_m(pg, acc011, a0, b11);
                                acc111 = svmla_m(pg, acc111, a1, b11);
                                acc211 = svmla_m(pg, acc211, a2, b11);
                                const svfloat32_t b12 = svld1(pg, b12_ptr); b12_ptr += b_increment;
                                acc012 = svmla_m(pg, acc012, a0, b12);
                                acc112 = svmla_m(pg, acc112, a1, b12);
                                acc212 = svmla_m(pg, acc212, a2, b12);
                                const svfloat32_t b13 = svld1(pg, b13_ptr); b13_ptr += b_increment;
                                acc013 = svmla_m(pg, acc013, a0, b13);
                                acc113 = svmla_m(pg, acc113, a1, b13);
                                acc213 = svmla_m(pg, acc213, a2, b13);
                                const svfloat32_t b14 = svld1(pg, b14_ptr); b14_ptr += b_increment;
                                acc014 = svmla_m(pg, acc014, a0, b14);
                                acc114 = svmla_m(pg, acc114, a1, b14);
                                acc214 = svmla_m(pg, acc214, a2, b14);
                                const svfloat32_t b15 = svld1(pg, b15_ptr); b15_ptr += b_increment;
                                acc015 = svmla_m(pg, acc015, a0, b15);
                                acc115 = svmla_m(pg, acc115, a1, b15);
                                acc215 = svmla_m(pg, acc215, a2, b15);

                                }
                        } while (--k);

                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        for(int i1=0;i1<simd_width;i1+=svcntw())
                        {
                   //     printf("I am going in sve");
                        svbool_t pg = svwhilelt_b32(i1,simd_width);
                        if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); crow0 += simd_width;
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); crow1 += simd_width;
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20)); crow2 += simd_width;
                                if (nr >= 2) {
                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc01)); crow0 += simd_width;
                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc11)); crow1 += simd_width;
                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc21)); crow2 += simd_width;
                                        if (nr >= 3) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc02)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc12)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc22)); crow2 += simd_width;
                                        if (nr >= 4) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc03)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc13));crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc23)); crow2 += simd_width;
                                        if (nr >= 5) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc04));crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc14));crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc24));crow2 += simd_width;
                                        if (nr >= 6) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc05));crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc15));crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc25));crow2 += simd_width;
                                        if (nr >= 7) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc06));crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc16));crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc26)); crow2 += simd_width;
                                        if (nr >= 8) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc07)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc17)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc27));  crow2 += simd_width;
                                                ///
                                        if (nr >= 9) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc08)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc18)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc28)); crow2 += simd_width;
                                        if (nr >= 10) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc09)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc19)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc29)); crow2 += simd_width;
                                        if (nr >= 11) {
                                                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc010));  crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc110)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc210)); crow2 += simd_width;
                                        if (nr >= 12) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc011)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc111)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc211)); crow2 += simd_width;
                                        if (nr >= 13) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc012)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc112)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc212));  crow2 += simd_width;
                                        if (nr >= 14) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc013)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc113)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc213)); crow2 += simd_width;
                                        if (nr >= 15) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc014)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc114)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc214)); crow2 += simd_width;
                                        if (nr >= 16) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc015));
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc115));
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc215));
                                      }}}}}}}}}}}}}}
                                }
                        } else {
                                svst1(pg, crow0, acc00); crow0 += simd_width;
                                svst1(pg, crow1, acc10); crow1 += simd_width;
                                svst1(pg, crow2, acc20); crow2 += simd_width;
                                if (nr >= 2) {
                                        svst1(pg, crow0, acc01); crow0 += simd_width;
                                        svst1(pg, crow1, acc11); crow1 += simd_width;
                                        svst1(pg, crow2, acc21); crow2 += simd_width;
                                        if (nr >= 3) {
                                                svst1(pg, crow0, acc02);crow0 += simd_width;
                                                svst1(pg, crow1, acc12);crow1 += simd_width;
                                                svst1(pg, crow2, acc22);crow2 += simd_width;
                                        if (nr >= 4) {
                                               svst1(pg, crow0, acc03);crow0 += simd_width;
                                                svst1(pg, crow1, acc13);crow1 += simd_width;
                                                svst1(pg, crow2, acc23);crow2 += simd_width;
                                        if (nr >= 5) {
                                                svst1(pg, crow0, acc04);crow0 += simd_width;
                                                svst1(pg, crow1, acc14);crow1 += simd_width;
                                                svst1(pg, crow2, acc24);crow2 += simd_width;
                                        if (nr >= 6) {
                                                svst1(pg, crow0, acc05);crow0 += simd_width;
                                                svst1(pg, crow1, acc15);crow1 += simd_width;
                                                svst1(pg, crow2, acc25);crow2 += simd_width;
                                        if (nr >= 7) {
                                                svst1(pg, crow0, acc06);crow0 += simd_width;
                                                svst1(pg, crow1, acc16);crow1 += simd_width;
                                                svst1(pg, crow2, acc26); crow2 += simd_width;
                                        if (nr >= 8) {
                                                svst1(pg, crow0, acc07); crow0 += simd_width;
                                                svst1(pg, crow1, acc17); crow1 += simd_width;
                                                svst1(pg, crow2, acc27); crow2 += simd_width;

                                        ////
                                        if (nr >= 9) {
                                                svst1(pg, crow0, acc08); crow0 += simd_width;
                                                svst1(pg, crow1, acc18); crow1 += simd_width;
                                                svst1(pg, crow2, acc28); crow2 += simd_width;
                                        if (nr >= 10) {
                                                svst1(pg, crow0, acc09); crow0 += simd_width;
                                                svst1(pg, crow1, acc19); crow1 += simd_width;
                                                svst1(pg, crow2, acc29); crow2 += simd_width;
                                        if (nr >= 11) {
                                                svst1(pg, crow0, acc010); crow0 += simd_width;
                                                svst1(pg, crow1, acc110); crow1 += simd_width;
                                                svst1(pg, crow2, acc210); crow2 += simd_width;
                                        if (nr >= 12) {
                                                svst1(pg, crow0, acc011); crow0 += simd_width;
                                                svst1(pg, crow1, acc111);  crow1 += simd_width;
                                                svst1(pg, crow2, acc211); crow2 += simd_width;
                                        if (nr >= 13) {
                                                svst1(pg, crow0, acc012); crow0 += simd_width;
                                                svst1(pg, crow1, acc112); crow1 += simd_width;
                                                svst1(pg, crow2, acc212); crow2 += simd_width;
                                        if (nr >= 14) {
                                                svst1(pg, crow0, acc013); crow0 += simd_width;
                                                svst1(pg, crow1, acc113); crow1 += simd_width;
                                                svst1(pg, crow2, acc213); crow2 += simd_width;
                                        if (nr >= 15) {
                                                svst1(pg, crow0, acc014); crow0 += simd_width;
                                                svst1(pg, crow1, acc114); crow1 += simd_width;
                                                svst1(pg, crow2, acc214); crow2 += simd_width;
                                        if (nr >= 16) {
                                                svst1(pg, crow0, acc015);
                                                svst1(pg, crow1, acc115);
                                                svst1(pg, crow2, acc215);
                                        }}}}}}}}}}}}}}
                                }
                        }}
                        break;
                }
                 case 4:
                {
                        //printf("I am going in sve");
                        do {
                                for(int i1=0;i1<4;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,4);
                                const svfloat32_t a0 = svld1(pg, a); a += 4;
                                const svfloat32_t a1 = svld1(pg, a); a += 4;
                                const svfloat32_t a2 = svld1(pg, a); a += 4;
                                const svfloat32_t a3 = svld1(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);
                                acc31 = svmla_m(pg, acc31, a3, b1);

                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                acc22 = svmla_m(pg, acc22, a2, b2);
                                acc32 = svmla_m(pg, acc32, a3, b2);
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);
                                acc23 = svmla_m(pg, acc23, a2, b3);
                                acc33 = svmla_m(pg, acc33, a3, b3);
                                const svfloat32_t b4 = svld1(pg, b4_ptr); b4_ptr += b_increment;
                                acc04 = svmla_m(pg, acc04, a0, b4);
                                acc14 = svmla_m(pg, acc14, a1, b4);
                                acc24 = svmla_m(pg, acc24, a2, b4);
                                acc34 = svmla_m(pg, acc34, a3, b4);
                                const svfloat32_t b5 = svld1(pg, b5_ptr); b5_ptr += b_increment;
                                acc05 = svmla_m(pg, acc05, a0, b5);
                                acc15 = svmla_m(pg, acc15, a1, b5);
                                acc25 = svmla_m(pg, acc25, a2, b5);
                                acc35 = svmla_m(pg, acc35, a3, b5);
                                const svfloat32_t b6 = svld1(pg, b6_ptr); b6_ptr += b_increment;
                                acc06 = svmla_m(pg, acc06, a0, b6);
                                acc16 = svmla_m(pg, acc16, a1, b6);
                                acc26 = svmla_m(pg, acc26, a2, b6);
                                acc36 = svmla_m(pg, acc36, a3, b6);
                                const svfloat32_t b7 = svld1(pg, b7_ptr); b7_ptr += b_increment;
                                acc07 = svmla_m(pg, acc07, a0, b7);
                                acc17 = svmla_m(pg, acc17, a1, b7);
                                acc27 = svmla_m(pg, acc27, a2, b7);
                                acc37 = svmla_m(pg, acc37, a3, b7);
                                ///
                                const svfloat32_t b8 = svld1(pg, b8_ptr); b8_ptr += b_increment;
                               acc08 = svmla_m(pg, acc08, a0, b8);
                                acc18 = svmla_m(pg, acc18, a1, b8);
                                acc28 = svmla_m(pg, acc28, a2, b8);
                                acc38 = svmla_m(pg, acc38, a3, b8);
                                const svfloat32_t b9 = svld1(pg, b9_ptr); b9_ptr += b_increment;
                                acc09 = svmla_m(pg, acc09, a0, b9);
                                acc19 = svmla_m(pg, acc19, a1, b9);
                                acc29 = svmla_m(pg, acc29, a2, b9);
                                acc39 = svmla_m(pg, acc39, a3, b9);
                                const svfloat32_t b10 = svld1(pg, b10_ptr); b10_ptr += b_increment;
                                acc010 = svmla_m(pg, acc010, a0, b10);
                                acc110 = svmla_m(pg, acc110, a1, b10);
                                acc210 = svmla_m(pg, acc210, a2, b10);
                                acc310 = svmla_m(pg, acc310, a3, b10);
                                const svfloat32_t b11 = svld1(pg, b11_ptr); b11_ptr += b_increment;
                                acc011 = svmla_m(pg, acc011, a0, b11);
                                acc111 = svmla_m(pg, acc111, a1, b11);
                                acc211 = svmla_m(pg, acc211, a2, b11);
                                acc311 = svmla_m(pg, acc311, a3, b11);
                                const svfloat32_t b12 = svld1(pg, b12_ptr); b12_ptr += b_increment;
                                acc012 = svmla_m(pg, acc012, a0, b12);
                                acc112 = svmla_m(pg, acc112, a1, b12);
                                acc212 = svmla_m(pg, acc212, a2, b12);
                                acc312 = svmla_m(pg, acc312, a3, b12);
                                const svfloat32_t b13 = svld1(pg, b13_ptr); b13_ptr += b_increment;
                                acc013 = svmla_m(pg, acc013, a0, b13);
                                acc113 = svmla_m(pg, acc113, a1, b13);
                                acc213 = svmla_m(pg, acc213, a2, b13);
                                acc313 = svmla_m(pg, acc313, a3, b13);
                                const svfloat32_t b14 = svld1(pg, b14_ptr); b14_ptr += b_increment;
                                acc014 = svmla_m(pg, acc014, a0, b14);
                                acc114 = svmla_m(pg, acc114, a1, b14);
                                acc214 = svmla_m(pg, acc214, a2, b14);
                                acc314 = svmla_m(pg, acc314, a3, b14);
                                const svfloat32_t b15 = svld1(pg, b15_ptr); b15_ptr += b_increment;
                                acc015 = svmla_m(pg, acc015, a0, b15);
                                acc115 = svmla_m(pg, acc115, a1, b15);
                                acc215 = svmla_m(pg, acc215, a2, b15);
                                acc315 = svmla_m(pg, acc315, a3, b15);


                               }
                        } while (--k);
                          float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                        for(int i1=0;i1<4;i1+=svcntw())
                        {
                        svbool_t pg = svwhilelt_b32(i1,4);
                        if (update != 0) {
                        //      printf("in update not 0");
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); crow0 += simd_width;
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); crow1 += simd_width;
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20)); crow2 += simd_width;
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30)); crow3 += simd_width;

                                if (nr >= 2) {
                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc01)); crow0 += simd_width;
                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc11)); crow1 += simd_width;
                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc21)); crow2 += simd_width;
                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc31)); crow3 += simd_width;

                                        if (nr >= 3) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc02)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc12)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc22));crow2 += simd_width;
                                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc32)); crow3 += simd_width;
                                                if (nr >= 4) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc03)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc13)); crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc23));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc33)); crow3 += simd_width;
                                                if (nr >= 5) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc04));  crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc14));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc24));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc34)); crow3 += simd_width;
                                           
	                                               if (nr >= 6) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc05)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc15)); crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc25));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc35)); crow3 += simd_width;
                                                if (nr >= 7) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc06)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc16)); crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc26));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc36));  crow3 += simd_width;
                                                if (nr >= 8) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc07)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc17));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc27));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc37));  crow3 += simd_width;
                                                ////
                                                if (nr >= 9) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc08)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc18));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc28));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc38));  crow3 += simd_width;
                                                if (nr >= 10) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc09)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc19));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc29));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc39));  crow3 += simd_width;
                                                if (nr >= 11) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc010)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc110));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc210));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc310));  crow3 += simd_width;
                                                if (nr >= 12) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc011)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc111));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc211));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc311));  crow3 += simd_width;
                                                if (nr >= 13) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc012)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc112));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc212));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc312));  crow3 += simd_width;
                                                if (nr >= 14) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc013)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc113));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc213));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc313));  crow3 += simd_width;
                                                if (nr >= 15) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc014)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc114));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc214));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc314));  crow3 += simd_width;
                                                if (nr >= 16) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc015));
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc115));
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc215));
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc315));
                                                        }
                                        }}}}}}}}}}}}}
                                }
                        } else {
                        //      printf("in update  0");
                                svst1(pg, crow0, acc00); crow0 += 4;
                                svst1(pg, crow1, acc10); crow1 += 4;
                                svst1(pg, crow2, acc20); crow2 += 4;
                                svst1(pg, crow3, acc30); crow3 += 4;
                                if (nr >= 2) {
                                        svst1(pg, crow0, acc01); crow0 += 4;
                                       svst1(pg, crow1, acc11); crow1 += simd_width;
                                        svst1(pg, crow2, acc21); crow2 += simd_width;
                                        svst1(pg, crow3, acc31); crow3 += simd_width;
                                        if (nr >= 3) {
                                               svst1(pg, crow0, acc02); crow0 += 4;
                                                svst1(pg, crow1, acc12); crow1 += simd_width;
                                                svst1(pg, crow2, acc22); crow2 += simd_width;
                                                svst1(pg, crow3, acc32); crow3 += simd_width;
                                                if (nr >= 4) {
                                                        svst1(pg, crow0, acc03);  crow0 += 4;
                                                        svst1(pg, crow1, acc13); crow1 += simd_width;
                                                        svst1(pg, crow2, acc23); crow2 += simd_width;
                                                        svst1(pg, crow3, acc33); crow3 += simd_width;
                                                if (nr >= 5) {
                                                        svst1(pg, crow0, acc04);  crow0 += 4;
                                                        svst1(pg, crow1, acc14); crow1 += simd_width;
                                                        svst1(pg, crow2, acc24); crow2 += simd_width;
                                                        svst1(pg, crow3, acc34); crow3 += simd_width;
                                                if (nr >= 6) {
                                                        svst1(pg, crow0, acc05);  crow0 += 4;
                                                        svst1(pg, crow1, acc15); crow1 += simd_width;
                                                        svst1(pg, crow2, acc25); crow2 += simd_width;
                                                        svst1(pg, crow3, acc35); crow3 += simd_width;
                                                if (nr >= 7) {
                                                        svst1(pg, crow0, acc06);  crow0 += 4;
                                                        svst1(pg, crow1, acc16); crow1 += simd_width;
                                                        svst1(pg, crow2, acc26); crow2 += simd_width;
                                                        svst1(pg, crow3, acc36); crow3 += simd_width;
                                                if (nr >= 8) {
                                                        svst1(pg, crow0, acc07); crow0 += 4;
                                                        svst1(pg, crow1, acc17);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc27);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc37);   crow3 += simd_width;
                                                if (nr >= 9) {
                                                       svst1(pg, crow0, acc08); crow0 += 4;
                                                        svst1(pg, crow1, acc18);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc28);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc38);   crow3 += simd_width;
                                                if (nr >= 10) {
                                                        svst1(pg, crow0, acc09); crow0 += 4;
                                                        svst1(pg, crow1, acc19);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc29);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc39);   crow3 += simd_width;
                                                if (nr >= 11) {
                                                        svst1(pg, crow0, acc010); crow0 += 4;
                                                        svst1(pg, crow1, acc110);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc210);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc310);   crow3 += simd_width;
                                                if (nr >= 12) {
                                                        svst1(pg, crow0, acc011); crow0 += 4;
                                                        svst1(pg, crow1, acc111);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc211);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc311);   crow3 += simd_width;
                                                if (nr >= 13) {
                                                        svst1(pg, crow0, acc012); crow0 += 4;
                                                        svst1(pg, crow1, acc112);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc212);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc312);   crow3 += simd_width;
                                                if (nr >= 14) {
                                                        svst1(pg, crow0, acc013); crow0 += 4;
                                                        svst1(pg, crow1, acc113);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc213);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc313);   crow3 += simd_width;
                                                if (nr >= 15) {
                                                        svst1(pg, crow0, acc014); crow0 += 4;
                                                        svst1(pg, crow1, acc114);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc214);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc314);   crow3 += simd_width;
                                                if (nr >= 16) {
                                                        svst1(pg, crow0, acc015);
                                                        svst1(pg, crow1, acc115);
                                                        svst1(pg, crow2, acc215);
                                                        svst1(pg, crow3, acc315);
                                       } }}}}}}}}}}}}}}
                               }
                       }

                        break;
                }
                case 9:
                {
                        //printf("I am going in sve");
                        do {
                                for(int i1=0;i1<4;i1+=4)
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,4);
                                const svfloat32_t a0 = svld1(pg, a); a += 4;
                                const svfloat32_t a1 = svld1(pg, a); a += 4;
                                const svfloat32_t a2 = svld1(pg, a); a += 4;
                                const svfloat32_t a3 = svld1(pg, a); a += 4;
                                const svfloat32_t a4 = svld1(pg, a); a += 4;
                                const svfloat32_t a5 = svld1(pg, a); a += 4;
                                const svfloat32_t a6 = svld1(pg, a); a += 4;
                                const svfloat32_t a7 = svld1(pg, a); a += 4;
                                const svfloat32_t a8 = svld1(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                acc40 = svmla_m(pg, acc40, a4, b0);
                                acc50 = svmla_m(pg, acc50, a5, b0);
                                acc60 = svmla_m(pg, acc60, a6, b0);
                                acc70 = svmla_m(pg, acc70, a7, b0);
                                acc80 = svmla_m(pg, acc80, a8, b0);

                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);
                                acc31 = svmla_m(pg, acc31, a3, b1);
                                acc41 = svmla_m(pg, acc41, a4, b1);
                                acc51 = svmla_m(pg, acc51, a5, b1);
                                acc61 = svmla_m(pg, acc61, a6, b1);
                                acc71 = svmla_m(pg, acc71, a7, b1);
                                acc81 = svmla_m(pg, acc81, a8, b1);
                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                acc22 = svmla_m(pg, acc22, a2, b2);
                                acc32 = svmla_m(pg, acc32, a3, b2);
                                acc42 = svmla_m(pg, acc42, a4, b2);
                                acc52 = svmla_m(pg, acc52, a5, b2);
                                acc62 = svmla_m(pg, acc62, a6, b2);
                                acc72 = svmla_m(pg, acc72, a7, b2);
                                acc82 = svmla_m(pg, acc82, a8, b2);
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);
                                acc23 = svmla_m(pg, acc23, a2, b3);
                                acc33 = svmla_m(pg, acc33, a3, b3);
                                acc43 = svmla_m(pg, acc43, a4, b3);
                                acc53 = svmla_m(pg, acc53, a5, b3);
                                acc63 = svmla_m(pg, acc63, a6, b3);
                                acc73 = svmla_m(pg, acc73, a7, b3);
                                acc83 = svmla_m(pg, acc83, a8, b3);
                                const svfloat32_t b4 = svld1(pg, b4_ptr); b4_ptr += b_increment;
                                acc04 = svmla_m(pg, acc04, a0, b4);
                                acc14 = svmla_m(pg, acc14, a1, b4);
                                acc24 = svmla_m(pg, acc24, a2, b4);
                                acc34 = svmla_m(pg, acc34, a3, b4);
                                acc44 = svmla_m(pg, acc44, a4, b4);
                                acc54 = svmla_m(pg, acc54, a5, b4);
                                acc64 = svmla_m(pg, acc64, a6, b4);
                                acc74 = svmla_m(pg, acc74, a7, b4);
                                acc84 = svmla_m(pg, acc84, a8, b4);
                                const svfloat32_t b5 = svld1(pg, b5_ptr); b5_ptr += b_increment;
                                acc05 = svmla_m(pg, acc05, a0, b5);
                                acc15 = svmla_m(pg, acc15, a1, b5);
                                acc25 = svmla_m(pg, acc25, a2, b5);
                                acc35 = svmla_m(pg, acc35, a3, b5);
                                acc45 = svmla_m(pg, acc45, a4, b5);
                                acc55 = svmla_m(pg, acc55, a5, b5);
                                acc65 = svmla_m(pg, acc65, a6, b5);
                                acc75 = svmla_m(pg, acc75, a7, b5);
                                acc85 = svmla_m(pg, acc85, a8, b5);
                                const svfloat32_t b6 = svld1(pg, b6_ptr); b6_ptr += b_increment;
                                acc06 = svmla_m(pg, acc06, a0, b6);
                               acc16 = svmla_m(pg, acc16, a1, b6);
                                acc26 = svmla_m(pg, acc26, a2, b6);
                                acc36 = svmla_m(pg, acc36, a3, b6);
                                acc46 = svmla_m(pg, acc46, a4, b6);
                                acc56 = svmla_m(pg, acc56, a5, b6);
                                acc66 = svmla_m(pg, acc66, a6, b6);
                                acc76 = svmla_m(pg, acc76, a7, b6);
                                acc86 = svmla_m(pg, acc86, a8, b6);
                                const svfloat32_t b7 = svld1(pg, b7_ptr); b7_ptr += b_increment;
                                acc07 = svmla_m(pg, acc07, a0, b7);
                                acc17 = svmla_m(pg, acc17, a1, b7);
                                acc27 = svmla_m(pg, acc27, a2, b7);
                                acc37 = svmla_m(pg, acc37, a3, b7);
                                acc47 = svmla_m(pg, acc47, a4, b7);
                                acc57 = svmla_m(pg, acc57, a5, b7);
                                acc67 = svmla_m(pg, acc67, a6, b7);
                                acc77 = svmla_m(pg, acc77, a7, b7);
                                acc87 = svmla_m(pg, acc87, a8, b7);
                                ///
                                const svfloat32_t b8 = svld1(pg, b8_ptr); b8_ptr += b_increment;
                                acc08 = svmla_m(pg, acc08, a0, b8);
                                acc18 = svmla_m(pg, acc18, a1, b8);
                                acc28 = svmla_m(pg, acc28, a2, b8);
                                acc38 = svmla_m(pg, acc38, a3, b8);
                                acc48 = svmla_m(pg, acc48, a4, b8);
                                acc58 = svmla_m(pg, acc58, a5, b8);
                                acc68 = svmla_m(pg, acc68, a6, b8);
                                acc78 = svmla_m(pg, acc78, a7, b8);
                                acc88 = svmla_m(pg, acc88, a8, b8);
                                const svfloat32_t b9 = svld1(pg, b9_ptr); b9_ptr += b_increment;
                                acc09 = svmla_m(pg, acc09, a0, b9);
                                acc19 = svmla_m(pg, acc19, a1, b9);
                                acc29 = svmla_m(pg, acc29, a2, b9);
                                acc39 = svmla_m(pg, acc39, a3, b9);
                                acc49 = svmla_m(pg, acc49, a4, b9);
                                acc59 = svmla_m(pg, acc59, a5, b9);
                                acc69 = svmla_m(pg, acc69, a6, b9);
                                acc79 = svmla_m(pg, acc79, a7, b9);
                                acc89 = svmla_m(pg, acc89, a8, b9);
                               const svfloat32_t b10 = svld1(pg, b10_ptr); b10_ptr += b_increment;
                                acc010 = svmla_m(pg, acc010, a0, b10);
                                acc110 = svmla_m(pg, acc110, a1, b10);
                                acc210 = svmla_m(pg, acc210, a2, b10);
                                acc310 = svmla_m(pg, acc310, a3, b10);
                                acc410 = svmla_m(pg, acc410, a4, b10);
                                acc510 = svmla_m(pg, acc510, a5, b10);
                                acc610 = svmla_m(pg, acc610, a6, b10);
                                acc710 = svmla_m(pg, acc710, a7, b10);
                                acc810 = svmla_m(pg, acc810, a8, b10);
                                const svfloat32_t b11 = svld1(pg, b11_ptr); b11_ptr += b_increment;
                                acc011 = svmla_m(pg, acc011, a0, b11);
                                acc111 = svmla_m(pg, acc111, a1, b11);
                                acc211 = svmla_m(pg, acc211, a2, b11);
                                acc311 = svmla_m(pg, acc311, a3, b11);
                                acc411 = svmla_m(pg, acc411, a4, b11);
                                acc511 = svmla_m(pg, acc511, a5, b11);
                                acc611 = svmla_m(pg, acc611, a6, b11);
                                acc711 = svmla_m(pg, acc711, a7, b11);
                                acc811 = svmla_m(pg, acc811, a8, b11);
                                const svfloat32_t b12 = svld1(pg, b12_ptr); b12_ptr += b_increment;
                                acc012 = svmla_m(pg, acc012, a0, b12);
                                acc112 = svmla_m(pg, acc112, a1, b12);
                                acc212 = svmla_m(pg, acc212, a2, b12);
                                acc312 = svmla_m(pg, acc312, a3, b12);
                                acc412 = svmla_m(pg, acc412, a4, b12);
                                acc512 = svmla_m(pg, acc512, a5, b12);
                                acc612 = svmla_m(pg, acc612, a6, b12);
                                acc712 = svmla_m(pg, acc712, a7, b12);
                                acc812 = svmla_m(pg, acc812, a8, b12);
                                const svfloat32_t b13 = svld1(pg, b13_ptr); b13_ptr += b_increment;
                               acc013 = svmla_m(pg, acc013, a0, b13);
                                acc113 = svmla_m(pg, acc113, a1, b13);
                                acc213 = svmla_m(pg, acc213, a2, b13);
                                acc313 = svmla_m(pg, acc313, a3, b13);
                                acc413 = svmla_m(pg, acc413, a4, b13);
                                acc513 = svmla_m(pg, acc513, a5, b13);
                                acc613 = svmla_m(pg, acc613, a6, b13);
                                acc713 = svmla_m(pg, acc713, a7, b13);
                                acc813 = svmla_m(pg, acc813, a8, b13);
                                const svfloat32_t b14 = svld1(pg, b14_ptr); b14_ptr += b_increment;
                                acc014 = svmla_m(pg, acc014, a0, b14);
                                acc114 = svmla_m(pg, acc114, a1, b14);
                                acc214 = svmla_m(pg, acc214, a2, b14);
                                acc314 = svmla_m(pg, acc314, a3, b14);
                                acc414 = svmla_m(pg, acc414, a4, b14);
                                acc514 = svmla_m(pg, acc514, a5, b14);
                                acc614 = svmla_m(pg, acc614, a6, b14);
                                acc714 = svmla_m(pg, acc714, a7, b14);
                                acc814 = svmla_m(pg, acc814, a8, b14);
                                const svfloat32_t b15 = svld1(pg, b15_ptr); b15_ptr += b_increment;
                                acc015 = svmla_m(pg, acc015, a0, b15);
                                acc115 = svmla_m(pg, acc115, a1, b15);
                                acc215 = svmla_m(pg, acc215, a2, b15);
                                acc315 = svmla_m(pg, acc315, a3, b15);
                                acc415 = svmla_m(pg, acc415, a4, b15);
                                acc515 = svmla_m(pg, acc515, a5, b15);
                                acc615 = svmla_m(pg, acc615, a6, b15);
                                acc715 = svmla_m(pg, acc715, a7, b15);
                                acc815 = svmla_m(pg, acc815, a8, b15);


                                }
                        } while (--k);
                          float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                          float* restrict crow4 = crow3 + row_stride_c;
                        float* restrict crow5 = crow4 + row_stride_c;
                        float* restrict crow6 = crow5 + row_stride_c;
                        float* restrict crow7 = crow6 + row_stride_c;
                        float* restrict crow8 = crow7 + row_stride_c;
                        for(int i1=0;i1<4;i1+=4)
                        {
                        svbool_t pg = svwhilelt_b32(i1,4);
                        if (update != 0) {
                        //      printf("in update not 0");
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); crow0 += simd_width;
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); crow1 += simd_width;
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20)); crow2 += simd_width;
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30)); crow3 += simd_width;
                               svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc40)); crow4 += simd_width;
                                svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc50)); crow5 += simd_width;
                                svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc60)); crow6 += simd_width;
                                svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc70)); crow7 += simd_width;
                                svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc80)); crow8 += simd_width;

                                if (nr >= 2) {
                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc01)); crow0 += simd_width;
                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc11)); crow1 += simd_width;
                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc21)); crow2 += simd_width;
                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc31)); crow3 += simd_width;
                                        svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc41)); crow4 += simd_width;
                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc51)); crow5 += simd_width;
                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc61)); crow6 += simd_width;
                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc71)); crow7 += simd_width;
                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc81)); crow8 += simd_width;

                                        if (nr >= 3) {
                                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc02)); crow0 += simd_width;
                                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc12)); crow1 += simd_width;
                                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc22));crow2 += simd_width;
                                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc32)); crow3 += simd_width;
                                                svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc42)); crow4 += simd_width;
                                                svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc52)); crow5 += simd_width;
                                                svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc62));crow6 += simd_width;
                                                svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc72)); crow7 += simd_width;
                                                svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc82)); crow8 += simd_width;
                                                if (nr >= 4) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc03)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc13)); crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc23));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc33)); crow3 += simd_width;
                                                        svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc43)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc53)); crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc63));crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc73)); crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc83)); crow8 += simd_width;
                                                if (nr >= 5) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc04));  crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc14));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc24));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc34)); crow3 += simd_width;
                                                        svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc44));  crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc54));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc64));crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc74)); crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc84)); crow8 += simd_width;
                                                if (nr >= 6) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc05)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc15)); crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc25));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc35)); crow3 += simd_width;
                                                        svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc45)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc55)); crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc65));crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc75)); crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc85)); crow8 += simd_width;
                                                                                             if (nr >= 7) {
                                                        svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc06)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc16)); crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc26));crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc36));  crow3 += simd_width;
                                                        svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc46)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc56)); crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc66));crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc76));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc86));  crow8 += simd_width;
                                                if (nr >= 8) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc07)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc17));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc27));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc37));  crow3 += simd_width;
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc47)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc57));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc67));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc77));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc87));  crow8 += simd_width;
                                                ////
                                                if (nr >= 9) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc08)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc18));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc28));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc38));  crow3 += simd_width;
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc48)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc58));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc68));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc78));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc88));  crow8 += simd_width;
                                                if (nr >= 10) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc09)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc19));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc29));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc39));  crow3 += simd_width;
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc49)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc59));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc69));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc79));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc89));  crow8 += simd_width;
                                                                              if (nr >= 11) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc010)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc110));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc210));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc310));  crow3 += simd_width;
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc410)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc510));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc610));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc710));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc810));  crow8 += simd_width;
                                                if (nr >= 12) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc011)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc111));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc211));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc311));  crow3 += simd_width;
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc411)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc511));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc611));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc711));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc811));  crow8 += simd_width;
                                                if (nr >= 13) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc012)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc112));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc212));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc312));  crow3 += simd_width;
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc412)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc512));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc612));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc712));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc812));  crow8 += simd_width;
                                                if (nr >= 14) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc013)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc113));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc213));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc313));  crow3 += simd_width;
                                                        svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc413));  crow4 += simd_width;
                                                         svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc513)); crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc613));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc713));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc813));  crow8 += simd_width;
                                                if (nr >= 15) {
                                                                                                     svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc014)); crow0 += simd_width;
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc114));  crow1 += simd_width;
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc214));  crow2 += simd_width;
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc314));  crow3 += simd_width;
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc414)); crow4 += simd_width;
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc514));  crow5 += simd_width;
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc614));  crow6 += simd_width;
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc714));  crow7 += simd_width;
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc814));  crow8 += simd_width;
                                                if (nr >= 16) {
                                                         svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc015));
                                                        svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc115));
                                                        svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc215));
                                                        svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc315));
                                                         svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc415));
                                                        svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc515));
                                                        svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc615));
                                                        svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc715));
                                                        svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc815));
                                                        }
                                        }}}}}}}}}}}}}
                                }
                        }else {
                        //      printf("in update  0");
                                svst1(pg, crow0, acc00); crow0 += 4;
                                svst1(pg, crow1, acc10); crow1 += 4;
                                svst1(pg, crow2, acc20); crow2 += 4;
                                svst1(pg, crow3, acc30); crow3 += 4;
                                svst1(pg, crow4, acc40); crow4 += 4;
                                svst1(pg, crow5, acc50); crow5 += 4;
                                svst1(pg, crow6, acc60); crow6 += 4;
                                svst1(pg, crow7, acc70); crow7 += 4;
                                svst1(pg, crow8, acc80); crow8 += 4;
                               if (nr >= 2) {
                                        svst1(pg, crow0, acc01); crow0 += 4;
                                       svst1(pg, crow1, acc11); crow1 += simd_width;
                                        svst1(pg, crow2, acc21); crow2 += simd_width;
                                        svst1(pg, crow3, acc31); crow3 += simd_width;
                                        svst1(pg, crow4, acc41); crow4 += 4;
                                       svst1(pg, crow5, acc51); crow5 += simd_width;
                                        svst1(pg, crow6, acc61); crow6 += simd_width;
                                        svst1(pg, crow7, acc71); crow7 += simd_width;
                                        svst1(pg, crow8, acc81); crow8 += simd_width;
                                        if (nr >= 3) {
                                               svst1(pg, crow0, acc02); crow0 += 4;
                                                svst1(pg, crow1, acc12); crow1 += simd_width;
                                                svst1(pg, crow2, acc22); crow2 += simd_width;
                                                svst1(pg, crow3, acc32); crow3 += simd_width;
                                               svst1(pg, crow4, acc42); crow4 += 4;
                                                svst1(pg, crow5, acc52); crow5 += simd_width;
                                                svst1(pg, crow6, acc62); crow6 += simd_width;
                                                svst1(pg, crow7, acc72); crow7 += simd_width;
                                                svst1(pg, crow8, acc82); crow8 += simd_width;
                                                if (nr >= 4) {
                                                        svst1(pg, crow0, acc03);  crow0 += 4;
                                                        svst1(pg, crow1, acc13); crow1 += simd_width;
                                                        svst1(pg, crow2, acc23); crow2 += simd_width;
                                                        svst1(pg, crow3, acc33); crow3 += simd_width;
                                                        svst1(pg, crow4, acc43);  crow4 += 4;
                                                        svst1(pg, crow5, acc53); crow5 += simd_width;
                                                        svst1(pg, crow6, acc63); crow6 += simd_width;
                                                        svst1(pg, crow7, acc73); crow7 += simd_width;
                                                        svst1(pg, crow8, acc83); crow8 += simd_width;
                                                if (nr >= 5) {
                                                        svst1(pg, crow0, acc04);  crow0 += 4;
                                                        svst1(pg, crow1, acc14); crow1 += simd_width;
                                                        svst1(pg, crow2, acc24); crow2 += simd_width;
                                                        svst1(pg, crow3, acc34); crow3 += simd_width;
                                                        svst1(pg, crow4, acc44);  crow4 += 4;
                                                        svst1(pg, crow5, acc54); crow5 += simd_width;
                                                        svst1(pg, crow6, acc64); crow6 += simd_width;
                                                        svst1(pg, crow7, acc74); crow7 += simd_width;
                                                        svst1(pg, crow8, acc84); crow8 += simd_width;
                                                                                               if (nr >= 6) {
                                                        svst1(pg, crow0, acc05);  crow0 += 4;
                                                        svst1(pg, crow1, acc15); crow1 += simd_width;
                                                        svst1(pg, crow2, acc25); crow2 += simd_width;
                                                        svst1(pg, crow3, acc35); crow3 += simd_width;
                                                        svst1(pg, crow4, acc45);  crow4 += 4;
                                                        svst1(pg, crow5, acc55); crow5 += simd_width;
                                                        svst1(pg, crow6, acc65); crow6 += simd_width;
                                                        svst1(pg, crow7, acc75); crow7 += simd_width;
                                                        svst1(pg, crow8, acc85); crow8 += simd_width;
                                                if (nr >= 7) {
                                                        svst1(pg, crow0, acc06);  crow0 += 4;
                                                        svst1(pg, crow1, acc16); crow1 += simd_width;
                                                        svst1(pg, crow2, acc26); crow2 += simd_width;
                                                        svst1(pg, crow3, acc36); crow3 += simd_width;
                                                        svst1(pg, crow4, acc46);  crow4 += 4;
                                                        svst1(pg, crow5, acc56); crow5 += simd_width;
                                                        svst1(pg, crow6, acc66); crow6 += simd_width;
                                                        svst1(pg, crow7, acc76); crow7 += simd_width;
                                                        svst1(pg, crow8, acc86); crow8 += simd_width;
                                                if (nr >= 8) {
                                                        svst1(pg, crow0, acc07); crow0 += 4;
                                                        svst1(pg, crow1, acc17);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc27);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc37);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc47); crow4 += 4;
                                                        svst1(pg, crow5, acc57);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc67);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc77);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc87);   crow8 += simd_width;
                                                if (nr >= 9) {
                                                        svst1(pg, crow0, acc08); crow0 += 4;
                                                        svst1(pg, crow1, acc18);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc28);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc38);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc48); crow4 += 4;
                                                        svst1(pg, crow5, acc58);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc68);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc78);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc88);   crow8 += simd_width;
                                                if (nr >= 10) {
                                                                                          svst1(pg, crow0, acc09); crow0 += 4;
                                                        svst1(pg, crow1, acc19);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc29);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc39);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc49); crow4 += 4;
                                                        svst1(pg, crow5, acc59);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc69);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc79);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc89);   crow8 += simd_width;
                                                if (nr >= 11) {
                                                        svst1(pg, crow0, acc010); crow0 += 4;
                                                        svst1(pg, crow1, acc110);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc210);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc310);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc410); crow4 += 4;
                                                        svst1(pg, crow5, acc510);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc610);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc710);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc810);   crow8 += simd_width;
                                                if (nr >= 12) {
                                                        svst1(pg, crow0, acc011); crow0 += 4;
                                                        svst1(pg, crow1, acc111);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc211);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc311);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc411); crow4 += 4;
                                                        svst1(pg, crow5, acc511);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc611);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc711);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc811);   crow8 += simd_width;
                                                if (nr >= 13) {
                                                        svst1(pg, crow0, acc012); crow0 += 4;
                                                        svst1(pg, crow1, acc112);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc212);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc312);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc412); crow4 += 4;
                                                        svst1(pg, crow5, acc512);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc612);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc712);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc812);   crow8 += simd_width;
                                               if (nr >= 14) {
                                                        svst1(pg, crow0, acc013); crow0 += 4;
                                                        svst1(pg, crow1, acc113);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc213);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc313);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc413); crow4 += 4;
                                                        svst1(pg, crow5, acc513);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc613);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc713);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc813);   crow8 += simd_width;
                                                if (nr >= 15) {
                                                        svst1(pg, crow0, acc014); crow0 += 4;
                                                        svst1(pg, crow1, acc114);  crow1 += simd_width;
                                                        svst1(pg, crow2, acc214);   crow2 += simd_width;
                                                        svst1(pg, crow3, acc314);   crow3 += simd_width;
                                                        svst1(pg, crow4, acc414); crow4 += 4;
                                                        svst1(pg, crow5, acc514);  crow5 += simd_width;
                                                        svst1(pg, crow6, acc614);   crow6 += simd_width;
                                                        svst1(pg, crow7, acc714);   crow7 += simd_width;
                                                        svst1(pg, crow8, acc814);   crow8 += simd_width;
                                                if (nr >= 16) {
                                                        svst1(pg, crow0, acc015);
                                                        svst1(pg, crow1, acc115);
                                                        svst1(pg, crow2, acc215);
                                                        svst1(pg, crow3, acc315);
                                                        svst1(pg, crow4, acc415);
                                                        svst1(pg, crow5, acc515);
                                                        svst1(pg, crow6, acc615);
                                                        svst1(pg, crow7, acc715);
                                                        svst1(pg, crow8, acc815);
                                       } }}}}}}}}}}}}}}
                        }
                        }
                        break;
                }
                default:
                        printf("not an option");
        }
}
*/

void nnp_s4gemm_upto_3x3__sve(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	int simd_width=nnp_hwinfo.simd_width;

        svfloat32_t acc00 = svdup_f32(0.0f), acc01 = svdup_f32(0.0f), acc02 = svdup_f32(0.0f), acc03 = svdup_f32(0.0f);
        svfloat32_t acc04 = svdup_f32(0.0f), acc05 = svdup_f32(0.0f), acc06 = svdup_f32(0.0f), acc07 = svdup_f32(0.0f);
        svfloat32_t acc10 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f), acc12 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f);
        svfloat32_t acc14 = svdup_f32(0.0f), acc15 = svdup_f32(0.0f), acc16 = svdup_f32(0.0f), acc17 = svdup_f32(0.0f);
        svfloat32_t acc20 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f), acc22 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f);
        svfloat32_t acc24 = svdup_f32(0.0f), acc25 = svdup_f32(0.0f), acc26 = svdup_f32(0.0f), acc27 = svdup_f32(0.0f);
        svfloat32_t acc30 = svdup_f32(0.0f), acc31 = svdup_f32(0.0f), acc32 = svdup_f32(0.0f), acc33 = svdup_f32(0.0f);
        svfloat32_t acc34 = svdup_f32(0.0f), acc35 = svdup_f32(0.0f), acc36 = svdup_f32(0.0f), acc37 = svdup_f32(0.0f);
        svfloat32_t acc40 = svdup_f32(0.0f), acc50 = svdup_f32(0.0f), acc60 = svdup_f32(0.0f), acc70 = svdup_f32(0.0f), acc80 = svdup_f32(0.0f);
        svfloat32_t acc41 = svdup_f32(0.0f), acc51 = svdup_f32(0.0f), acc61 = svdup_f32(0.0f), acc71 = svdup_f32(0.0f), acc81 = svdup_f32(0.0f);
        svfloat32_t acc42 = svdup_f32(0.0f), acc52 = svdup_f32(0.0f), acc62 = svdup_f32(0.0f), acc72 = svdup_f32(0.0f), acc82 = svdup_f32(0.0f);
        svfloat32_t acc43 = svdup_f32(0.0f), acc53 = svdup_f32(0.0f), acc63 = svdup_f32(0.0f), acc73 = svdup_f32(0.0f), acc83 = svdup_f32(0.0f);
	int rem = 64 / svcntw();
	if(rem == 4){
        const float* b0_ptr = b;
        const float* b1_ptr =  b + 16;
        const float* b2_ptr =  b + 32;
        const float* b3_ptr =  b + 48;
        const size_t b_increment = nr * simd_width;
	switch (mr) {
                case 1:
                        do {
                                for(int i1=0;i1<16;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                svbool_t pg = svwhilelt_b32(i1,16);

                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a0, b1);
                                acc20 = svmla_m(pg, acc20, a0, b2);
                                acc30 = svmla_m(pg, acc30, a0, b3);
                                }
                        } while (--k);
                        for(int i1=0;i1<16;i1+=svcntw())
                        {
                       // printf("I am going in sve");
                        svbool_t pg = svwhilelt_b32(i1,16);

                        if (update != 0) {
                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc00));
                                svst1(pg, c+16, svadd_m(pg, svld1(pg, c+16), acc10));
                                svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc20));
                                svst1(pg, c+48, svadd_m(pg, svld1(pg, c+48), acc30));
                        } else {
                                svst1(pg, c, acc00);
                                svst1(pg, c+16, acc10);
                                svst1(pg, c+32, acc20);
                                svst1(pg, c+48, acc30);
                        }
			}
			 break;
                case 2:
                {
                        do {
                                for(int i1=0;i1<16;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,16);

                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);

                                }
                        } while (--k);
                        for(int i1=0;i1<16;i1+=svcntw())
                        {
                         //       printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,16);
                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); 
                                svst1(pg, crow0+16, svadd_m(pg, svld1(pg, crow0+16), acc01));
                                svst1(pg, crow1+16, svadd_m(pg, svld1(pg, crow1+16), acc11)); 
                                svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc02));
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc12)); 
                                svst1(pg, crow0+48, svadd_m(pg, svld1(pg, crow0+48), acc03));
                                svst1(pg, crow1+48, svadd_m(pg, svld1(pg, crow1+48), acc13)); 
                        } else {

                                svst1(pg, crow0, acc00); 
                                svst1(pg, crow1, acc10); 
                                svst1(pg, crow0+16, acc01); 
                                svst1(pg, crow1+16, acc11); 
                                svst1(pg, crow0+32, acc02); 
                                svst1(pg, crow1+32, acc12); 
                                svst1(pg, crow0+48, acc03); 
                                svst1(pg, crow1+48, acc13); 
                        }}
                        break;
                }
                case 3:
                {
                        do {
                                for(int i1=0;i1<16;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,16);
                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                acc22 = svmla_m(pg, acc22, a2, b2);
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);
                                acc23 = svmla_m(pg, acc23, a2, b3);

                                }
                        } while (--k);

                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        for(int i1=0;i1<16;i1+=svcntw())
                        {
                   //     printf("I am going in sve");
                        svbool_t pg = svwhilelt_b32(i1,16);
                        if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); 
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); 
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20)); 
                                svst1(pg, crow0+16, svadd_m(pg, svld1(pg, crow0+16), acc01)); 
                                svst1(pg, crow1+16, svadd_m(pg, svld1(pg, crow1+16), acc11)); 
                                svst1(pg, crow2+16, svadd_m(pg, svld1(pg, crow2+16), acc21)); 
                                svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc02)); 
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc12)); 
                                svst1(pg, crow2+32, svadd_m(pg, svld1(pg, crow2+32), acc22)); 
                                svst1(pg, crow0+48, svadd_m(pg, svld1(pg, crow0+48), acc03)); 
                                svst1(pg, crow1+48, svadd_m(pg, svld1(pg, crow1+48), acc13)); 
                                svst1(pg, crow2+48, svadd_m(pg, svld1(pg, crow2+48), acc23)); 
                        } else {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10); 
                                svst1(pg, crow2, acc20);
                                svst1(pg, crow0+16, acc01);
                                svst1(pg, crow1+16, acc11); 
                                svst1(pg, crow2+16, acc21);
                                svst1(pg, crow0+32, acc02);
                                svst1(pg, crow1+32, acc12); 
                                svst1(pg, crow2+32, acc22);
                                svst1(pg, crow0+48, acc03);
                                svst1(pg, crow1+48, acc13); 
                                svst1(pg, crow2+48, acc23);
                        }}
                        break;
		}
		 case 4:
                {
                        //printf("I am going in sve");
                        do {    
                                for(int i1=0;i1<16;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,16);
                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a3 = svld1rq(pg, a); a += 4;
                                svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);
                                acc31 = svmla_m(pg, acc31, a3, b1);
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                acc22 = svmla_m(pg, acc22, a2, b2);
                                acc32 = svmla_m(pg, acc32, a3, b2);
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);
                                acc23 = svmla_m(pg, acc23, a2, b3);
                                acc33 = svmla_m(pg, acc33, a3, b3);
                                
                                }
                        } while (--k);
			  float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                        for(int i1=0;i1<16;i1+=svcntw())
                        {
                        svbool_t pg = svwhilelt_b32(i1,16);
                        if (update != 0) {
			//	printf("in update not 0");
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); 
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); 
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20)); 
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30)); 
                               svst1(pg, crow0+16, svadd_m(pg, svld1(pg, crow0+16), acc01)); 
                                svst1(pg, crow1+16, svadd_m(pg, svld1(pg, crow1+16), acc11)); 
                                svst1(pg, crow2+16, svadd_m(pg, svld1(pg, crow2+16), acc21)); 
                                svst1(pg, crow3+16, svadd_m(pg, svld1(pg, crow3+16), acc31)); 
                               svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc02)); 
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc12)); 
                                svst1(pg, crow2+32, svadd_m(pg, svld1(pg, crow2+32), acc22)); 
                                svst1(pg, crow3+32, svadd_m(pg, svld1(pg, crow3+32), acc32)); 
                               svst1(pg, crow0+48, svadd_m(pg, svld1(pg, crow0+48), acc03)); 
                                svst1(pg, crow1+48, svadd_m(pg, svld1(pg, crow1+48), acc13)); 
                                svst1(pg, crow2+48, svadd_m(pg, svld1(pg, crow2+48), acc23)); 
                                svst1(pg, crow3+48, svadd_m(pg, svld1(pg, crow3+48), acc33)); 

                        } else {
			//	printf("in update  0");
                                svst1(pg, crow0, acc00); 
                                svst1(pg, crow1, acc10); 
                                svst1(pg, crow2, acc20); 
                                svst1(pg, crow3, acc30); 
                                svst1(pg, crow0+16, acc01); 
                                svst1(pg, crow1+16, acc11); 
                                svst1(pg, crow2+16, acc21); 
                                svst1(pg, crow3+16, acc31); 
                                svst1(pg, crow0+32, acc02); 
                                svst1(pg, crow1+32, acc12); 
                                svst1(pg, crow2+32, acc22); 
                                svst1(pg, crow3+32, acc32); 
                                svst1(pg, crow0+48, acc03); 
                                svst1(pg, crow1+48, acc13); 
                                svst1(pg, crow2+48, acc23); 
                                svst1(pg, crow3+48, acc33); 
                               }
			}	
                        break;
                }
		case 9:
                {
                        //printf("I am going in sve");
                        do {
                                for(int i1=0;i1<16;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,16);
                                //printf("I am going in sve");
                               const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a3 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a4 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a5 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a6 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a7 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a8 = svld1rq(pg, a); a += 4;
                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                const svfloat32_t b2 = svld1(pg, b2_ptr); b2_ptr += b_increment;
                                const svfloat32_t b3 = svld1(pg, b3_ptr); b3_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                acc40 = svmla_m(pg, acc40, a4, b0);
                                acc50 = svmla_m(pg, acc50, a5, b0);
                                acc60 = svmla_m(pg, acc60, a6, b0);
                                acc70 = svmla_m(pg, acc70, a7, b0);
                                acc80 = svmla_m(pg, acc80, a8, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);
                                acc31 = svmla_m(pg, acc31, a3, b1);
                                acc41 = svmla_m(pg, acc41, a4, b1);
                                acc51 = svmla_m(pg, acc51, a5, b1);
                                acc61 = svmla_m(pg, acc61, a6, b1);
                                acc71 = svmla_m(pg, acc71, a7, b1);
                                acc81 = svmla_m(pg, acc81, a8, b1);
                                acc02 = svmla_m(pg, acc02, a0, b2);
                                acc12 = svmla_m(pg, acc12, a1, b2);
                                acc22 = svmla_m(pg, acc22, a2, b2);
                                acc32 = svmla_m(pg, acc32, a3, b2);
                                acc42 = svmla_m(pg, acc42, a4, b2);
                                acc52 = svmla_m(pg, acc52, a5, b2);
                                acc62 = svmla_m(pg, acc62, a6, b2);
                                acc72 = svmla_m(pg, acc72, a7, b2);
                                acc82 = svmla_m(pg, acc82, a8, b2);
                                acc03 = svmla_m(pg, acc03, a0, b3);
                                acc13 = svmla_m(pg, acc13, a1, b3);
                                acc23 = svmla_m(pg, acc23, a2, b3);
                                acc33 = svmla_m(pg, acc33, a3, b3);
                                acc43 = svmla_m(pg, acc43, a4, b3);
                                acc53 = svmla_m(pg, acc53, a5, b3);
                                acc63 = svmla_m(pg, acc63, a6, b3);
                                acc73 = svmla_m(pg, acc73, a7, b3);
                                acc83 = svmla_m(pg, acc83, a8, b3);
				 }
                        } while (--k);
                          float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                          float* restrict crow4 = crow3 + row_stride_c;
                        float* restrict crow5 = crow4 + row_stride_c;
                        float* restrict crow6 = crow5 + row_stride_c;
                        float* restrict crow7 = crow6 + row_stride_c;
                        float* restrict crow8 = crow7 + row_stride_c;
		         for(int i1=0;i1<16;i1+=(svcntw()))
                         {
                                        svbool_t pg = svwhilelt_b32(i1, 16);
                        if (update != 0) {
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10));
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20));
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30));
                               svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc40));
                                svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc50));
                                svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc60));
                                svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc70));
                                svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc80));
                               svst1(pg, crow0+16, svadd_m(pg, svld1(pg, crow0+16), acc01));
                                svst1(pg, crow1+16, svadd_m(pg, svld1(pg, crow1+16), acc11));
                                svst1(pg, crow2+16, svadd_m(pg, svld1(pg, crow2+16), acc21));
                                svst1(pg, crow3+16, svadd_m(pg, svld1(pg, crow3+16), acc31));
                               svst1(pg, crow4+16, svadd_m(pg, svld1(pg, crow4+16), acc41));
                                svst1(pg, crow5+16, svadd_m(pg, svld1(pg, crow5+16), acc51));
                                svst1(pg, crow6+16, svadd_m(pg, svld1(pg, crow6+16), acc61));
                                svst1(pg, crow7+16, svadd_m(pg, svld1(pg, crow7+16), acc71));
                                svst1(pg, crow8+16, svadd_m(pg, svld1(pg, crow8+16), acc81));
                               svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc02));
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc12));
                                svst1(pg, crow2+32, svadd_m(pg, svld1(pg, crow2+32), acc22));
                                svst1(pg, crow3+32, svadd_m(pg, svld1(pg, crow3+32), acc32));
                               svst1(pg, crow4+32, svadd_m(pg, svld1(pg, crow4+32), acc42));
                                svst1(pg, crow5+32, svadd_m(pg, svld1(pg, crow5+32), acc52));
                                svst1(pg, crow6+32, svadd_m(pg, svld1(pg, crow6+32), acc62));
                                svst1(pg, crow7+32, svadd_m(pg, svld1(pg, crow7+32), acc72));
                                svst1(pg, crow8+32, svadd_m(pg, svld1(pg, crow8+32), acc82));
                               svst1(pg, crow0+48, svadd_m(pg, svld1(pg, crow0+48), acc03));
                                svst1(pg, crow1+48, svadd_m(pg, svld1(pg, crow1+48), acc13));
                                svst1(pg, crow2+48, svadd_m(pg, svld1(pg, crow2+48), acc23));
                                svst1(pg, crow3+48, svadd_m(pg, svld1(pg, crow3+48), acc33));
                               svst1(pg, crow4+48, svadd_m(pg, svld1(pg, crow4+48), acc43));
                                svst1(pg, crow5+48, svadd_m(pg, svld1(pg, crow5+48), acc53));
                                svst1(pg, crow6+48, svadd_m(pg, svld1(pg, crow6+48), acc63));
                                svst1(pg, crow7+48, svadd_m(pg, svld1(pg, crow7+48), acc73));
                                svst1(pg, crow8+48, svadd_m(pg, svld1(pg, crow8+48), acc83));
			}
			else
                        {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10);
                                svst1(pg, crow2, acc20);
                                svst1(pg, crow3, acc30);
                                svst1(pg, crow4, acc40);
                                svst1(pg, crow5, acc50);
                                svst1(pg, crow6, acc60);
                                svst1(pg, crow7, acc70);
                                svst1(pg, crow8, acc80);
                                svst1(pg, crow0+16, acc01);
                                svst1(pg, crow1+16, acc11);
                                svst1(pg, crow2+16, acc21);
                                svst1(pg, crow3+16, acc31);
                                svst1(pg, crow4+16, acc41);
                                svst1(pg, crow5+16, acc51);
                                svst1(pg, crow6+16, acc61);
                                svst1(pg, crow7+16, acc71);
                                svst1(pg, crow8+16, acc81);
                                svst1(pg, crow0+32, acc02);
                                svst1(pg, crow1+32, acc12);
                                svst1(pg, crow2+32, acc22);
                                svst1(pg, crow3+32, acc32);
                                svst1(pg, crow4+32, acc42);
                                svst1(pg, crow5+32, acc52);
                                svst1(pg, crow6+32, acc62);
                                svst1(pg, crow7+32, acc72);
                                svst1(pg, crow8+32, acc82);
                                svst1(pg, crow0+48, acc03);
                                svst1(pg, crow1+48, acc13);
                                svst1(pg, crow2+48, acc23);
                                svst1(pg, crow3+48, acc33);
                                svst1(pg, crow4+48, acc43);
                                svst1(pg, crow5+48, acc53);
                                svst1(pg, crow6+48, acc63);
                                svst1(pg, crow7+48, acc73);
                                svst1(pg, crow8+48, acc83);
			}
			}
			break;
		}

                default:
                        printf("not an option");
        }
	}
	else if(rem == 2){
        const float* b0_ptr = b;
        const float* b1_ptr =  b + 32;
        const size_t b_increment = nr * simd_width;
	switch (mr) {
                case 1:
                        do {
                                for(int i1=0;i1<32;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                svbool_t pg = svwhilelt_b32(i1,32);

                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a0, b1);

                                }
                        } while (--k);
                        for(int i1=0;i1<32;i1+=svcntw())
                        {
                       // printf("I am going in sve");
                        svbool_t pg = svwhilelt_b32(i1,32);

                        if (update != 0) {
                                svst1(pg, c, svadd_m(pg, svld1(pg, c), acc00));
                                svst1(pg, c+32, svadd_m(pg, svld1(pg, c+32), acc10));
                        } else {
                                svst1(pg, c, acc00);
                                svst1(pg, c+32, acc10);
                        }
			}
			 break;
                case 2:
                {
                        do {
                                for(int i1=0;i1<32;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,32);

                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);

                                }
                        } while (--k);
                        for(int i1=0;i1<32;i1+=svcntw())
                        {
                         //       printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,32);
                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); 
                                svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc01));
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc11)); 
                        } else {

                                svst1(pg, crow0, acc00); 
                                svst1(pg, crow1, acc10); 
                                svst1(pg, crow0+32, acc01); 
                                svst1(pg, crow1+32, acc11); 
                        }}
                        break;
                }
                case 3:
                {
                        do {
                                for(int i1=0;i1<32;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,32);
                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);

                                }
                        } while (--k);

                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        for(int i1=0;i1<32;i1+=svcntw())
                        {
                   //     printf("I am going in sve");
                        svbool_t pg = svwhilelt_b32(i1,32);
                        if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); 
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); 
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20)); 
                                svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc01)); 
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc11)); 
                                svst1(pg, crow2+32, svadd_m(pg, svld1(pg, crow2+32), acc21)); 
                        } else {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10); 
                                svst1(pg, crow2, acc20);
                                svst1(pg, crow0+32, acc01);
                                svst1(pg, crow1+32, acc11); 
                                svst1(pg, crow2+32, acc21);
                        }}
                        break;
		}
		 case 4:
                {
                        //printf("I am going in sve");
                        do {    
                                for(int i1=0;i1<32;i1+=svcntw())
                                {
                                //printf("I am going in sve");
                                 svbool_t pg = svwhilelt_b32(i1,32);
                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a3 = svld1rq(pg, a); a += 4;
                                svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);
                                acc31 = svmla_m(pg, acc31, a3, b1);
                                
                                }
                        } while (--k);
			  float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                        for(int i1=0;i1<32;i1+=svcntw())
                        {
                        svbool_t pg = svwhilelt_b32(i1,32);
                        if (update != 0) {
			//	printf("in update not 0");
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00)); 
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10)); 
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20)); 
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30)); 
                               svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc01)); 
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc11)); 
                                svst1(pg, crow2+32, svadd_m(pg, svld1(pg, crow2+32), acc21)); 
                                svst1(pg, crow3+32, svadd_m(pg, svld1(pg, crow3+32), acc31)); 

                        } else {
			//	printf("in update  0");
                                svst1(pg, crow0, acc00); 
                                svst1(pg, crow1, acc10); 
                                svst1(pg, crow2, acc20); 
                                svst1(pg, crow3, acc30); 
                                svst1(pg, crow0+32, acc01); 
                                svst1(pg, crow1+32, acc11); 
                                svst1(pg, crow2+32, acc21); 
                                svst1(pg, crow3+32, acc31); 
                               }
			}	
                        break;
                }
		case 9:
                {
                        //printf("I am going in sve");
                        do {
                                for(int i1=0;i1<32;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,32);
                                //printf("I am going in sve");
                               const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a3 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a4 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a5 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a6 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a7 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a8 = svld1rq(pg, a); a += 4;
                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                const svfloat32_t b1 = svld1(pg, b1_ptr); b1_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                acc40 = svmla_m(pg, acc40, a4, b0);
                                acc50 = svmla_m(pg, acc50, a5, b0);
                                acc60 = svmla_m(pg, acc60, a6, b0);
                                acc70 = svmla_m(pg, acc70, a7, b0);
                                acc80 = svmla_m(pg, acc80, a8, b0);
                                acc01 = svmla_m(pg, acc01, a0, b1);
                                acc11 = svmla_m(pg, acc11, a1, b1);
                                acc21 = svmla_m(pg, acc21, a2, b1);
                                acc31 = svmla_m(pg, acc31, a3, b1);
                                acc41 = svmla_m(pg, acc41, a4, b1);
                                acc51 = svmla_m(pg, acc51, a5, b1);
                                acc61 = svmla_m(pg, acc61, a6, b1);
                                acc71 = svmla_m(pg, acc71, a7, b1);
                                acc81 = svmla_m(pg, acc81, a8, b1);
				 }
                        } while (--k);
                          float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                          float* restrict crow4 = crow3 + row_stride_c;
                        float* restrict crow5 = crow4 + row_stride_c;
                        float* restrict crow6 = crow5 + row_stride_c;
                        float* restrict crow7 = crow6 + row_stride_c;
                        float* restrict crow8 = crow7 + row_stride_c;
		         for(int i1=0;i1<32;i1+=(svcntw()))
                         {
                                        svbool_t pg = svwhilelt_b32(i1, 32);
                        if (update != 0) {
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10));
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20));
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30));
                               svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc40));
                                svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc50));
                                svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc60));
                                svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc70));
                                svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc80));
                               svst1(pg, crow0+32, svadd_m(pg, svld1(pg, crow0+32), acc01));
                                svst1(pg, crow1+32, svadd_m(pg, svld1(pg, crow1+32), acc11));
                                svst1(pg, crow2+32, svadd_m(pg, svld1(pg, crow2+32), acc21));
                                svst1(pg, crow3+32, svadd_m(pg, svld1(pg, crow3+32), acc31));
                               svst1(pg, crow4+32, svadd_m(pg, svld1(pg, crow4+32), acc41));
                                svst1(pg, crow5+32, svadd_m(pg, svld1(pg, crow5+32), acc51));
                                svst1(pg, crow6+32, svadd_m(pg, svld1(pg, crow6+32), acc61));
                                svst1(pg, crow7+32, svadd_m(pg, svld1(pg, crow7+32), acc71));
                                svst1(pg, crow8+32, svadd_m(pg, svld1(pg, crow8+32), acc81));
			}
			else
                        {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10);
                                svst1(pg, crow2, acc20);
                                svst1(pg, crow3, acc30);
                                svst1(pg, crow4, acc40);
                                svst1(pg, crow5, acc50);
                                svst1(pg, crow6, acc60);
                                svst1(pg, crow7, acc70);
                                svst1(pg, crow8, acc80);
                                svst1(pg, crow0+32, acc01);
                                svst1(pg, crow1+32, acc11);
                                svst1(pg, crow2+32, acc21);
                                svst1(pg, crow3+32, acc31);
                                svst1(pg, crow4+32, acc41);
                                svst1(pg, crow5+32, acc51);
                                svst1(pg, crow6+32, acc61);
                                svst1(pg, crow7+32, acc71);
                                svst1(pg, crow8+32, acc81);
			}
			}
			break;
		}

                default:
                        printf("not an option");
        }
	}
	else if(rem ==1)
	{
        const float* b0_ptr = b;
		const size_t b_increment = nr * simd_width;
       switch (mr) {
                case 1:
                        do {    
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {       
                                        svbool_t pg = svwhilelt_b32(i1,64); 
                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                
                                
                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                
                                }
                        } while (--k);
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                        if (update != 0) {
                                        svst1(pg, c, svadd_m(pg, svld1(pg, c), acc00));
                        } else {
                                svst1(pg, c, acc00);
                        }}
                         break;
                case 2:
                {
                        do {
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;

                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);

                                }
                        } while (--k);
                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                        if (update != 0) {
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10));
                        } else {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10);
                        }}
                        break;
                }
                case 3:
                {
                        do {
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                                const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;
                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);

                                }
                        } while (--k);

                        float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                       if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10));
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20));
                        } else {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10);
                                svst1(pg, crow2, acc20);
                        }}
                        break;
                }
                 case 4:
                {
                        //printf("I am going in sve");
                        do {
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                               const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a3 = svld1rq(pg, a); a += 4;
                                svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                }
                        } while (--k);
                          float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                       if (update != 0) {
                                svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10));
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20));
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30));
                        } else {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10);
                                svst1(pg, crow2, acc20);
                                svst1(pg, crow3, acc30);
                        }}
                        break;
                }
                case 9:
                {
                        //printf("I am going in sve");
                        do {
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                               const svfloat32_t a0 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a1 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a2 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a3 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a4 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a5 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a6 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a7 = svld1rq(pg, a); a += 4;
                                const svfloat32_t a8 = svld1rq(pg, a); a += 4;
                                const svfloat32_t b0 = svld1(pg, b0_ptr); b0_ptr += b_increment;
                                acc00 = svmla_m(pg, acc00, a0, b0);
                                acc10 = svmla_m(pg, acc10, a1, b0);
                                acc20 = svmla_m(pg, acc20, a2, b0);
                                acc30 = svmla_m(pg, acc30, a3, b0);
                                acc40 = svmla_m(pg, acc40, a4, b0);
                                acc50 = svmla_m(pg, acc50, a5, b0);
                                acc60 = svmla_m(pg, acc60, a6, b0);
                                acc70 = svmla_m(pg, acc70, a7, b0);
                                acc80 = svmla_m(pg, acc80, a8, b0);

                                }
                        } while (--k);
                          float* restrict crow0 = c;
                        float* restrict crow1 = crow0 + row_stride_c;
                        float* restrict crow2 = crow1 + row_stride_c;
                        float* restrict crow3 = crow2 + row_stride_c;
                          float* restrict crow4 = crow3 + row_stride_c;
                        float* restrict crow5 = crow4 + row_stride_c;
                        float* restrict crow6 = crow5 + row_stride_c;
                        float* restrict crow7 = crow6 + row_stride_c;
                        float* restrict crow8 = crow7 + row_stride_c;
                                for(int i1=0;i1<64;i1+=(svcntw()))
                                {
                                        svbool_t pg = svwhilelt_b32(i1,64);
                        if (update != 0) {
                               svst1(pg, crow0, svadd_m(pg, svld1(pg, crow0), acc00));
                                svst1(pg, crow1, svadd_m(pg, svld1(pg, crow1), acc10));
                                svst1(pg, crow2, svadd_m(pg, svld1(pg, crow2), acc20));
                                svst1(pg, crow3, svadd_m(pg, svld1(pg, crow3), acc30));
                               svst1(pg, crow4, svadd_m(pg, svld1(pg, crow4), acc40));
                                svst1(pg, crow5, svadd_m(pg, svld1(pg, crow5), acc50));
                                svst1(pg, crow6, svadd_m(pg, svld1(pg, crow6), acc60));
                                svst1(pg, crow7, svadd_m(pg, svld1(pg, crow7), acc70));
                                svst1(pg, crow8, svadd_m(pg, svld1(pg, crow8), acc80));
                        }
                        else
                        {
                                svst1(pg, crow0, acc00);
                                svst1(pg, crow1, acc10);
                                svst1(pg, crow2, acc20);
                                svst1(pg, crow3, acc30);
                                svst1(pg, crow4, acc40);
                                svst1(pg, crow5, acc50);
                                svst1(pg, crow6, acc60);
                                svst1(pg, crow7, acc70);
                                svst1(pg, crow8, acc80);
                        }}
                        break;
                }
                default:
                        printf("not an option");
        }

	}
}

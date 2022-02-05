#include <stdio.h>
#include <matx.h>

int main(int argc, char **argv) {
    auto a = matx::make_tensor<float>({10});
    a.SetVals({1,2,3,4,5,6,7,8,9,10});

    printf("You should see the values 1-10 printed\n");
    a.Print();


  tensorShape_t<2> shape({2, 3});
//   using complex = cuda::std::complex<float>;
//   tensor_t<complex, 2> A(shape);
//   tensor_t<complex, 2> B(shape);
}
#include <immintrin.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace std;

// Evita que el compilador elimine por completo
// los calculos realizados durante el benchmark.
volatile float benchmark_sink = 0.0f;

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif


// ============================================================
// HORNER ESCALAR EN PRECISION SIMPLE
// ============================================================

NOINLINE float horner(float X, const float* coef, long size)
{
    float ACC = 0.0f;

    for (long i = 0; i < size; ++i)
    {
        ACC = (ACC + coef[i]) * X;
    }

    return ACC;
}


// ============================================================
// HORNER VECTORIAL UTILIZANDO AVX
//
// __m256 contiene 8 valores float:
//
// | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 |
//
// Los coeficientes deben estar alineados a 32 bytes porque
// se utiliza _mm256_load_ps().
// ============================================================

NOINLINE float horner_intrinsic(float X, const float* coef, long size)
{
    if (size <= 0)
    {
        return 0.0f;
    }

    constexpr long AVX_WIDTH = 8;

    // Numero de bloques completos de 8 floats
    const long blocks = size / AVX_WIDTH;


    // --------------------------------------------------------
    // Si hay menos de 8 coeficientes, no hay ningun bloque
    // completo que se pueda procesar con AVX.
    // --------------------------------------------------------

    if (blocks == 0)
    {
        return horner(X, coef, size);
    }


    // --------------------------------------------------------
    // Calculo de X^8
    //
    // Cada lane del vector trabaja con coeficientes separados
    // entre si por 8 posiciones, por lo que entre bloques
    // necesitamos multiplicar por X^8.
    // --------------------------------------------------------

    const float X2 = X * X;
    const float X4 = X2 * X2;
    const float X8_scalar = X4 * X4;

    const __m256 X8 = _mm256_set1_ps(X8_scalar);


    // Acumulador AVX:
    //
    // __m256 = 8 floats de 32 bits = 256 bits
    //
    __m256 Y = _mm256_setzero_ps();


    // ========================================================
    // PROCESAMIENTO DE BLOQUES DE 8 COEFICIENTES
    // ========================================================

    for (long b = 0; b < blocks; ++b)
    {
        // Carga 8 coeficientes:
        //
        // coef[8*b + 0]
        // coef[8*b + 1]
        // ...
        // coef[8*b + 7]
        //
        const __m256 C =
            _mm256_load_ps(coef + b * AVX_WIDTH);


        // Y = Y + C
        Y = _mm256_add_ps(Y, C);


        // Multiplicamos por X^8 entre bloques.
        //
        // No se hace despues del ultimo bloque porque
        // posteriormente cada lane se multiplicara por:
        //
        // X, X^2, ..., X^8
        //
        if (b + 1 < blocks)
        {
            Y = _mm256_mul_ps(Y, X8);
        }
    }


    // ========================================================
    // EXTRAER LAS 8 LANES
    // ========================================================

    alignas(32) float lanes[AVX_WIDTH];

    _mm256_store_ps(lanes, Y);


    // ========================================================
    // RECONSTRUIR EL RESULTADO ESCALAR
    //
    // El resultado vectorial representa:
    //
    // lane[7] * X
    // lane[6] * X^2
    // lane[5] * X^3
    // ...
    // lane[0] * X^8
    //
    // Esto generaliza la version original de 4 doubles
    // a 8 floats.
    // ========================================================

    float P = 0.0f;
    float power = X;

    for (long lane = AVX_WIDTH - 1; lane >= 0; --lane)
    {
        P += lanes[lane] * power;
        power *= X;
    }


    // ========================================================
    // TAIL
    //
    // Si size no es multiplo de 8:
    //
    // size = 10003
    //
    // AVX procesa:
    // 10000 coeficientes
    //
    // Tail:
    // coef[10000]
    // coef[10001]
    // coef[10002]
    //
    // Se continua Horner normalmente desde el resultado
    // calculado mediante AVX.
    // ========================================================

    for (long i = blocks * AVX_WIDTH; i < size; ++i)
    {
        P = (P + coef[i]) * X;
    }


    return P;
}


// ============================================================
// MAIN
// ============================================================

int main()
{
    constexpr long N_COEF = 10000;

    const int num_trials = 100000;


    // Precision simple
    float X = 1.1f;

    float R = 0.0f;


    srand(static_cast<unsigned int>(time(nullptr)));


    // ========================================================
    // RESERVA DE MEMORIA ALINEADA A 32 BYTES
    //
    // AVX trabaja con registros de:
    //
    // 256 bits = 32 bytes
    //
    // 8 floats x 4 bytes = 32 bytes
    // ========================================================

    float* coeficientes =
        static_cast<float*>(
            _mm_malloc(
                N_COEF * sizeof(float),
                32
            )
        );


    if (coeficientes == nullptr)
    {
        cerr << "Error: no se pudo reservar memoria."
             << endl;

        return 1;
    }


    // ========================================================
    // GENERAR COEFICIENTES
    // ========================================================

    for (long i = 0; i < N_COEF; ++i)
    {
        coeficientes[i] =
            static_cast<float>(rand() % 1000)
            / 1000.0f;


        if (i < 10)
        {
            cout << coeficientes[i] << '\n';
        }
    }


    // ========================================================
    // COMPROBAR AMBAS IMPLEMENTACIONES
    // ========================================================

    const float R_scalar =
        horner(
            X,
            coeficientes,
            N_COEF
        );


    const float R_avx =
        horner_intrinsic(
            X,
            coeficientes,
            N_COEF
        );


    cout << "Horner escalar: "
         << R_scalar
         << '\n';

    cout << "Horner AVX:     "
         << R_avx
         << '\n';


    // ========================================================
    // PROTECCION CONTRA OPTIMIZACION DEL BENCHMARK
    //
    // El valor se lee desde volatile en cada iteracion.
    // Esto impide que el compilador simplemente calcule la
    // funcion una sola vez y saque el calculo fuera del loop.
    // ========================================================

    volatile float X_benchmark = X;


    using clock_type = chrono::steady_clock;


    // ========================================================
    // BENCHMARK HORNER ESCALAR
    // ========================================================

    float checksum_scalar = 0.0f;


    auto start = clock_type::now();


    for (int j = 0; j < num_trials; ++j)
    {
        R = horner(
            X_benchmark,
            coeficientes,
            N_COEF
        );

        checksum_scalar += R;
    }


    auto end = clock_type::now();


    // Evita que el compilador descarte los resultados.
    benchmark_sink = checksum_scalar;


    const chrono::duration<float> scalar_elapsed =
        end - start;


    const float scalar_avg_ns =
        chrono::duration<float, nano>(
            end - start
        ).count()
        / num_trials;


    cout << fixed << setprecision(9);


    cout << "Tiempo total Horner escalar: "
         << scalar_elapsed.count()
         << " s\n";


    cout << "Tiempo promedio por ejecucion: "
         << scalar_avg_ns
         << " ns\n";


    // ========================================================
    // BENCHMARK HORNER AVX
    // ========================================================

    float checksum_avx = 0.0f;


    start = clock_type::now();


    for (int j = 0; j < num_trials; ++j)
    {
        R = horner_intrinsic(
            X_benchmark,
            coeficientes,
            N_COEF
        );

        checksum_avx += R;
    }


    end = clock_type::now();


    benchmark_sink = checksum_avx;


    const chrono::duration<float> avx_elapsed =
        end - start;


    const float avx_avg_ns =
        chrono::duration<float, nano>(
            end - start
        ).count()
        / num_trials;


    cout << "Tiempo total Horner AVX:     "
         << avx_elapsed.count()
         << " s\n";


    cout << "Tiempo promedio por ejecucion: "
         << avx_avg_ns
         << " ns\n";


    // ========================================================
    // LIBERAR MEMORIA
    // ========================================================

    _mm_free(coeficientes);


    return 0;
}
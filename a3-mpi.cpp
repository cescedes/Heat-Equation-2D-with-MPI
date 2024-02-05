#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <mpi.h>

#include "a3-helpers.hpp"

using namespace std;

int main(int argc, char **argv)
{
    int max_iterations = 1000;
    double epsilon = 1.0e-3;
    bool verify = true, print_config = false;

    // default values for M rows and N columns
    int N = 12;
    int M = 12;
    
    int i, j;
    double diffnorm;
    int iteration_count = 0;

    MPI_Init(&argc, &argv); 

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    process_input(argc, argv, N, M, max_iterations, epsilon, verify, print_config);

    if (rank == 0 && print_config)
        std::cout << "Configuration: m: " << M << ", n: " << N << ", max-iterations: " << max_iterations << ", epsilon: " << epsilon << std::endl;

    double time_1 = MPI_Wtime();

    // calculate rows per process and adjust for M % size != 0
    int rows_per_process = M / size;
    int extra_rows = M % size;
    int local_M = rows_per_process + (rank < extra_rows ? 1 : 0);
    int start_row = rank * rows_per_process + min(rank, extra_rows);
    
    // adjust for halo rows based on the rank
    int local_M_with_halo = local_M + ((rank == 0 || rank == size-1) ? 1 : 2); 
    // all processes should have 2 additional rows for halos, except for the processes at the boundaries which need only 1 extra row.
    Mat U(local_M_with_halo, N), W(local_M_with_halo, N);

    // Init & Boundary
    for (i = 0; i < local_M_with_halo; ++i) {
        int global_i = start_row + i - (rank == 0 ? 0 : 1);
        for (j = 0; j < N; ++j) {
            W[i][j] = U[i][j] = 0.0;
            // left and right boundaries
            if (j == 0) W[i][j] = U[i][j] = 0.05;
            if (j == N-1) W[i][j] = U[i][j] = 0.1;
        }
        // top and bottom boundaries if at global top or bottom
        if (global_i == 0) for (j = 0; j < N; ++j) W[i][j] = U[i][j] = 0.02;
        if (global_i == M-1) for (j = 0; j < N; ++j) W[i][j] = U[i][j] = 0.2;
    }
    // End init

    iteration_count = 0;
    do
    {
        iteration_count++;
        diffnorm = 0.0;

        // Compute new values (but not on boundary) 
        for (i = 1; i < local_M_with_halo - 1; ++i)
        {
            for (j = 1; j < N - 1; ++j)
            {
                W[i][j] = (U[i][j + 1] + U[i][j - 1] + U[i + 1][j] + U[i - 1][j]) * 0.25;
                diffnorm += (W[i][j] - U[i][j]) * (W[i][j] - U[i][j]);
            }
        }

        if (rank > 0) {
            // send the first computational row to the previous rank and receive into the top halo row
            MPI_Sendrecv(&U[1][0], N, MPI_DOUBLE, rank - 1, 0,
                        &U[0][0], N, MPI_DOUBLE, rank - 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank < size - 1) {
            // send the last computational row to the next rank and receive into the bottom halo row
            MPI_Sendrecv(&U[local_M - 1][0], N, MPI_DOUBLE, rank + 1, 0,
                        &U[local_M_with_halo - 1][0], N, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        // Only transfer the interior points
        for (i = 1; i < local_M_with_halo - 1; ++i)
            for (j = 1; j < N - 1; ++j)
                U[i][j] = W[i][j];

        
        double global_diffnorm;
        MPI_Allreduce(&diffnorm, &global_diffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        diffnorm = sqrt(global_diffnorm); // all processes need to know when to stop

    } while (epsilon <= diffnorm && iteration_count < max_iterations);
    
    double time_2 = MPI_Wtime();

    // ensure all processes finish computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    double time_3 = MPI_Wtime();

    Mat bigU(M, N); // allocate memory for the big matrix

    // prepare counts and displacements for Gatherv
    int* recvcounts = new int[size];
    int* displs = new int[size];

    int sum = 0;
    for (int r = 0; r < size; ++r) {
        int rows = M / size + (r < extra_rows ? 1 : 0);
        recvcounts[r] = rows * N; // each process contributes its rows * N elements
        displs[r] = sum;
        sum += recvcounts[r];
    }
    int local_send_count = (local_M_with_halo - 2) * N; // exclude top and bottom halo for inner processes
    if (rank == 0 || rank == size - 1) {
        local_send_count += N; // include one halo row for edge processes
    }

    // gather all local matrices into bigU at rank 0
    MPI_Gatherv(&(U[1][0]), local_send_count, MPI_DOUBLE,
            rank == 0 ? &(bigU[0][0]) : NULL, recvcounts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

    double time_4 = MPI_Wtime();

    // Print time measurements 
    if (rank == 0) {
        cout << " Execution: "; 
        cout << std::fixed << std::setprecision(4) << (time_2 - time_1);
        cout << " seconds, iterations: " << iteration_count << endl; 
        cout << " Collecting data: "; 
        cout << std::fixed << std::setprecision(4) << (time_4 - time_3);
        cout << " seconds" << endl; 
    }
    // verification     
    if (rank == 0 & verify) {
        Mat U_sequential(M, N); // init another matrix for the verification

        int iteration_count_seq = 0;
        heat2d_sequential(U_sequential, max_iterations, epsilon, iteration_count_seq); 

        // Here we need both results - from the sequential (U_sequential) and also from the MPI version, then we compare them with the compare(...) function 
        cout << "Verification: " << (bigU.compare(U_sequential) && iteration_count == iteration_count_seq ? "OK" : "NOT OK") << std::endl;

    }

    delete[] recvcounts;
    delete[] displs;

    MPI_Finalize(); 

    return 0;
}
#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <chrono>

constexpr size_t SIZE = 1000;
constexpr int MASTER = 0;              
constexpr int FROM_MASTER = 1;         
constexpr int FROM_WORKER = 2;         

int	numtasks,                          
    process,                           
    numworkers,                        
    source,                            
    dest,                              
    rows,                              
    averow, extra, offset,             
    i, j, k, rc;                       

MPI_Status status;

double** CreateMatrix(const double& val = double())
{
    double** ptr = nullptr;
    double* pool = nullptr;
    
    ptr = new double* [SIZE];  
    pool = new double[SIZE * SIZE]{ val };  

    
    for (size_t i = 0; i < SIZE; ++i, pool += SIZE)
        ptr[i] = pool;

    return ptr;
}

void MultiplyMatricesParallel(double**& matrix1, double**& matrix2, double**& result)
{

    MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
    MPI_Recv(*matrix1, rows * SIZE, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
    MPI_Bcast(*matrix2, SIZE * SIZE, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);


    for (k = 0; k < SIZE; ++k)
    {
        for (i = 0; i < rows; ++i)
        {
            result[i][k] = 0.0;
            for (j = 0; j < SIZE; ++j)
                result[i][k] += matrix1[i][j] * matrix2[j][k];
        }
    }
       
    MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    MPI_Send(*result, rows * SIZE, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
    srand(time(NULL));

    double** matrix1 = CreateMatrix();
    double** matrix2 = CreateMatrix();
    double** result  = CreateMatrix();
    
	MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &process);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) 
    {
        std::cout << "Need at least 2 MPI processes\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks - 1;

    if (process == MASTER)
    {
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                matrix1[i][j] = rand() % 15;
                matrix2[i][j] = rand() % 15;
                result[i][j] = 0.0;
            }
        }

        averow = SIZE / numworkers;
        extra = SIZE % numworkers;
        offset = 0;

        auto start = std::chrono::high_resolution_clock::now();

        for (dest = 1; dest <= numworkers; ++dest)
        {
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&matrix1[offset][0], rows * SIZE, MPI_DOUBLE, dest, FROM_MASTER,
                MPI_COMM_WORLD);
            
            offset = offset + rows;
        }
        MPI_Bcast(*matrix2, SIZE * SIZE, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);


        for (i = 1; i <= numworkers; ++i)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&result[offset][0], rows * SIZE, MPI_DOUBLE, source, FROM_WORKER,
                MPI_COMM_WORLD, &status);
        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = finish - start;
        std::cout << "\nTime: " << duration.count() << " sec\n";
    }

    if (process > MASTER)
    {
        MultiplyMatricesParallel(matrix1, matrix2, result);
    }

    MPI_Finalize();
}
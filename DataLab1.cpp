#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <cassert>

constexpr size_t SIZE = 1000;
std::random_device rd;
using matrix = std::vector<std::vector<double>>;

void Fill(std::vector<double>& right, matrix& coeff, std::vector<double>& result)
{
	std::mt19937 rng(rd());
	std::uniform_int_distribution<int> val(-50, 50);
	// Заполняем матрицу коэффициентов и вектор ответов
	for (size_t i = 0; i < SIZE; ++i)
	{
		for (size_t j = 0; j < SIZE; ++j)
		{
			coeff[i][j] = static_cast<double>(val(rng));
		}
		double value = static_cast<double>(val(rng));
		value += (value == 0) ? 1 : 0;
		result[i] = value;
	}
	//Получение вектора свободных членов
	for (size_t i = 0; i < SIZE; ++i)
	{
		for (size_t j = 0; j < SIZE; ++j)
		{
			right[i] += coeff[i][j] * result[j];
		}
	}
}

void GetInverse(matrix& coeff)
{
	matrix E(SIZE, std::vector<double>(SIZE));
	//Заполнение единичной матрицы
	for(size_t i = 0; i < SIZE; ++i)
	{
		for(size_t j = 0; j < SIZE; ++j)
		{
			E[i][j] = (i == j) ? 1.0f : 0.0f;
		}
	}
	//Зануление элементов матрицы коэффицентов под главной диагональю
	for(size_t k = 0; k < SIZE; ++k)
	{
		//Если на главной диагонали число близкое к 0
		if(abs(coeff[k][k]) < 1e-8)
		{
			for(size_t i = k + 1; i < SIZE; ++i)
			{
				if(abs(coeff[i][k]) > 1e-8)
				{
					swap(coeff[k], coeff[i]);
					swap(E[k], E[i]);
					break;
				}
			}
		}
		double div = coeff[k][k];
		//Деление элементов строки на элемент главной диагонали
		for(size_t j = 0; j < SIZE; ++j)
		{
			coeff[k][j] /= div;
			E[k][j] /= div;
		}
		//Зануление элементов столбца под главной диагональю
		#pragma omp parallel
		{
		#pragma omp for
			for (int i = k + 1; i < SIZE; ++i)
			{
				double correction = coeff[i][k];

				for (int j = 0; j < SIZE; j++)
				{
					coeff[i][j] -= correction * coeff[k][j];
					E[i][j] -= correction * E[k][j];
				}
			}
		}
	}

	//Формирование единичной матрицы из исходной и обратной из единичной
	for (int k = SIZE - 1; k > 0; --k)
	{
		#pragma omp parallel
		{
		#pragma omp for
			for (int i = k - 1; i > -1; --i)
			{
				double correction = coeff[i][k];
				for (int j = 0; j < SIZE; ++j)
				{
					coeff[i][j] -= correction * coeff[k][j];
					E[i][j] -= correction * E[k][j];
				}
			}
		}
	}
	coeff = E;
}
void Solve(std::vector<double>& right, matrix& coeff, std::vector<double>& result)
{
	std::vector<double> X(SIZE);
	//Нахождение ответа
	#pragma omp parallel
	{
	#pragma omp for
		for(int i = 0; i < SIZE; ++i)
		{
			X[i] = 0;
			for (int j = 0; j < SIZE; ++j)
			{
				X[i] += coeff[i][j] * right[j];
			}
			assert(fabs(X[i] - result[i]) < 0.0001f);
		}
	}
}
int main()
{
	matrix coeff(SIZE, std::vector<double>(SIZE));
	std::vector<double> right(SIZE);
	std::vector<double> result(SIZE);
	Fill(right, coeff, result);

	auto start = std::chrono::high_resolution_clock::now();
	GetInverse(coeff);
	Solve(right, coeff, result);
	auto finish = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = finish - start;
	std::cout << "\nTime: " << duration.count() << " sec\n";
}
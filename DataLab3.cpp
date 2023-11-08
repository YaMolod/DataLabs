#include <iostream>
#include <tbb/tbb.h>
#include <vector>
#include <random>
#include <chrono>
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
	for (size_t i = 0; i < SIZE; ++i)
	{
		for (size_t j = 0; j < SIZE; ++j)
		{
			E[i][j] = (i == j) ? 1.0f : 0.0f;
		}
	}
	//Зануление элементов матрицы коэффицентов под главной диагональю
	for (size_t k = 0; k < SIZE; ++k)
	{
		double div = coeff[k][k];
		//Деление элементов строки на элемент главной диагонали
		tbb::parallel_for(tbb::blocked_range<size_t>(0, SIZE),
		[&](tbb::blocked_range<size_t> r)
		{
			for (size_t j = r.begin(); j < r.end(); ++j)
			{
				coeff[k][j] /= div;
				E[k][j] /= div;
			}
		});
		//Зануление элементов столбца под главной диагональю
		tbb::parallel_for(tbb::blocked_range<size_t>(k + 1, SIZE),
		[&](tbb::blocked_range<size_t> r)
		{
		for (size_t i = r.begin(); i < r.end(); ++i)
		{
			double correction = coeff[i][k];
			for (size_t j = 0; j < SIZE; ++j)
			{
				coeff[i][j] -= correction * coeff[k][j];
				E[i][j] -= correction * E[k][j];
			}
		}
		});
	}
	//Формирование единичной матрицы из исходной и обратной из единичной
	for (int k = SIZE - 1; k > 0; --k)
	{
		tbb::parallel_for(tbb::blocked_range<int>(-1, k - 1), [&](tbb::blocked_range<int> r)
		{
			for (int i = r.end(); i > r.begin(); --i)
			{
				double correction = coeff[i][k];
				for (size_t j = 0; j < SIZE; ++j)
				{
					coeff[i][j] -= correction * coeff[k][j];
					E[i][j] -= correction * E[k][j];
				}
			}
		});
	}
	coeff = E;
}

//Вычисление ответа
void Solve(std::vector<double>& right, matrix& coeff, std::vector<double>& result)
{
	std::vector<double> X(SIZE);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, SIZE), [&](tbb::blocked_range<size_t> r)
	{
		for (size_t i = r.begin(); i < r.end(); ++i)
		{
			X[i] = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, SIZE), 0.0f,
			[&](tbb::blocked_range<size_t> r, double total)
			{
				for (size_t j = r.begin(); j < r.end(); ++j)
				{
					total += coeff[i][j] * right[j];
				}
			return total;
			}, std::plus<double>());
		}
	});
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
	std::cout << "\n\Time: " << duration.count() << " sec\n";
}
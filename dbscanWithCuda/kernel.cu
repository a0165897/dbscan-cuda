#include "cuda_runtime.h"
#include "device_functions.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <windows.h>
#include <math.h>
#include <queue>

using namespace std;

struct Point {
	float		x;
	float       y;
	int			cluster;			
	int			noise;  //-1 noise

};

int eps = 2;//neighborhood radius
int min_nb = 3;
Point host_sample[500];//312
int block_num, thread_num;

float __device__ dev_euclidean_distance(const Point &src, const Point &dest) {

	float res = (src.x - dest.x) * (src.x - dest.x) + (src.y - dest.y) * (src.y - dest.y);

	return sqrt(res);
}

/*to get the total list*/
void __global__ dev_region_query(Point* sample, int num, int* neighbors, int eps, int min_nb) {

	unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int	line,col,pointer = tid;
	unsigned int	count;
	 
	while (pointer < num * num) {//全场唯一id
		line = pointer / num;
		col = pointer % num;
		float radius;
		if (line <= col) {
			radius = dev_euclidean_distance(sample[line], sample[col]);
			if (radius < eps) {
				neighbors[pointer] = 1;
			}
			neighbors[col * num + line] = neighbors[pointer];//对角线
		}
		pointer += blockDim.x * gridDim.x;
	}
	__syncthreads();

	pointer = tid;
	while (pointer < num) {
		count = 0;
		line = pointer * num;
		for (int i = 0; i < num; i++) {
			if (pointer != i && neighbors[line+i]) {//除了p点外邻域元素个数
					count++;
			}
		}
		if (count >= min_nb) {
			sample[pointer].noise++;
		}
		pointer += blockDim.x * gridDim.x;
	}
}

void host_algorithm_dbscan(Point* host_sample, int num) {
	/*sample*/
	Point* cuda_sample;
	cudaMalloc((void**)&cuda_sample, num * sizeof(Point));
	cudaMemcpy(cuda_sample, host_sample, num * sizeof(Point), cudaMemcpyHostToDevice);

	/*neighbor list*/
	int *host_neighbor = new int[num*num]();
	int *dev_neighbor;
	cudaMalloc((void**)&dev_neighbor, num * num * sizeof(int));
	
	dev_region_query << <block_num, thread_num >> > (cuda_sample, num, dev_neighbor, eps, min_nb);

	cudaMemcpy(host_sample, cuda_sample, num * sizeof(Point), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_neighbor, dev_neighbor, num * num * sizeof(int), cudaMemcpyDeviceToHost);
	
	queue<int> expand;
	int cur_cluster = 0;

	for (int i = 0; i < num; i++) {
		if (host_sample[i].noise >= 0 && host_sample[i].cluster < 1) {
			host_sample[i].cluster = ++cur_cluster; 
			int src = i * num;
			for (int j = 0; j < num; j++) {
				if (host_neighbor[src + j]) {
					host_sample[j].cluster = cur_cluster;
					expand.push(j);
				}
			}

			while (!expand.empty()) {/*expand the cluster*/
				if (host_sample[expand.front()].noise >= 0) {
					src = expand.front() * num;
					for (int j = 0; j < num; j++) {
						if (host_neighbor[src + j] && host_sample[j].cluster < 1) {
							host_sample[j].cluster = cur_cluster;
							expand.push(j);
						}
					}
				}
				expand.pop();
			}
		}
	}
	cudaFree(cuda_sample);cudaFree(dev_neighbor);
}

int main(int argc, char* argv[]) {
	clock_t starts, finishs;
	double duration;
	starts = clock();
	ifstream fin("3spiral.txt");
	ofstream fout;
	fout.open("result.txt");
	int sample_num = 0;
	double a, b;
	while (fin >> a >> b) {
		host_sample[sample_num].x = a;
		host_sample[sample_num].y = b;
		host_sample[sample_num].noise = -1;
		host_sample[sample_num].cluster = -1;
		sample_num++;
	}

	cout << "------>TOTAL SAMPLE NUMB0->" << sample_num << "<-----" << endl;
	cout << "------>BL0CK=10 & THREAD=100<-------- "<< endl;
	block_num = 10;
	thread_num = 100;
	cout<<"CALCULATING BY CUDA GTX965M......\n"<<endl;
	
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	host_algorithm_dbscan(host_sample, sample_num);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	float time;
	cudaEventElapsedTime(&time, start, end);
	cout<<"time: "<< time <<"ms --device\n"<<endl;
	
	finishs = clock();
	duration = (double)(finishs - starts) / CLOCKS_PER_SEC;
	cout << duration << "s --total" << endl;

	for (int i = 0; i < sample_num; i++) {
		fout <<"["<<host_sample[i].x << "," << host_sample[i].y << "] -->"<<host_sample[i].cluster<< endl;
	}

	fout.close();
	system("pause");
	return 0;
}
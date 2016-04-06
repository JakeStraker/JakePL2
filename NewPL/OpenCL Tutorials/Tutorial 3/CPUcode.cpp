#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

vector<int> getHist(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t input_size, size_t input_elements, vector<int> hostOut, size_t local_size, int count, int minval, int maxval) {
	cl::Kernel kernel_Hist = cl::Kernel(program, "histogram");
	kernel_Hist.setArg(0, buffer_A);
	kernel_Hist.setArg(1, buffer_B);
	kernel_Hist.setArg(2, count);
	kernel_Hist.setArg(3, minval);
	kernel_Hist.setArg(4, maxval);
	queue.enqueueFillBuffer(buffer_B, 0, 0, input_size);//zero B buffer on device memory
	queue.enqueueNDRangeKernel(kernel_Hist, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, input_size, &hostOut[0]);
	return hostOut;

}
int averageTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t input_size, size_t input_elements, vector<int> hostOutput, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Average = cl::Kernel(program, "averageTemperature");//Average

	kernel_Average.setArg(0, buffer_A);
	kernel_Average.setArg(1, buffer_B);
	kernel_Average.setArg(2, cl::Local(1));
	queue.enqueueFillBuffer(buffer_B, 0, 0, input_size);//zero B buffer on device memory
	queue.enqueueNDRangeKernel(kernel_Average, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, input_size, &hostOutput[0]);

	return hostOutput[0];
}
int calcMin(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t input_size, size_t input_elements, vector<int> hostOutput, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Average = cl::Kernel(program, "minTemperature");//Average

	kernel_Average.setArg(0, buffer_A);
	kernel_Average.setArg(1, buffer_B);
	kernel_Average.setArg(2, cl::Local(1));
	queue.enqueueFillBuffer(buffer_B, 0, 0, input_size);//zero B buffer on device memory
	queue.enqueueNDRangeKernel(kernel_Average, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
	
	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, input_size, &hostOutput[0]);

	return hostOutput[0];
}
int calcMax(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t input_size, size_t input_elements, vector<int> hostOutput, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Average = cl::Kernel(program, "maxTemperature");//Average

	kernel_Average.setArg(0, buffer_A);
	kernel_Average.setArg(1, buffer_B);
	kernel_Average.setArg(2, cl::Local(1));
	queue.enqueueFillBuffer(buffer_B, 0, 0, input_size);//zero B buffer on device memory

	queue.enqueueNDRangeKernel(kernel_Average, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, input_size, &hostOutput[0]);

	return hostOutput[0];
}
string hashPrint(int n)
{
	string stream = "";
	for (int i = 0; i < n; i++)
	{
		stream += '#';
	}
	return stream;
}


int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}
	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);
		//2.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "gpuCode.cl");
		cl::Program program(context, sources);
		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		typedef int mytype;
		//Part 4 - memory allocation
		//host - input
		vector<int> temperature;
		vector<int> month;
		//All variables used to manage data entered from text file and store in relevant vectors
		string line;
		char searchItem = ' ';
		string word = "";
		int seperatorCount = 0;
		int temp = 0; //Used to store a temporary value of temp as atomic_add only works on integers.
		ifstream myfile("temp_lincolnshire_short.txt");
		string values[5] = { "Weather Station", "Year", "Month", "Day" , "Time" };
		string findText[6];
		for (int i = 0; i <= 4; i++)
		{
			std::cout << "Enter a " << values[i] << " or leave blank for all" << endl;
			string input;
			getline(cin, input);

			if (!input.empty())
			{
				istringstream istr(input);
				istr >> findText[i];
			}
			else { findText[i] = ""; }
		}
		if (myfile.is_open())
		{
			std::cout << "File Reading...\n";
			while (getline(myfile, line))
			{
				string lineItems[6];
				int ndex = 0;
				for (int i = 0; i < line.length(); i++)
				{
					if ((line[i] != searchItem))
					{
						lineItems[ndex] += line[i];
					}
					else { ndex++; }
				}
				bool okay = true;
				for (int i = 0; i <= 5; i++)
				{
					if (findText[i] == "") { continue; }
					else if (lineItems[i] == findText[i]) { continue; }
					else { okay = false; break; }
				}
				if (okay == true) {
					for (int i = 0; i < line.length(); i++)
					{
						word += line[i];
						if (line[i] == searchItem || i == line.length() - 1)
						{
							seperatorCount++;
							switch (seperatorCount)
							{
							case 3:
								month.push_back(stoi(word));
								word = "";
								break;
							case 6:
								temp = int(stof(word) * 10);
								temperature.push_back(temp);
								seperatorCount = 0;
								word = "";
								break;
							default:
								word = "";
								break;
							}
						}
					}
				}
			}
			cout << "FileRead Complete \n";
			myfile.close();
		}
		size_t local_size = (64, 1);
		size_t padding_size = temperature.size() % local_size;
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected (make work for my working set of data)
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			temperature.insert(temperature.end(), A_ext.begin(), A_ext.end());
		}
		

		size_t input_elements = temperature.size();//number of input elements
		size_t input_size = temperature.size()*sizeof(mytype);//size in bytes
		size_t m_elements = month.size();
		size_t m_size = month.size()*sizeof(mytype);
		//host - output
		std::vector<int> hostOutput(input_elements);
		std::vector<int> seasonalOut(m_elements);
		size_t output_size = hostOutput.size()*sizeof(mytype);//size in bytes
		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_output_size(context, CL_MEM_READ_WRITE, output_size);
		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temperature[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		float minV = (float)(calcMin(program, buffer_A, buffer_output_size, queue, input_size, input_elements, hostOutput, local_size));
		float maxV = (float)(calcMax(program, buffer_A, buffer_output_size, queue, input_size, input_elements, hostOutput, local_size));
		float avgVal = (averageTemperature(program, buffer_A, buffer_output_size, queue, input_size, input_elements, hostOutput, local_size) / 10);
		std::cout << "The maximum temperature is = " << maxV / 10 << std::endl;
		std::cout << "The minimum temperature is = " << minV / 10 << std::endl;
		std::cout << "The Average Temperature is = " << avgVal / input_elements << endl;
		std::cout << "Enter number of Histogram Bins" << endl;
		int binNum = 1;
		std::cin >> binNum;
		while (binNum <= 0 || std::cin.fail())
		{
			std::cout << "Invalid Number of bins, please enter a number above 0" << endl;
			std::cin.clear();
			std::cin.ignore(numeric_limits<streamsize>::max(), '\n');
			std::cin >> binNum;
		}
		hostOutput = getHist(program, buffer_A, buffer_output_size, queue, input_size, input_elements, hostOutput, local_size, binNum, minV, maxV);
		std::cout << "-----------------------------------------------------------------------------------------------------" << endl;
		float increment = ((maxV - minV) / binNum);
		float histMax = 0;
		for (int i = 1; i < binNum + 1; i++) { if (hostOutput[i - 1] > histMax) { histMax = hostOutput[i - 1]; } }
		for (int i = 1; i < binNum + 1; i++) {
		int hashCount = (((hostOutput[i - 1]) / histMax) * 60);
		std::cout << std::fixed << std::setprecision(2)  << "  (" << ((minV + ((i - 1)*increment)) / 10) << ")   \t - \t   (" << ((minV + (i*increment)) / 10) << "):\t" << (hostOutput[i - 1]) << "\t |" << hashPrint(hashCount) << endl;
		}
		}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	int f = 0;
	std::cin >> f;
	return 0;
}

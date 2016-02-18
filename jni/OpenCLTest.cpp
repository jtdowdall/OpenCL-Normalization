
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <math.h>

/*
 * needed for loadProgram function
 */
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>


#include <sys/time.h>

#include <CL/cl.h>


// Commonly-defined shortcuts for LogCat output from native C applications.
#define  LOG_TAG    "AndroidBasic"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


/* Container for all OpenCL-specific objects used in the sample.
 *
 * The container consists of the following parts:
 *   - Regular OpenCL objects, used in almost each
 *     OpenCL application.
 *   - Specific OpenCL objects - buffers, used in this
 *     particular sample.
 *
 * For convenience, collect all objects in one structure.
 * Avoid global variables and make easier the process of passing
 * all arguments in functions.
 */
struct OpenCLObjects
{
    // Regular OpenCL objects:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel Normalize;

	// First matrix (device)
	cl_mem matrixA;

	bool isInputBufferInitialized;

};

// Hold all OpenCL objects.
OpenCLObjects openCLObjects;

/*
 * Load the program out of the file in to a string for opencl compiling.
 */
inline std::string loadProgram(std::string input)
{
	std::ifstream stream(input.c_str());
	if (!stream.is_open()) {
		LOGE("Cannot open input file\n");
		exit(1);
	}
	return std::string( std::istreambuf_iterator<char>(stream),
						(std::istreambuf_iterator<char>()));
}

/* This function helps to create informative messages in
 * case when OpenCL errors occur. The function returns a string
 * representation for an OpenCL error code.
 * For example, "CL_DEVICE_NOT_FOUND" instead of "-1".
 */
const char* opencl_error_to_str (cl_int error)
{
#define CASE_CL_CONSTANT(NAME) case NAME: return #NAME;

    // Suppose that no combinations are possible.
    switch(error)
    {
        CASE_CL_CONSTANT(CL_SUCCESS)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
        CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
        CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
        CASE_CL_CONSTANT(CL_MAP_FAILURE)
        CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
        CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE)
        CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
        CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
        CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
        CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
        CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
        CASE_CL_CONSTANT(CL_INVALID_BINARY)
        CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL)
        CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
        CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
        CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
        CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
        CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_EVENT)
        CASE_CL_CONSTANT(CL_INVALID_OPERATION)
        CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_PROPERTY)

    default:
        return "UNKNOWN ERROR CODE";
    }

#undef CASE_CL_CONSTANT
}


/* The following macro is used after each OpenCL call
 * to check if OpenCL error occurs. In the case when ERR != CL_SUCCESS
 * the macro forms an error message with OpenCL error code mnemonic,
 * puts it to LogCat, and returns from a caller function.
 *
 * The approach helps to implement consistent error handling tactics
 * because it is important to catch OpenCL errors as soon as
 * possible to avoid missing the origin of the problem.
 *
 * You may chose a different way to do that. The macro is
 * simple and context-specific as it assumes you use it in a function
 * that doesn't have a return value, so it just returns in the end.
 */
#define SAMPLE_CHECK_ERRORS(ERR)                                                      \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGE                                                                          \
        (                                                                             \
            "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
            opencl_error_to_str(ERR), __FILE__, __LINE__                              \
        );                                                                            \
                                                                                      \
        return;                                                                       \
    }


void initOpenCL
(
    JNIEnv* env,
    jobject thisObject,
    jstring kernelName,
    cl_device_type required_device_type,
    OpenCLObjects& openCLObjects
)
{
    /*
     * This function picks and creates all necessary OpenCL objects
     * to be used at each filter iteration. The objects are:
     * OpenCL platform, device, context, command queue, program,
     * and kernel.
     *
     * Almost all of these steps need to be performed in all
	 * OpenCL applications before the actual compute kernel calls
     * are performed.
     *
     * For convenience, in this application all basic OpenCL objects
     * are stored in the OpenCLObjects structure,
     * so, this function populates fields of this structure,
     * which is passed as parameter openCLObjects.
     * Consider reviewing the fields before going further.
     * The structure definition is in the beginning of this file.
     */

    using namespace std;

    // Search for the Intel OpenCL platform.
    // Platform name includes "Intel" as a substring, consider this
    // method to be a recommendation for Intel OpenCL platform search.
    const char* required_platform_subname = "PowerVR";

    // The following variable stores return codes for all OpenCL calls.
    // In the code it is used with the SAMPLE_CHECK_ERRORS macro defined
    // before this function.
    cl_int err = CL_SUCCESS;

    /* -----------------------------------------------------------------------
     * Step 1: Query for all available OpenCL platforms on the system.
     * Enumerate all platforms and pick one which name has
     * required_platform_subname as a sub-string.
     */

    cl_uint num_of_platforms = 0;
    // Get total number of the available platforms.
    err = clGetPlatformIDs(1, &openCLObjects.platform, &num_of_platforms);
    //SAMPLE_CHECK_ERRORS(err);
    //LOGD("Number of available platforms: %u", num_of_platforms);

    vector<cl_platform_id> platforms(num_of_platforms);
    // Get IDs for all platforms.
    err = clGetPlatformIDs(num_of_platforms, &platforms[0], 0);
    //SAMPLE_CHECK_ERRORS(err);

    //cl_uint selected_platform_index = num_of_platforms;

    //LOGD("Platform names:");

    cl_uint i = 0;
        // Get the length for the i-th platform name.
        size_t platform_name_length = 0;
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            0,
            0,
            &platform_name_length
        );
        //SAMPLE_CHECK_ERRORS(err);

        // Get the name itself for the i-th platform.
        vector<char> platform_name(platform_name_length);
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            platform_name_length,
            &platform_name[0],
            0
        );
        //SAMPLE_CHECK_ERRORS(err);
    //selected_platform_index = 0;
    openCLObjects.platform = platforms[0];


    /* -----------------------------------------------------------------------
     * Step 2: Create context with a device of the specified type.
     * Required device type is passed as function argument required_device_type.
     * Use this function to create context for any CPU or GPU OpenCL device.
     */

    cl_context_properties context_props[] = {
        CL_CONTEXT_PLATFORM,
        cl_context_properties(openCLObjects.platform),
        0
    };

    openCLObjects.context =
        clCreateContextFromType
        (
            context_props,
            required_device_type,
            0,
            0,
            &err
        );
    //SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * Step 3: Query for OpenCL device that was used for context creation.
     */

    err = clGetContextInfo
    (
        openCLObjects.context,
        CL_CONTEXT_DEVICES,
        sizeof(openCLObjects.device),
        &openCLObjects.device,
        0
    );
    /*char deviceName[1024];
    err = clGetDeviceInfo( openCLObjects.device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL );
    LOGD("CL_DEVICE_NAME: %s", deviceName );

    int deviceCores;
    err = clGetDeviceInfo( openCLObjects.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceCores), &deviceCores, NULL );
    LOGD("Total Cores: %d", deviceCores );

    long deviceGlobalMem = 0;
	err = clGetDeviceInfo( openCLObjects.device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceGlobalMem) * 2, &deviceGlobalMem, NULL );
	LOGD("Global Memory Cache (bytes) : %d", deviceGlobalMem );*/

    //SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * Step 4: Create OpenCL program from its source code.
     * The file name is passed bij java.
     * Convert the jstring to const char* and append the needed directory path.
     */
    const char* fileName = env->GetStringUTFChars(kernelName, 0);
    std::string fileDir;
    fileDir.append("/data/data/com.example.opencltest/app_execdir/");
    fileDir.append(fileName);
    fileDir.append(".cl");
    std::string kernelSource = loadProgram(fileDir);
    //std::string to const char* needed for the clCreateProgramWithSource function
    const char* kernelSourceChar = kernelSource.c_str();

    openCLObjects.program =
        clCreateProgramWithSource
        (
            openCLObjects.context,
            1,
            &kernelSourceChar,
            0,
            &err
        );

    //SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * Step 5: Build the program.
     * During creation a program is not built. Call the build function explicitly.
     * This example utilizes the create-build sequence, still other options are applicable,
     * for example, when a program consists of several parts, some of which are libraries.
     * Consider using clCompileProgram and clLinkProgram as alternatives.
     * Also consider looking into a dedicated chapter in the OpenCL specification
     * for more information on applicable alternatives and options.
     */

    err = clBuildProgram(openCLObjects.program, 0, 0, 0, 0, 0);

    if(err == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_length = 0;
        err = clGetProgramBuildInfo(
            openCLObjects.program,
            openCLObjects.device,
            CL_PROGRAM_BUILD_LOG,
            0,
            0,
            &log_length
        );
        //SAMPLE_CHECK_ERRORS(err);

        vector<char> log(log_length);

        err = clGetProgramBuildInfo(
            openCLObjects.program,
            openCLObjects.device,
            CL_PROGRAM_BUILD_LOG,
            log_length,
            &log[0],
            0
        );
        //SAMPLE_CHECK_ERRORS(err);

        LOGE
        (
            "Error happened during the build of OpenCL program.\nBuild log:%s",
            &log[0]
        );

        return;
    }

    /* -----------------------------------------------------------------------
     * Step 6: Extract kernel from the built program.
     * An OpenCL program consists of kernels. Each kernel can be called (enqueued) from
     * the host part of an application.
     * First create a kernel to call it from the existing program.
     * Creating a kernel via clCreateKernel is similar to obtaining an entry point of a specific function
     * in an OpenCL program.
     */

    openCLObjects.Normalize = clCreateKernel(openCLObjects.program, "Normalize", &err);

    /* -----------------------------------------------------------------------
     * Step 7: Create command queue.
     * OpenCL kernels are enqueued for execution to a particular device through
     * special objects called command queues. Command queue provides ordering
     * of calls and other OpenCL commands.
     * This sample uses a simple in-order OpenCL command queue that doesn't
     * enable execution of two kernels in parallel on a target device.
     */

    openCLObjects.queue =
        clCreateCommandQueue
        (
            openCLObjects.context,
            openCLObjects.device,
            0,    // Creating queue properties, refer to the OpenCL specification for details.
            &err
        );
    //SAMPLE_CHECK_ERRORS(err);

    // -----------------------------------------------------------------------

    LOGD("initOpenCL finished successfully");
}


extern "C" void Java_com_example_opencltest_MainActivity_initOpenCL
(
    JNIEnv* env,
    jobject thisObject,
    jstring kernelName
)
{
    initOpenCL
    (
        env,
        thisObject,
        kernelName,
        CL_DEVICE_TYPE_GPU,
        openCLObjects
    );
}


void shutdownOpenCL (OpenCLObjects& openCLObjects)
{
    /* Release all OpenCL objects.
     * This is a regular sequence of calls to deallocate
     * all created OpenCL resources in bootstrapOpenCL.
     *
     * You can call these deallocation procedures in the middle
     * of your application execution (not at the end) if you don't
     * need OpenCL runtime any more.
     * Use deallocation, for example, to free memory or recreate
     * OpenCL objects with different parameters.
     *
     * Calling deallocation in the end of application
     * execution might be not so useful, as upon killing
     * an application, which is a common thing in the Android OS,
     * all OpenCL resources are deallocated automatically.
     */

    cl_int err = CL_SUCCESS;

    if(openCLObjects.isInputBufferInitialized)
    {
		err = clReleaseMemObject(openCLObjects.matrixA);
		//SAMPLE_CHECK_ERRORS(err);
		openCLObjects.isInputBufferInitialized = false;
    }
    err = clReleaseKernel(openCLObjects.Normalize);
	//SAMPLE_CHECK_ERRORS(err);

    err = clReleaseProgram(openCLObjects.program);
    //SAMPLE_CHECK_ERRORS(err);

    err = clReleaseCommandQueue(openCLObjects.queue);
    //SAMPLE_CHECK_ERRORS(err);

    err = clReleaseContext(openCLObjects.context);
    //SAMPLE_CHECK_ERRORS(err);

    /* There is no procedure to deallocate OpenCL devices or
     * platforms as both are not created at the startup,
     * but queried from the OpenCL runtime.
     */
}


extern "C" void Java_com_example_opencltest_MainActivity_shutdownOpenCL
(
    JNIEnv* env,
    jobject thisObject
)
{
    LOGD("shutdownOpenCL(openCLObjects) was called");
    shutdownOpenCL(openCLObjects);
}


/*
 * Effect step.
 * This function is called each time you need to process the image.
 *
 * The function consists of the following parts:
 *   - reading input image content if changed
 *   - running OpenCL kernel on the input image
 *   - reading results of image processing
 */

/*void NormalizeCPU(float *elements, int size, int flag) {
	float epsilon = (float) 0.0011;

	if (flag ==1){
		float min = elements[0];
		float max = elements[0];
		for (int i = 0; i < size; i++){
			if(elements[i] < min){min = elements[i];}
			if(elements[i] > max){max = elements[i];}
		}
		float diff = max-min + epsilon;
		for(int i = 0; i < size; i++){
			elements[i] = (elements[i]-min)/diff;
		}
		float mean = 0;
		for (int i = 0; i < size; i++){
			mean += elements[i];
		}
		mean = mean/size;
		for (int i = 0; i < size; i++){
			elements[i] = elements[i]-mean + epsilon;
		}
		float norm = 0;
		for (int i = 0; i < size; i++){
			norm += elements[i]*elements[i];
		}
		norm = (float) sqrt(norm);
		if (norm > 0){
			for (int i = 0; i < size; i++){
				elements[i] = elements[i]/norm;
			}
		}
	}

	if(flag==2){
		float norm = 0;
		for (int i = 0; i < size; i++){
			norm += elements[i]* elements[i];
		}
		norm = (float) sqrt(norm);
		if (norm > 0){
			for (int i = 0; i < size; i++){
				elements[i] = elements[i]/norm;
			}
		}
	}

	if (flag == 3){
		float norm = 0;
		for (int i = 0; i < size; i++){
			norm += elements[i];
		}
		norm = norm+ epsilon;
		if (norm > 0){
			for (int i = 0; i < size; i++){
				elements[i] = elements[i]/norm;
			}
		}
	}
}*/

void NormalizeGPU
  (JNIEnv * env,
   jobject obj,
   OpenCLObjects& openCLObjects,
   jobjectArray matrix,
   jint num_rows,
   jint num_cols,
   jint rf_size,
   jint flag)
{
	cl_int err = CL_SUCCESS;

	if(openCLObjects.isInputBufferInitialized)
	{
		/* If this is not the first time, you need to deallocate the previously
		 * allocated buffer as the new buffer will be allocated in
		 * the next statements.
		 *
		 * It is important to remember that unlike Java, there is no
		 * garbage collector for OpenCL objects, so deallocate all resources
		 * explicitly to avoid running out of memory.
		 *
		 * It is especially important in case of image buffers,
		 * because they are relatively large and even one lost buffer
		 * can significantly limit free resources for the application.
		 */

		err = clReleaseMemObject(openCLObjects.matrixA);
		//SAMPLE_CHECK_ERRORS(err);
		openCLObjects.isInputBufferInitialized = false;
	}

	int size = num_rows * num_cols;

	// Create device buffer;
	openCLObjects.matrixA = clCreateBuffer(
			openCLObjects.context,
			CL_MEM_READ_WRITE,
			size * sizeof(cl_float),
			NULL,
			&err );
	//SAMPLE_CHECK_ERRORS(err);

	openCLObjects.isInputBufferInitialized = true;

	// Map A for writing
	float *pMatrixA = (float*) clEnqueueMapBuffer(
			openCLObjects.queue,
			openCLObjects.matrixA,
			CL_TRUE,
			CL_MAP_READ | CL_MAP_WRITE,
			0,
			size * sizeof(float),
			0,
			NULL,
			NULL,
			NULL);

	// Fill A with data
	for( cl_int row = 0; row < num_rows; row++ )
	{
		jfloatArray vector = (jfloatArray) env->GetObjectArrayElement(matrix, row);
		float *elements = env->GetFloatArrayElements(vector, 0);
		for( cl_int column = 0; column < num_cols; column++ )
		{
			float val = elements[column];
			pMatrixA[ row * num_cols + column ] = val;
			//LOGD("VAL: A[%d][%d]: %f", row, col, pMatrixA[ row * num_cols + col ]);
		}
	}

	// Unmap A
	err = clEnqueueUnmapMemObject( openCLObjects.queue, openCLObjects.matrixA, pMatrixA, 0, NULL, NULL );
	//SAMPLE_CHECK_ERRORS(err);

	// Set the kernel arguments

	size_t globalWorkSize[2] = { num_rows, 2048 };
	size_t localWorkSize[2] = { 1, 128};
	size_t workGroupSize = 16;
	size_t vectorSize = workGroupSize*localWorkSize[1];

	err |= clSetKernelArg( openCLObjects.Normalize, 0, sizeof(int), &num_cols );
	err |= clSetKernelArg( openCLObjects.Normalize, 1, sizeof(int), &workGroupSize );
	err |= clSetKernelArg( openCLObjects.Normalize, 2, sizeof(cl_mem), &openCLObjects.matrixA );
	err |= clSetKernelArg( openCLObjects.Normalize, 3, sizeof(float)*vectorSize, NULL); //vector
	err |= clSetKernelArg( openCLObjects.Normalize, 4, sizeof(float)*vectorSize, NULL); //difference
	SAMPLE_CHECK_ERRORS(err);

	// Queue the kernel for execution

	size_t kernelSize[1];
	err = clGetKernelWorkGroupInfo ( openCLObjects.Normalize,
			openCLObjects.device,
			CL_KERNEL_WORK_GROUP_SIZE,
	     	sizeof(kernelSize),
	     	kernelSize,
	     	NULL);
	LOGD("CL_KERNEL_WORK_GROUP_SIZE: %d", kernelSize[0]);

	err = clEnqueueNDRangeKernel( openCLObjects.queue, openCLObjects.Normalize, 2, NULL,
						globalWorkSize, localWorkSize, 0, NULL, NULL );
	SAMPLE_CHECK_ERRORS(err);
	/*
	err = clEnqueueNDRangeKernel( openCLObjects.queue, openCLObjects.UpdateDifference, 2, NULL,
							globalWorkSize, NULL, 0, NULL, NULL );
	//SAMPLE_CHECK_ERRORS(err);

	//Mean
	err = clEnqueueNDRangeKernel( openCLObjects.queue, openCLObjects.ComputeMean, 2, NULL,
							globalWorkSize, NULL, 0, NULL, NULL );
	//SAMPLE_CHECK_ERRORS(err);
	err = clEnqueueNDRangeKernel( openCLObjects.queue, openCLObjects.UpdateMean, 2, NULL,
							globalWorkSize, NULL, 0, NULL, NULL );
	//SAMPLE_CHECK_ERRORS(err);

	// Norm
	err = clEnqueueNDRangeKernel( openCLObjects.queue, openCLObjects.ComputeNorm, 2, NULL,
							globalWorkSize, NULL, 0, NULL, NULL );
	//SAMPLE_CHECK_ERRORS(err);
	err = clEnqueueNDRangeKernel( openCLObjects.queue, openCLObjects.UpdateNorm, 2, NULL,
							globalWorkSize, NULL, 0, NULL, NULL );
	//SAMPLE_CHECK_ERRORS(err);*/

	clFinish(openCLObjects.queue);

	// Copy result back to java resultArray
	for( cl_int row = 0; row < num_rows; row++ )
	{
		jfloatArray resultVector = (jfloatArray) env->GetObjectArrayElement(matrix, row);
		float* resultVectorElements = env->GetFloatArrayElements(resultVector, 0);
		for( cl_int column = 0; column < num_cols; column++ )
		{
			resultVectorElements[column] = pMatrixA[ row * num_cols + column ];
			//LOGD("RESULT[%d][%d]: %f", row, column, pMatrixA[ row * num_cols + column ]);
		}
		//NormalizeCPU(resultVectorElements, num_cols, 1);
		env->ReleaseFloatArrayElements(resultVector, resultVectorElements, 0);
		env->DeleteLocalRef(resultVector);
	}
}

//NORMALIZE NormalizeGPU(float[][] weight, int num_rows, int num_cols, int rf_size, int flag);
extern "C" jfloatArray Java_com_example_opencltest_MainActivity_NormalizeGPU
(JNIEnv * env,
   jobject obj,
   jobjectArray matrix,
   jint num_rows,
   jint num_cols,
   jint rf_size,
   jint flag)
{
	//LOGD("SIZE: %d", size );
    NormalizeGPU
    (
        env,
        obj,
        openCLObjects,
        matrix,
		num_rows,
		num_cols,
		rf_size,
		flag
    );
    return 0;
}

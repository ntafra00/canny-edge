#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

const char *inputImagePath = "input-cpu.jpg";
const char *outputImagePath = "output-cpu.jpg";

__global__ void imgToArray(const uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth, int *pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = threadIdx.z;

    if (i < sizeRows && j < sizeCols && k < sizeDepth)
    {
        int idx = i * sizeCols * sizeDepth + j * sizeDepth + k;
        pixels[idx] = static_cast<int>(pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + 2 - k]);
    }
}

__global__ void arrayToImg(int *pixels, uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sizeRows && j < sizeCols)
    {
        for (int k = 0; k < sizeDepth; k++)
        {
            pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                (uint8_t)pixels[i * sizeCols * sizeDepth + j * sizeDepth + (sizeDepth - 1 - k)];
        }
    }
}

__global__ void gaussianBlur(int *inputPixels, int sizeRows, int sizeCols, int sizeDepth, int *outputPixels)
{
    double kernel[5][5] = {{2.0, 4.0, 5.0, 4.0, 2.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {5.0, 12.0, 15.0, 12.0, 5.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < sizeDepth; k++)
    {
        double sum = 0;
        double sumKernel = 0;
        for (int y = -2; y <= 2; y++)
        {
            for (int x = -2; x <= 2; x++)
            {
                if ((i + x) >= 0 && (i + x) < sizeRows && (j + y) >= 0 && (j + y) < sizeCols)
                {
                    double channel = (double)inputPixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
                    sum += channel * kernelConst * kernel[x + 2][y + 2];
                    sumKernel += kernelConst * kernel[x + 2][y + 2];
                }
            }
        }
        outputPixels[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
    }
}

__global__ void rgbToGrayscale(int *pixels, int *pixelsGray, int sizeRows, int sizeCols, int sizeDepth)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sizeRows && j < sizeCols)
    {
        int sum = 0;
        for (int k = 0; k < sizeDepth; k++)
        {
            sum += pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
        }
        pixelsGray[i * sizeCols + j] = sum / sizeDepth;
    }
}

void cannyEdgeDetection(uint8_t *inputImage, double lowerThreshold, double higherThreshold, int width, int height, int channels)
{
    int imageSize = height * width * channels;
    uint8_t *inputImagePtr;
    int *outputImagePtr;
    uint8_t *outputImagePixels;

    // Allocate memory on GPU
    cudaMalloc((void **)&inputImagePtr, imageSize * sizeof(uint8_t));
    cudaMalloc((void **)&outputImagePtr, imageSize * sizeof(int));
    cudaMalloc((void **)&outputImagePixels, imageSize * sizeof(uint8_t));

    // Copy input image to CUDA
    cudaMemcpy(inputImagePtr, inputImage, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, channels);
    dim3 gridSize((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y, (channels + blockSize.z - 1) / blockSize.z);

    // Convert given image to array of pixels
    imgToArray<<<blockSize, gridSize>>>(inputImagePtr, height, width, channels, outputImagePtr);

    std::cout << "Exited imgToArray" << std::endl;

    // Performing gaussian blur
    gaussianBlur<<<blockSize, gridSize>>>(outputImagePtr, height, width, channels, outputImagePtr);

    // GRAYSCALE:

    rgbToGrayscale<<<blockSize, gridSize>>>(outputImagePtr, outputImagePtr, height, width, channels);

    std::cout << "Exited rgbToGrayscale" << std::endl;

    // CANNY_FILTER:

    // std::vector<int> pixelsCanny = cannyFilter(pixelsGray, height, width, 1, lowerThreshold, higherThreshold);

    arrayToImg<<<blockSize, gridSize>>>(outputImagePtr, outputImagePixels, height, width, 1);

    std::cout << "Exited arrayToImg" << std::endl;

    uint8_t *outputImage = (unsigned char *)malloc(width * height);

    cudaMemcpy(outputImage, outputImagePixels, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

    cudaFree(inputImagePtr);
    cudaFree(outputImagePtr);
    cudaFree(outputImagePixels);

    free(outputImage);
}

int main()
{
    int width, height, channels;

    // Load the input image
    uint8_t *inputImage = stbi_load(inputImagePath, &width, &height, &channels, STBI_rgb);
    if (inputImage == NULL)
    {
        printf("Failed to load image: %s\n", inputImagePath);
        return -1;
    }

    // if (width < 1400 || height < 1400)
    // {
    //     printf("Choose image with greater resolution\n");
    //     return -1;
    // }

    if (channels != 3)
    {
        printf("Images is not in RGB format\n");
        return -1;
    }

    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;

    cannyEdgeDetection(inputImage, lowerThreshold, higherThreshold, width, height, channels);

    return 0;
}
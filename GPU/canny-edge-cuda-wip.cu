#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

const char *inputImagePath = "../input.jpg";
const char *outputImagePath = "output-gpu.jpg";

int *imgToArray(uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
{
    int *pixels = (int *)malloc((sizeRows * sizeCols * sizeDepth) * sizeof(int));
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                // converting BGR to RGB colors
                pixels[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (int)pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + 2 - k];
            }
        }
    }
    return pixels;
}

void arrayToImg(int *pixels, uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth)
{
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            for (int k = 0; k < sizeDepth; k++)
            {
                pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    (uint8_t)pixels[i * sizeCols * sizeDepth + j * sizeDepth + (sizeDepth - 1 - k)];
            }
        }
    }
    return;
}

std::vector<int> gaussianBlur(std::vector<int> &pixels, std::vector<std::vector<double>> &kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
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
                            double channel = (double)pixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
                            sum += channel * kernelConst * kernel[x + 2][y + 2];
                            sumKernel += kernelConst * kernel[x + 2][y + 2];
                        }
                    }
                }
                pixelsBlur[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
            }
        }
    }
    return pixelsBlur;
}

std::vector<int> rgbToGrayscale(std::vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth)
{
    std::vector<int> pixelsGray(sizeRows * sizeCols);
    for (int i = 0; i < sizeRows; i++)
    {
        for (int j = 0; j < sizeCols; j++)
        {
            int sum = 0;
            for (int k = 0; k < sizeDepth; k++)
            {
                sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
            }
            pixelsGray[i * sizeCols + j] = (int)(sum / sizeDepth);
        }
    }
    return pixelsGray;
}

__global__ void gaussianBlur(int *inputPixels, int *blurredPixels, int sizeRows, int sizeCols, int sizeDepth)
{
    double kernel[5][5] = {{2.0, 4.0, 5.0, 4.0, 2.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {5.0, 12.0, 15.0, 12.0, 5.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);

    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate of the pixel
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate of the pixel
    int k = blockIdx.z * blockDim.z + threadIdx.z; // z-coordinate of the pixel

    if (i < sizeRows && j < sizeCols && k < sizeDepth)
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
        blurredPixels[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
    }
}

__global__ void rgbToGrayscale(int *blurredPixels, int *grayscaledPixels, int sizeRows, int sizeCols)
{
}

void cannyEdgeDetection(uint8_t *inputImage, double lowerThreshold, double higherThreshold, int width, int height, int channels)
{
    int *pixels = imgToArray(inputImage, height, width, channels);

    // cudaMalloc((void **)&pixelsPtr, height * width * channels * sizeof(int));

    // cudaMemcpy(pixelsPtr, pixels, height * width * channels * sizeof(int), cudaMemcpyHostToDevice);

    // dim3 blockSize(16, 16);
    // dim3 numBlocks((width + blockSize.x - 1) / blockSize.x,
    //                (height + blockSize.y - 1) / blockSize.y,
    //                (channels + blockSize.z - 1) / blockSize.z);

    // // GAUSSIAN_BLUR:

    // gaussianBlur<<<numBlocks, blockSize>>>(pixelsPtr, pixelsPtr, height, width, channels);

    // // rgbToGrayscale<<<numBlocks, threadsPerBlock>>>(pixelsPtr, pixelsPtr, height, width);

    // cudaMemcpy(pixels, pixelsPtr, height * width * channels * sizeof(int), cudaMemcpyDeviceToHost);

    // std::vector<int>
    //     pixelsBlur = gaussianBlur(pixels, kernel, kernelConst, height, width, channels);

    // // GRAYSCALE:

    // std::vector<int> pixelsGray = rgbToGrayscale(pixelsBlur, height, width, channels);

    // // CANNY_FILTER:

    // std::vector<int> pixelsCanny = cannyFilter(pixelsGray, height, width, 1, lowerThreshold, higherThreshold);

    uint8_t *outputImage = (unsigned char *)malloc(width * height);
    arrayToImg(pixelsCanny, outputImage, height, width, 1);

    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

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
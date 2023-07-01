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

__global__ void gaussianBlur(int *inputPixels, int *blurredPixels, int sizeRows, int sizeCols, int sizeDepth)
{
    double kernel[5][5] = {{2.0, 4.0, 5.0, 4.0, 2.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {5.0, 12.0, 15.0, 12.0, 5.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

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

__global__ void rgbToGrayscale(int *blurredPixels, int *grayscaledPixels, int sizeRows, int sizeCols, int sizeDepth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int sum = 0;
    if (i < sizeRows && j < sizeCols)
    {
        for (int k = 0; k < sizeDepth; k++)
        {
            sum = sum + blurredPixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
        }
        grayscaledPixels[i * sizeCols + j] = (int)(sum / sizeDepth);
    }
}

__global__ void cannyFilter(int *cannyPixels, int *grayscaledPixels, int sizeRows, int sizeCols, double *G, int *theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;

    // perform canny edge detection on everything but the edges

    // find gx and gy for each pixel
    double gxValue = 0;
    double gyValue = 0;
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            gxValue = gxValue + (gx[1 - x][1 - y] * (double)(grayscaledPixels[(i + x) * sizeCols + j + y]));
            gyValue = gyValue + (gy[1 - x][1 - y] * (double)(grayscaledPixels[(i + x) * sizeCols + j + y]));
        }
    }

    // calculate G and theta
    G[i * sizeCols + j] = std::sqrt(std::pow(gxValue, 2) + std::pow(gyValue, 2));
    double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
    theta[i * sizeCols + j] = (int)(180.0 + atanResult);

    // setting the edges
    if (i == 1)
    {
        G[i * sizeCols + j - 1] = G[i * sizeCols + j];
        theta[i * sizeCols + j - 1] = theta[i * sizeCols + j];
    }
    else if (j == 1)
    {
        G[(i - 1) * sizeCols + j] = G[i * sizeCols + j];
        theta[(i - 1) * sizeCols + j] = theta[i * sizeCols + j];
    }
    else if (i == sizeRows - 1)
    {
        G[i * sizeCols + j + 1] = G[i * sizeCols + j];
        theta[i * sizeCols + j + 1] = theta[i * sizeCols + j];
    }
    else if (j == sizeCols - 1)
    {
        G[(i + 1) * sizeCols + j] = G[i * sizeCols + j];
        theta[(i + 1) * sizeCols + j] = theta[i * sizeCols + j];
    }

    // setting the corners
    if (i == 1 && j == 1)
    {
        G[(i - 1) * sizeCols + j - 1] = G[i * sizeCols + j];
        theta[(i - 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
    }
    else if (i == 1 && j == sizeCols - 1)
    {
        G[(i - 1) * sizeCols + j + 1] = G[i * sizeCols + j];
        theta[(i - 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
    }
    else if (i == sizeRows - 1 && j == 1)
    {
        G[(i + 1) * sizeCols + j - 1] = G[i * sizeCols + j];
        theta[(i + 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
    }
    else if (i == sizeRows - 1 && j == sizeCols - 1)
    {
        G[(i + 1) * sizeCols + j + 1] = G[i * sizeCols + j];
        theta[(i + 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
    }

    // round to the nearest 45 degrees
    theta[i * sizeCols + j] = theta[i * sizeCols + j] / 45 * 45;

    // non-maximum suppression

    if (theta[i * sizeCols + j] == 0 || theta[i * sizeCols + j] == 180)
    {
        if (G[i * sizeCols + j] < G[i * sizeCols + j - 1] || G[i * sizeCols + j] < G[i * sizeCols + j + 1])
        {
            G[i * sizeCols + j] = 0;
        }
    }
    else if (theta[i * sizeCols + j] == 45 || theta[i * sizeCols + j] == 225)
    {
        if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j + 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j - 1])
        {
            G[i * sizeCols + j] = 0;
        }
    }
    else if (theta[i * sizeCols + j] == 90 || theta[i * sizeCols + j] == 270)
    {
        if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j])
        {
            G[i * sizeCols + j] = 0;
        }
    }
    else
    {
        if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j - 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j + 1])
        {
            G[i * sizeCols + j] = 0;
        }
    }

    cannyPixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / 511.517));

    // double threshold
    bool changes;
    do
    {
        changes = false;

        if (G[i * sizeCols + j] < (lowerThreshold * 511.517))
        {
            G[i * sizeCols + j] = 0;
        }
        else if (G[i * sizeCols + j] >= (higherThreshold * 511.517))
        {
            continue;
        }
        else if (G[i * sizeCols + j] < (higherThreshold * 511.517))
        {
            int tempG = G[i * sizeCols + j];
            G[i * sizeCols + j] = 0;
            for (int x = -1; x <= 1; x++)
            {
                bool breakNestedLoop = false;
                for (int y = -1; y <= 1; y++)
                {
                    if (x == 0 && y == 0)
                    {
                        continue;
                    }
                    if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * 511.517))
                    {
                        G[i * sizeCols + j] = (higherThreshold * 511.517);
                        changes = true;
                        breakNestedLoop = true;
                        break;
                    }
                }
                if (breakNestedLoop)
                {
                    break;
                }
            }
        }
        cannyPixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / 511.517));

    } while (changes);
}

void cannyEdgeDetection(uint8_t *inputImage, double lowerThreshold, double higherThreshold, int width, int height, int channels)
{
    int *pixels = imgToArray(inputImage, height, width, channels);
    int *pixelsPtr;
    int *blurredPixelsPtr;
    int *grayscalePixelsPtr;
    int *cannyPixelsPtr;

    cudaMalloc((void **)&pixelsPtr, height * width * channels * sizeof(int));
    cudaMalloc((void **)&blurredPixelsPtr, height * width * channels * sizeof(int));
    cudaMalloc((void **)&grayscalePixelsPtr, height * width * channels * sizeof(int));
    cudaMalloc((void **)&cannyPixelsPtr, height * width * channels * sizeof(int));

    cudaMemcpy(pixelsPtr, pixels, height * width * channels * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x,
                   (height + blockSize.y - 1) / blockSize.y,
                   (channels + blockSize.z - 1) / blockSize.z);

    gaussianBlur<<<numBlocks, blockSize>>>(pixelsPtr, blurredPixelsPtr, height, width, channels);
    std::cout << "Exited gaussian" << std::endl;

    rgbToGrayscale<<<numBlocks, blockSize>>>(blurredPixelsPtr, grayscalePixelsPtr, height, width, channels);
    std::cout << "Exited grayscale" << std::endl;

    double *gradientPtr;
    int *thetaPtr;

    cudaMalloc((void **)&gradientPtr, height * width * sizeof(double));
    cudaMalloc((void **)&thetaPtr, height * width * sizeof(int));

    cannyFilter<<<numBlocks, blockSize>>>(grayscalePixelsPtr, cannyPixelsPtr, height, width, gradientPtr, thetaPtr);
    std::cout << "Exited cannyFilter" << std::endl;

    int *outputPixels = (int *)malloc(width * height * sizeof(int));
    cudaMemcpy(outputPixels, cannyPixelsPtr, height * width * sizeof(int), cudaMemcpyDeviceToHost);

    uint8_t *outputImage = (unsigned char *)malloc(width * height);
    arrayToImg(pixels, outputImage, height, width, 1);

    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

    free(outputImage);
    cudaFree(pixelsPtr);
    cudaFree(gradientPtr);
    cudaFree(thetaPtr);
    cudaFree(grayscalePixelsPtr);
    cudaFree(blurredPixelsPtr);
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
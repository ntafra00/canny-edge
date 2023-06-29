#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurKernel(const uint8_t *inputImage, int sizeRows, int sizeCols, int sizeDepth,
                                   const double *kernel, double kernelConst, int kernelSize, int *outputPixels)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sizeRows && col < sizeCols)
    {
        for (int k = 0; k < sizeDepth; k++)
        {
            double sum = 0;
            double sumKernel = 0;
            for (int i = -2; i <= 2; i++)
            {
                for (int j = -2; j <= 2; j++)
                {
                    int rowIndex = row + i;
                    int colIndex = col + j;
                    if (rowIndex >= 0 && rowIndex < sizeRows && colIndex >= 0 && colIndex < sizeCols)
                    {
                        double channel = (double)inputImage[(rowIndex * sizeCols + colIndex) * sizeDepth + k];
                        sum += channel * kernelConst * kernel[(i + 2) * kernelSize + (j + 2)];
                        sumKernel += kernelConst * kernel[(i + 2) * kernelSize + (j + 2)];
                    }
                }
            }
            outputPixels[(row * sizeCols + col) * sizeDepth + k] = (int)(sum / sumKernel);
        }
    }
}

// CUDA kernel for RGB to grayscale conversion
__global__ void rgbToGrayscaleKernel(const int *inputPixels, int sizeRows, int sizeCols, int sizeDepth, int *outputPixels)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sizeRows && col < sizeCols)
    {
        int sum = 0;
        for (int k = 0; k < sizeDepth; k++)
        {
            sum += inputPixels[(row * sizeCols + col) * sizeDepth + k];
        }
        outputPixels[row * sizeCols + col] = sum / sizeDepth;
    }
}

// CUDA kernel for Canny edge detection
__global__ void cannyFilterKernel(const int *inputPixels, int sizeRows, int sizeCols, int sizeDepth,
                                  double lowerThreshold, double higherThreshold, double largestG, int *outputPixels)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > 0 && row < sizeRows - 1 && col > 0 && col < sizeCols - 1)
    {
        // Find gx and gy for each pixel
        int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        double gxValue = 0;
        double gyValue = 0;
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                gxValue += inputPixels[((row + i) * sizeCols + (col + j)) * sizeDepth] * gx[i + 1][j + 1];
                gyValue += inputPixels[((row + i) * sizeCols + (col + j)) * sizeDepth] * gy[i + 1][j + 1];
            }
        }

        // Calculate gradient magnitude and direction
        double gradientMagnitude = sqrt(gxValue * gxValue + gyValue * gyValue);
        double gradientDirection = atan2(gyValue, gxValue) * 180 / 3.14159;

        // Non-maximum suppression
        double q = 255;
        double r = 0;
        if ((gradientDirection < 22.5 && gradientDirection >= -22.5) || (gradientDirection >= 157.5 || gradientDirection < -157.5))
        {
            double gradientLeft = inputPixels[((row - 1) * sizeCols + col) * sizeDepth];
            double gradientRight = inputPixels[((row + 1) * sizeCols + col) * sizeDepth];
            if (gradientMagnitude >= gradientLeft && gradientMagnitude >= gradientRight)
            {
                q = gradientMagnitude;
            }
            else
            {
                r = gradientMagnitude;
            }
        }
        else if ((gradientDirection >= 22.5 && gradientDirection < 67.5) || (gradientDirection < -112.5 && gradientDirection >= -157.5))
        {
            double gradientTopRight = inputPixels[((row - 1) * sizeCols + (col + 1)) * sizeDepth];
            double gradientBottomLeft = inputPixels[((row + 1) * sizeCols + (col - 1)) * sizeDepth];
            if (gradientMagnitude >= gradientTopRight && gradientMagnitude >= gradientBottomLeft)
            {
                q = gradientMagnitude;
            }
            else
            {
                r = gradientMagnitude;
            }
        }
        else if ((gradientDirection >= 67.5 && gradientDirection < 112.5) || (gradientDirection < -67.5 && gradientDirection >= -112.5))
        {
            double gradientTop = inputPixels[((row - 1) * sizeCols + col) * sizeDepth];
            double gradientBottom = inputPixels[((row + 1) * sizeCols + col) * sizeDepth];
            if (gradientMagnitude >= gradientTop && gradientMagnitude >= gradientBottom)
            {
                q = gradientMagnitude;
            }
            else
            {
                r = gradientMagnitude;
            }
        }
        else if ((gradientDirection >= 112.5 && gradientDirection < 157.5) || (gradientDirection < -22.5 && gradientDirection >= -67.5))
        {
            double gradientTopLeft = inputPixels[((row - 1) * sizeCols + (col - 1)) * sizeDepth];
            double gradientBottomRight = inputPixels[((row + 1) * sizeCols + (col + 1)) * sizeDepth];
            if (gradientMagnitude >= gradientTopLeft && gradientMagnitude >= gradientBottomRight)
            {
                q = gradientMagnitude;
            }
            else
            {
                r = gradientMagnitude;
            }
        }

        // Double thresholding
        int pixelValue;
        if (gradientMagnitude >= higherThreshold)
        {
            pixelValue = 255;
        }
        else if (gradientMagnitude >= lowerThreshold && gradientMagnitude < higherThreshold)
        {
            if (gradientMagnitude >= largestG)
            {
                largestG = gradientMagnitude;
            }
            pixelValue = 255;
        }
        else
        {
            pixelValue = 0;
        }

        outputPixels[row * sizeCols + col] = pixelValue;
    }
}

int main()
{
    const char *inputImageFile = "input.png";
    const char *outputImageFile = "output.png";

    int width, height, channels;
    uint8_t *inputImage = stbi_load(inputImageFile, &width, &height, &channels, 0);
    if (!inputImage)
    {
        std::cout << "Error loading image: " << inputImageFile << std::endl;
        return -1;
    }

    int sizeRows = height;
    int sizeCols = width;
    int sizeDepth = channels;

    // Allocate memory for input and output image pixels
    size_t imageSize = sizeRows * sizeCols * sizeDepth * sizeof(uint8_t);
    uint8_t *d_inputImage;
    cudaMalloc((void **)&d_inputImage, imageSize);
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    int *d_outputPixels;
    cudaMalloc((void **)&d_outputPixels, sizeRows * sizeCols * sizeof(int));

    // Define Gaussian blur kernel
    double sigma = 1.4;
    int kernelSize = 5;
    double *kernel = (double *)malloc(kernelSize * kernelSize * sizeof(double));
    double sumKernel = 0;
    int kernelRadius = kernelSize / 2;
    for (int i = -kernelRadius; i <= kernelRadius; i++)
    {
        for (int j = -kernelRadius; j <= kernelRadius; j++)
        {
            double r = sqrt(i * i + j * j);
            kernel[(i + kernelRadius) * kernelSize + (j + kernelRadius)] = exp(-(r * r) / (2 * sigma * sigma));
            sumKernel += kernel[(i + kernelRadius) * kernelSize + (j + kernelRadius)];
        }
    }
    double kernelConst = 1.0 / sumKernel;

    double lowerThreshold = 20;
    double higherThreshold = 40;
    double largestG = 0;

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((sizeCols + blockSize.x - 1) / blockSize.x, (sizeRows + blockSize.y - 1) / blockSize.y);

    // Apply Gaussian blur
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_inputImage, sizeRows, sizeCols, sizeDepth, kernel, kernelConst, kernelSize, d_outputPixels);

    // Convert RGB to grayscale
    rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_outputPixels, sizeRows, sizeCols, sizeDepth, d_outputPixels);

    // Apply Canny edge detection
    cannyFilterKernel<<<gridSize, blockSize>>>(d_outputPixels, sizeRows, sizeCols, sizeDepth, lowerThreshold, higherThreshold, largestG, d_outputPixels);

    // Copy output pixels from device to host memory
    int *outputPixels = (int *)malloc(sizeRows * sizeCols * sizeof(int));
    cudaMemcpy(outputPixels, d_outputPixels, sizeRows * sizeCols * sizeof(int), cudaMemcpyDeviceToHost);

    // Save output image
    stbi_write_png(outputImageFile, sizeCols, sizeRows, 1, outputPixels, sizeCols);

    // Free memory
    stbi_image_free(inputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputPixels);
    free(kernel);
    free(outputPixels);

    return 0;
}

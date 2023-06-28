#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel for RGB to grayscale conversion
__global__ void convertRGBToGrayscaleGPU(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < width * height; i += stride)
    {
        unsigned char r = inputImage[3 * i];
        unsigned char g = inputImage[3 * i + 1];
        unsigned char b = inputImage[3 * i + 2];
        outputImage[i] = (unsigned char)(0.2989 * r + 0.5870 * g + 0.1140 * b);
    }
}

// CUDA kernel for Gaussian filter
__global__ void applyGaussianFilterGPU(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
{
    // Calculate the offset based on kernel size
    int offset = KERNEL_SIZE / 2;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= offset && col < width - offset && row >= offset && row < height - offset)
    {
        // Generate the Gaussian kernel and apply the filter
        // ... (code for applying the Gaussian filter on GPU)
    }
}

// CUDA kernel for intensity gradient calculation
__global__ void calculateIntensityGradientGPU(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
{
    // Sobel operator kernels
    const int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};

    const int sobelY[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= 1 && col < width - 1 && row >= 1 && row < height - 1)
    {
        int gx = 0;
        int gy = 0;

        // Convolve with Sobel kernels
        for (int k = -1; k <= 1; k++)
        {
            for (int l = -1; l <= 1; l++)
            {
                gx += inputImage[(row + k) * width + (col + l)] * sobelX[k + 1][l + 1];
                gy += inputImage[(row + k) * width + (col + l)] * sobelY[k + 1][l + 1];
            }
        }

        int gradientMagnitude = sqrt(gx * gx + gy * gy);

        if (gradientMagnitude >= HIGH_THRESHOLD)
        {
            outputImage[row * width + col] = 255; // Strong edge
        }
        else if (gradientMagnitude >= LOW_THRESHOLD)
        {
            // Check if any neighbor is a strong edge
            int strongEdgeFound = 0;
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    if (outputImage[(row + k) * width + (col + l)] == 255)
                    {
                        strongEdgeFound = 1;
                        break;
                    }
                }
            }
            if (strongEdgeFound)
            {
                outputImage[row * width + col] = 255; // Track the edge
            }
            else
            {
                outputImage[row * width + col] = 0; // Suppress weak edge
            }
        }
        else
        {
            outputImage[row * width + col] = 0; // Non-edge
        }
    }
}

int performCannyEdgeDetectionGPU(unsigned char *inputImage, unsigned char *outputImage, int width, int height)
{
    // Allocate memory on the GPU
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;
    cudaMalloc((void **)&d_inputImage, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_outputImage, width * height * sizeof(unsigned char));

    // Copy input image from host to GPU
    cudaMemcpy(d_inputImage, inputImage, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Convert RGB to grayscale on GPU
    dim3 gridSize((width * height + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);
    convertRGBToGrayscaleGPU<<<gridSize, blockSize>>>(width, height, d_outputImage, d_inputImage);

    // Apply Gaussian filter on GPU
    gridSize = dim3((width + 15) / 16, (height + 15) / 16, 1);
    blockSize = dim3(16, 16, 1);
    applyGaussianFilterGPU<<<gridSize, blockSize>>>(width, height, d_outputImage, d_inputImage);

    // Calculate intensity gradient on GPU
    gridSize = dim3((width + 15) / 16, (height + 15) / 16, 1);
    blockSize = dim3(16, 16, 1);
    calculateIntensityGradientGPU<<<gridSize, blockSize>>>(width, height, d_outputImage, d_inputImage);

    // Copy the result from GPU to host
    cudaMemcpy(outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}

int main()
{
    const char *inputImagePath = "input-cpu.jpg";
    const char *outputImagePath = "output-cpu.jpg";
    int width, height, channels;

    // Load the input image
    unsigned char *inputImage = stbi_load(inputImagePath, &width, &height, &channels, STBI_rgb);
    if (inputImage == NULL)
    {
        printf("Failed to load image: %s\n", inputImagePath);
        return -1;
    }

    if (width < 1400 || height < 1400)
    {
        printf("Choose image with greater resolution\n");
        return -1;
    }

    if (channels != 3)
    {
        printf("Images is not in RGB format\n");
        return -1;
    }

    // Allocate memory for the output image
    unsigned char *outputImage = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    // Apply Canny edge detection using GPU
    performCannyEdgeDetectionGPU(inputImage, outputImage, width, height);

    // Write the output image to file
    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

    // Free the allocated memory
    stbi_image_free(inputImage);
    free(outputImage);

    return 0;
}
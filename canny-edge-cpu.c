#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int KERNEL_SIZE = 20;
int LOW_THRESHOLD = 50;
int HIGH_THRESHOLD = 150;
float SIGMA = 1.0; // Adjust the SIGMA value as needed

int convertRGBToGrayscale(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
{
    int imageSize = width * height;

    for (int i = 0; i < imageSize; i++)
    {
        unsigned char r = inputImage[3 * i];
        unsigned char g = inputImage[3 * i + 1];
        unsigned char b = inputImage[3 * i + 2];
        // applying luminosity formula for grayscale
        outputImage[i] = (unsigned char)(0.2989 * r + 0.5870 * g + 0.1140 * b);
    }

    return 0;
}

int applyGaussianFilter(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
{
    int offset = KERNEL_SIZE / 2;
    float *kernel = (float *)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // Generate the Gaussian kernel
    float sum = 0.0;
    for (int i = -offset; i <= offset; i++)
    {
        for (int j = -offset; j <= offset; j++)
        {
            int index = (i + offset) * KERNEL_SIZE + (j + offset);
            kernel[index] = exp(-(i * i + j * j) / (2 * SIGMA * SIGMA));
            sum += kernel[index];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++)
    {
        kernel[i] /= sum;
    }

    // Create a temporary image to store the filtered result
    unsigned char *tempImage = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    // Apply the Gaussian filter
    for (int row = offset; row < height - offset; row++)
    {
        for (int col = offset; col < width - offset; col++)
        {
            float sum = 0.0;
            for (int k = -offset; k <= offset; k++)
            {
                for (int l = -offset; l <= offset; l++)
                {
                    int imageIndex = (row + k) * width + (col + l);
                    int kernelIndex = (k + offset) * KERNEL_SIZE + (l + offset);
                    sum += inputImage[imageIndex] * kernel[kernelIndex];
                }
            }
            outputImage[row * width + col] = (unsigned char)round(sum);
        }
    }

    // Free memory
    free(kernel);

    return 0;
}

int calculateIntensityGradient(int width, int height, unsigned char *outputImage, unsigned char *inputImage)
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

    // Temporary gradient images
    int *gradientX = (int *)malloc(width * height * sizeof(int));
    int *gradientY = (int *)malloc(width * height * sizeof(int));

    // Apply Sobel operator to calculate gradients in X and Y directions
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            int gx = 0;
            int gy = 0;

            // Convolve with Sobel kernels
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    gx += inputImage[(i + k) * width + (j + l)] * sobelX[k + 1][l + 1];
                    gy += inputImage[(i + k) * width + (j + l)] * sobelY[k + 1][l + 1];
                }
            }

            gradientX[i * width + j] = gx;
            gradientY[i * width + j] = gy;
        }
    }

    // Calculate gradient magnitude and apply double thresholding
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            int gx = gradientX[i * width + j];
            int gy = gradientY[i * width + j];
            int gradientMagnitude = sqrt(gx * gx + gy * gy);

            if (gradientMagnitude >= HIGH_THRESHOLD)
            {
                outputImage[i * width + j] = 255; // Strong edge
            }
            else if (gradientMagnitude >= LOW_THRESHOLD)
            {
                // Check if any neighbor is a strong edge
                int strongEdgeFound = 0;
                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        if (outputImage[(i + k) * width + (j + l)] == 255)
                        {
                            strongEdgeFound = 1;
                            break;
                        }
                    }
                }
                if (strongEdgeFound)
                {
                    outputImage[i * width + j] = 255; // Track the edge
                }
                else
                {
                    outputImage[i * width + j] = 0; // Suppress weak edge
                }
            }
            else
            {
                outputImage[i * width + j] = 0; // Non-edge
            }
        }
    }

    // Free allocated memory
    free(gradientX);
    free(gradientY);

    return 0;
}

int performCannyEdgeDetection(unsigned char *inputImage, unsigned char *outputImage, int width, int height)
{
    // convert given RGB image to grayscale
    convertRGBToGrayscale(width, height, outputImage, inputImage);

    // apply gaussian filter in order to reduce noise
    applyGaussianFilter(width, height, outputImage, inputImage);

    // // finding intensity gradient of the image and applying gradient magnitude thresholding to get rid of spurious response to edge detection
    // calculateIntensityGradient(width, height, outputImage, inputImage);

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
    unsigned char *outputImage = (unsigned char *)malloc(width * height);

    // Apply canny edge filter operation

    performCannyEdgeDetection(inputImage, outputImage, width, height);
    printf("Exited canny edge detection");
    // Write the output image to file
    stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

    // Free the allocated memory
    stbi_image_free(inputImage);
    free(outputImage);

    return 0;
}

from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt

# Path to image dataset folder
input_folder = 'C:/Users/ujjwol/OneDrive - Tribhuvan University/Desktop/BEI IV_I/Data Mining/Lab Report/Project/Face Dataset'

# Output folder for processed images
output_folder = 'C:/Users/ujjwol/OneDrive - Tribhuvan University/Desktop/BEI IV_I/Data Mining/Lab Report/Project/Processed Image'

# Desired size for the output images
output_size = (256, 256)

# Initialize lists to store images and vectors
image_vectors = []

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each image in the input folder
for filename in os.listdir(input_folder):
    image_path = os.path.join(input_folder, filename)
    
    # Open the image using Pillow
    img = Image.open(image_path)
        
    # Convert the image to grayscale
    img_gray = img.convert('L')
        
    # Enhance contrast to improve lighting uniformity
    img_enhanced = ImageOps.equalize(img_gray)
        
    # Resize the image to the desired size
    img_resized = img_gray.resize(output_size, Image.LANCZOS)
        
    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, filename)
    img_resized.save(output_path)
    
    # Convert the processed image to a vector
    img_vector = np.array(img_resized).flatten()
    
    # Append the image vector to the list
    image_vectors.append(img_vector)
    
    # Create a single matrix from the list of image vectors
    data_matrix = np.vstack(image_vectors)

num_images = len(image_vectors)
num_images

#Normalize the data
data_nor = (data_matrix-data_matrix.mean())/data_matrix.std()

#Calculate covariance matrix using formula
S_dat = (1/(len(data_nor)-1))*np.dot(data_nor, data_nor.T)

#Eigenvalues and Eigenvector calculation
eigenvalues, eigenvector = np.linalg.eig(S_dat)

#Sorting of eigenvalues and eigenvector in descending order
ind = np.argsort(eigenvalues)[::-1] #Get sorting indices 
eigVal_sorted = eigenvalues[ind]    #Sort eigenvalues according to indices
eigVec_sorted = eigenvector[:, ind]

Y_2d = np.dot(eigVec_sorted.T,data_nor)

num_images, num_pixels = data_matrix.shape

# Assuming Y_2d is your flattened dataset with shape (130, 16384)
num_images = Y_2d.shape[0]
image_size = 256

# Reshape each row back into a 128x128 image
images = np.abs(Y_2d.reshape(num_images, image_size, image_size))

selected_index = 0

plt.imshow(images[selected_index], cmap='gray')  # Assuming grayscale images
plt.title(f"Image Index: {selected_index}")
plt.axis('off')
plt.show()

# Calculate the mean face vector
mean_face = np.mean(data_nor, axis=0)

# Reshape the array to a 2D image format (256x256)
image_size = (256, 256)
mean_face_image = mean_face.reshape(image_size)

# Plot the image
plt.imshow(mean_face_image, cmap='gray')  # Display the image using grayscale colormap
plt.axis('off')  # Turn off axes
plt.title('Mean Face Image')
plt.show()

# Create separate plots for PC1 vs PC2 and PC3 vs PC4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot PC1 vs PC2
ax1.scatter(Y_2d[:, 0], Y_2d[:, 1], c='b', marker='o')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('PC1 vs PC2')

# Plot PC3 vs PC4
ax2.scatter(Y_2d[:, 2], Y_2d[:, 3], c='r', marker='o')
ax2.set_xlabel('PC3')
ax2.set_ylabel('PC4')
ax2.set_title('PC3 vs PC4')

plt.tight_layout()
plt.show()

# Create a 3D scatter plot of the first three principal components
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each image in the 3D space using the first three principal components
for i in range(num_images):
    pc1 = Y_2d[i, 0]
    pc2 = Y_2d[i, 1]
    pc3 = Y_2d[i, 2]
    ax.scatter(pc1, pc2, pc3, c='b', marker='o')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title("Images in First Three Principal Components")
plt.show()

# Reconstruct face images using the first 'num_pcs' principal components
num_pcs = 50  # Change this to the desired number of principal components
reconstructed_images_vec = np.dot(eigVec_sorted[:, :num_pcs], Y_2d[:num_pcs, :])
reconstructed_images = np.abs((reconstructed_images_vec * data_matrix.std()) + data_matrix.mean())

selected_index = 0

# Plot the original and reconstructed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(data_matrix[selected_index].reshape(output_size), cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_images[selected_index].reshape(output_size), cmap='gray')
plt.title(f"Reconstructed Image with {num_pcs} PCs")
plt.axis('off')

plt.tight_layout()
plt.show()
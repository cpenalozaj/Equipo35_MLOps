# Build the Docker image
docker build -t student-performance-api .

# Run the Docker container
docker run -p 8000:8000 student-performance-api
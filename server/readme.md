# Build the Docker image
docker build -t student-performance-api .

# Run the Docker container
docker run -p 8000:8000 student-performance-api

# Test the API

```bash
curl --location 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "features": {
        "ge": "F",
        "cst": "G",
        "tnp": "Good",
        "twp": "Good",
        "iap": "Vg",
        "arr": "Y",
        "ls": "V",
        "as_": "Paid",
        "fmi": "Medium",
        "fs": "Average",
        "fq": "Um",
        "mq": 10,
        "fo": "Farmer",
        "mo": "Housewife",
        "nf": "Large",
        "sh": "Poor",
        "ss": "Govt",
        "me": "Asm",
        "tt": "Small",
        "atd": "Good"
    }
}'
```


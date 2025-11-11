# MLOps Weekly Assignments – IRIS Model Pipeline

This repository contains an end-to-end MLOps pipeline that trains, versions, packages, deploys, and monitors an ML model using industry-standard tools.

### Key Features  
- DVC	tracks dataset and trained model versions (data.csv + model.joblib)
- GitHub Actions automate CI/CD on push (tests, build, deployment triggers)
- FastAPI app	exposes the trained model as a prediction API
- Dockerfile packages app into a reproducible container
- Kubernetes deploys scalable inference endpoint (with autoscaling support)
- wrk load testing tests performance and scaling behavior

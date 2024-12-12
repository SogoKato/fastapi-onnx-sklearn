# fastapi-onnx-sklearn

Blog (Japanese): https://sogo.dev/posts/2024/12/onnx-api

## Getting started

### Prerequisites

* [rye](https://rye.astral.sh/)

### Run locally

1. Create a virtual environment and install dependencies, unless you have not run `rye sync` yet.
   ```
   rye sync --no-dev
   ```
1. Train a model.
   ```
   rye run python train.py
   ```
1. Run API server.
   ```
   rye run uvicorn main:app
   ```
1. Predict!
   ```
   curl -s -X POST http://localhost:8000/prediction -F "file=@$(pwd)/examples/dog/dog1.png;type=image/jpeg" | jq
   ```
   output:
   ```json
   {
     "result": [
       {
         "label": "dog",
         "probability": 0.6963527798652649
       },
       {
         "label": "cat",
         "probability": 0.17074337601661682
       },
       {
         "label": "horse",
         "probability": 0.06515106558799744
       },
       {
         "label": "truck",
         "probability": 0.03153929114341736
       },
       {
         "label": "deer",
         "probability": 0.012792888097465038
       },
       {
         "label": "automobile",
         "probability": 0.010493496432900429
       },
       {
         "label": "ship",
         "probability": 0.005543916951864958
       },
       {
         "label": "airplane",
         "probability": 0.003050298895686865
       },
       {
         "label": "frog",
         "probability": 0.0022715579252690077
       },
       {
         "label": "bird",
         "probability": 0.002061390085145831
       }
     ]
   }
   ```

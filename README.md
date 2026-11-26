# TrashNet MLOps for Waste Classification

An image classification application for six waste categories, with a Streamlit dashboard, FastAPI inference service, model registry, and retraining/promotion workflow. The classifier is based on TensorFlow/Keras and MobileNetV2.

## Features

- Classify a JPEG, PNG, or WebP image and display the predicted category, confidence, and class probabilities.
- View API and production-model health, uptime, registry entries, and evaluation metrics.
- Upload labelled images individually or as a ZIP archive.
- Fine-tune the registered production model, evaluate the candidate against an independent holdout dataset, and promote it only when it passes the configured criteria.
- Run the API and dashboard together with Docker Compose.

The class labels and model output order are:

`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`

## Application architecture

```text
Browser
  └── Streamlit dashboard (:8501)
        └── FastAPI service (:8000)
              ├── Registered production Keras model
              └── Image prediction API

Streamlit retraining page
  └── MLOps pipeline → train/validation split → candidate evaluation
                       → model registry promotion decision
```

With Docker Compose, the dashboard reaches the API at `http://api:8000` over the private Compose network. The API loads the production model when it starts. After a model is promoted, restart the API so it loads the new production artifact:

```bash
docker compose restart api
```

## Run locally with Python

Requirements: Python 3.11 is recommended. TensorFlow may require additional system libraries on some platforms.

From the repository root, create and activate a virtual environment, then install dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the FastAPI service in one terminal:

```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Start the Streamlit dashboard in a second terminal, with the same virtual environment activated:

```powershell
streamlit run web_app/app.py
```

Open <http://localhost:8501> for the dashboard and <http://localhost:8000/docs> for interactive API documentation. The dashboard defaults to `http://127.0.0.1:8000`; set `API_URL` if the API is on another address.

## Run with Docker Compose

Requirements: Docker Engine or Docker Desktop with Docker Compose.

From the repository root:

```bash
docker compose up --build -d
```

Open:

- Dashboard: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>
- API health: <http://localhost:8000/health>

Useful commands:

```bash
docker compose ps
docker compose logs -f
docker compose restart api
docker compose down
```

Compose stores model artifacts/registry and application data in named volumes. `docker compose down` leaves those volumes in place. **Do not run `docker compose down --volumes` unless you intend to delete that persisted data.**

If a host port is already in use, create a local `.env` file beside `compose.yaml` and choose free ports:

```dotenv
API_PORT=8001
DASHBOARD_PORT=8502
```

Compose maps those host ports to the same internal container ports. The dashboard-to-API URL remains `http://api:8000` inside Compose. `.env` is ignored by Git; do not commit secrets.

## Dataset layout for retraining

The training and holdout datasets are not included in the Docker image or Git repository. They must be available under `data/` locally, or `/app/data/` in the containers. Both datasets need separate folders for all six labels:

```text
data/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── holdout/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

Put valid images in each class folder. The holdout set must contain at least one image for every class. Keep the holdout data independent of training data; it is used to evaluate both production and candidate models.

The pipeline also requires enough training examples in every class for a stratified train/validation split. If a class folder is missing, the pipeline stops with a `Missing training folder` error. Empty class folders do not make a valid training dataset.

### Transfer datasets to an AWS EC2 deployment using S3

This is an alternative to SSH/SCP for instances without an assigned key pair.

1. From Git Bash on your computer, package the two data directories:

   ```bash
   tar -czf trashnet-datasets.tar.gz -C data train holdout
   ```

2. Create a private S3 bucket in the same AWS Region as the EC2 instance, keep Block Public Access enabled, and upload `trashnet-datasets.tar.gz`.

3. Attach an IAM role to the EC2 instance that allows `s3:GetObject` on this bucket and `s3:ListBucket` on the bucket itself. Avoid putting long-lived AWS access keys on the instance. See [attaching an IAM role to an EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/attach-iam-role.html) and [using the EC2 instance role with AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-metadata.html).

4. In the EC2 terminal, install the AWS CLI and download/extract the archive:

   ```bash
   sudo apt-get update
   sudo apt-get install -y awscli
   aws s3 cp s3://YOUR_BUCKET_NAME/trashnet-datasets.tar.gz /home/ubuntu/trashnet-datasets.tar.gz
   mkdir -p /home/ubuntu/trashnet-datasets
   tar -xzf /home/ubuntu/trashnet-datasets.tar.gz -C /home/ubuntu/trashnet-datasets
   ```

5. From the project directory on EC2, copy both directories into the shared Compose data volume:

   ```bash
   cd ~/MLOps-for-TrashNet-Classification
   sudo docker compose cp /home/ubuntu/trashnet-datasets/train api:/app/data/
   sudo docker compose cp /home/ubuntu/trashnet-datasets/holdout api:/app/data/
   sudo docker compose exec api ls /app/data/train
   sudo docker compose exec api ls /app/data/holdout
   ```

Each listing should contain all six class folders. The `api` and `dashboard` services share the `app_data` volume, so the retraining page can read these folders.

## MLOps retraining workflow

The dashboard’s **Upload and Retrain** page supports individual image uploads or a ZIP whose paths include category folders, for example `cardboard/image1.jpg` and `plastic/image2.jpg`. These new samples are stored with status `pending`. The current page warns that uploaded samples are not automatically included in training; the pipeline uses original training images and approved uploaded records.

When **Start Retraining Pipeline** is selected, the workflow:

1. Reads the current production version from `models/registry/model_registry.json`.
2. Loads the images in `data/train` and any approved uploaded samples, removes duplicates, and creates a stratified training/validation split.
3. Fine-tunes the production model and saves the best validation candidate.
4. Evaluates production and candidate models on the independent holdout set.
5. Registers the candidate and promotes it only if its macro-F1 improves and its accuracy is at least `0.80`; otherwise it is rejected.

Model files, registry metadata, uploaded-image records, and uploaded images are persisted in the Compose `models` and `app_data` volumes. On EC2, those volumes reside on the instance’s attached storage; back up that storage if the artifacts must survive instance replacement.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API/model readiness |
| `GET` | `/uptime` | API process uptime |
| `POST` | `/predict` | Classify an image uploaded as multipart field `file` |
| `POST` | `/retraining/upload` | Save a labelled image using multipart fields `category` and `file` |

The prediction and retraining image endpoints accept JPEG, PNG, and WebP images up to 10 MB. The API docs at `/docs` provide interactive request forms.

## AWS EC2 deployment notes

- Launch an Ubuntu EC2 instance with enough memory and disk for TensorFlow and model artifacts.
- Install Docker Engine and Compose using [Docker's Ubuntu installation instructions](https://docs.docker.com/engine/install/ubuntu/).
- In the security group, restrict SSH (`22`) to your IP and allow the dashboard (`8501`) only from intended clients. Do not add an inbound rule for the API (`8000`); Compose publishes it on the host, but AWS security groups should keep it inaccessible from the internet. See [EC2 security group guidance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-security-group.html).
- Start the app on EC2 with `sudo docker compose up --build -d`.
- The dashboard has no built-in login or HTTPS. For a private demo, restrict port `8501` to your IP. Before making it publicly accessible, put it behind HTTPS and add access control.
- Persist and back up the EC2 storage that contains Docker volumes. Docker volumes survive container recreation, but data may be lost if the underlying instance storage is deleted. See [EC2 storage options](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Storage.html).

## Repository layout

```text
src/
  main.py                 FastAPI application and endpoints
  prediction.py           Image preprocessing and prediction
  model.py                 MobileNetV2 model construction helpers
  preprocessing.py         TensorFlow dataset preprocessing helpers
  retraining.py            Upload storage, dataset preparation, and fine-tuning
  mlops_pipeline.py        Evaluation and model lifecycle workflow
  model_registry.py        Production/candidate registry operations
  model_promotion.py       Candidate promotion criteria
web_app/
  app.py                   Streamlit dashboard home
  pages/                   Prediction, monitoring, visualization, retraining
models/
  trashnet_best_model.h5   Initial production model
  class_names.json         Output label order
  registry/                Model registry metadata
tests/                     Unit and workflow tests
notebook/                  Exploratory notebook
Dockerfile                 Shared API/dashboard image
compose.yaml               Local and EC2 multi-container setup
requirements.txt           Python dependencies
```

## Tests

The repository tests use pytest. Install it in the active virtual environment if needed, then run:

```bash
python -m pip install pytest
python -m pytest
```

## Data and model files in Git

Do not commit `.venv/`, local secrets, SQLite files, uploads, or training/holdout datasets. The production model `models/trashnet_best_model.h5`, `models/class_names.json`, and `models/registry/model_registry.json` are required for a fresh deployment and should be committed. Candidate model binaries can stay in deployment storage rather than Git.

# TrashNet MLOps for Waste Classification

## Project description

TrashNet MLOps is an image classification application for sorting waste into six categories: cardboard, glass, metal, paper, plastic, and trash. It combines a TensorFlow/Keras MobileNetV2 classifier with a FastAPI prediction service and a Streamlit dashboard. The project also includes model monitoring, a model registry, a retraining pipeline, and load testing for the prediction API.

## Demo video

[Watch the TrashNet MLOps demo on YouTube](https://youtu.be/23TlNrnso6I?si=nR3MvjmdhPj5T7Fx)

## Application URLs

- **AWS dashboard:** [http://16.170.228.234:8501](http://16.170.228.234:8501) — EC2 public IP may change when the instance is stopped and started. Access is restricted by the EC2 security group.
- **Local dashboard:** [http://localhost:8501](http://localhost:8501)
- **Local API documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Local API health:** [http://localhost:8000/health](http://localhost:8000/health)

The AWS API port is not open for general browser access. It is opened only to the Locust client's IP while running a load test.

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
Browser --> Streamlit dashboard (:8501) --> NGINX gateway (:8000) --+
                                                                    +--> FastAPI replicas --> Production model
Locust load-test client ------------------> NGINX gateway (:8000) --+

Streamlit retraining page
  └── MLOps pipeline → train/validation split → candidate evaluation
                       → model registry promotion decision
```

With Docker Compose, the dashboard reaches the NGINX gateway at `http://gateway:8000`; the gateway routes requests to the API replicas. The API loads the production model when it starts. After a model is promoted, restart the API so it loads the new production artifact:

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

Compose maps those host ports to the same internal gateway/dashboard ports. Inside Compose, the dashboard calls the gateway at `http://gateway:8000`. `.env` is ignored by Git; do not commit secrets.

## Flood request simulation with Locust

`locustfile.py` repeatedly posts a valid image to `POST /predict` and checks for a successful JSON prediction. Request statistics include latency percentiles, requests per second, and failures. Locust can write CSV statistics and an HTML report; see the [Locust configuration reference](https://docs.locust.io/en/stable/config-options.html).

### Prepare the test client

Install Locust in a local Python environment, then place a valid JPEG, PNG, or WebP image at `load_test_images/sample.jpg` (or set `LOCUST_IMAGE` to another image path). This image is only a request fixture and should not be committed.

```bash
python -m pip install locust
mkdir -p load_test_images locust-results
```

Copy a valid image to `load_test_images/sample.jpg`. Keep the Locust client on the same machine and use the same image for each replica-count run.

### Prepare the EC2 deployment

Push the gateway and Locust changes to GitHub. In the EC2 terminal, pull and deploy them:

```bash
cd ~/MLOps-for-TrashNet-Classification
git pull origin main
sudo docker compose up --build -d --scale api=1
sudo docker compose ps
```

For the duration of the test only, add an EC2 security group inbound rule for custom TCP port `8000`, with source set to the Locust client's public IP (`My IP`). The port routes through the NGINX gateway to the API replicas. Remove this rule when testing is finished; the application normally needs no public API access.

### Run comparable tests

From your local project folder, run the same Locust load each time. Replace the example IP with the EC2 public IP:

```bash
locust -f locustfile.py --host http://16.170.228.234:8000 --headless -u 10 -r 2 -t 2m --csv locust-results/api-1 --html locust-results/api-1.html
```

Then change the API replica count on EC2 and repeat the exact Locust command, changing only the report name:

```bash
sudo docker compose up -d --scale api=2
sudo docker compose ps
```

Run `api=2` using `--csv locust-results/api-2 --html locust-results/api-2.html`. You can also compare `api=3` if the EC2 instance has enough memory and CPU. Docker Compose's `--scale` option sets the requested number of service containers; the gateway discovers the replicas through Docker DNS. [Docker Compose scaling reference](https://docs.docker.com/reference/cli/docker/compose/up/)

Use the same user count (`-u`), spawn rate (`-r`), duration (`-t`), image, EC2 instance, and software version for every run. Start with a modest load such as 10 users and increase only if the instance remains stable. The HTML reports and CSV files are written under `locust-results/` and ignored by Git.

### Results

**Measurement status: pending.** No Locust results have been collected for the deployed EC2 instance yet, so this README does not claim measured latency or throughput. After running the one-, two-, and three-container scenarios above, copy the request count, median latency, 95th-percentile latency, requests per second, and failures from each HTML/CSV report into this table.

| API containers | Requests | Median latency | 95th-percentile latency | Requests/sec | Failures |
|---:|---:|---:|---:|---:|---:|
| 1 | Pending | Pending | Pending | Pending | Pending |
| 2 | Pending | Pending | Pending | Pending | Pending |
| 3 | Pending | Pending | Pending | Pending | Pending |

When finished, return the deployment to one API replica and remove the temporary port `8000` security group rule:

```bash
sudo docker compose up -d --scale api=1
```

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
- In the security group, restrict SSH (`22`) to your IP and allow the dashboard (`8501`) only from intended clients. Keep API port `8000` closed to inbound traffic except temporarily from the Locust client's IP during load testing. See [EC2 security group guidance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-security-group.html).
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
nginx.conf                 Gateway that routes across API replicas
locustfile.py              Locust image-prediction load scenario
requirements.txt           Python dependencies
```

## Tests

The repository tests use pytest. Install it in the active virtual environment if needed, then run:

```bash
python -m pip install pytest
python -m pytest
```

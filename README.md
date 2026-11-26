# TrashNet MLOps

## Run with Docker Compose

Requirements: Docker Engine or Docker Desktop with the Compose plugin.

```powershell
docker compose up --build -d
```

Open the dashboard at <http://localhost:8501>. The API is available at
<http://localhost:8000>; its health endpoint is <http://localhost:8000/health>.
The Compose volumes retain the model registry, model artifacts, uploaded
retraining images, and SQLite database when containers are replaced.

To view service logs or stop the app:

```powershell
docker compose logs -f
docker compose down
```

After the retraining workflow promotes a model, restart the API container so it
loads the newly registered production artifact:

```powershell
docker compose restart api
```

The retraining pipeline also needs the training and holdout datasets under
`data/train` and `data/holdout`. They are intentionally excluded from the image;
provide them to the Docker host before running retraining. Prediction works
without those datasets.

You can change the host ports with `API_PORT` and `DASHBOARD_PORT`, for example
by defining them in a `.env` file beside `compose.yaml`.

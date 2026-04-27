# Deploy

This project is prepared for deployment in `/srv/tuberculosis-recognition` and exposure through `tuberculosis-recognition.pandamia.org`.

## 1. Clone the repository on the server

```bash
cd /srv
git clone <YOUR_REPOSITORY_URL> tuberculosis-recognition
cd /srv/tuberculosis-recognition
```

## 2. Upload the model weights

The model weights are not stored in the repository. On the server, the file must exist at:

```text
/srv/tuberculosis-recognition/source/model/model_weights.pth
```

Example upload from your local machine:

```bash
scp ./model_weights.pth <user>@<server>:/srv/tuberculosis-recognition/source/model/model_weights.pth
```

If the file already exists somewhere else on the server:

```bash
cp /path/to/model_weights.pth /srv/tuberculosis-recognition/source/model/model_weights.pth
```

## 3. Start the container

```bash
cd /srv/tuberculosis-recognition
docker compose up -d --build
```

The application will be available locally on the server at `127.0.0.1:18501`.

## 4. Connect the project through Caddy

The repository includes a ready-to-use site block in `Caddyfile`:

```caddyfile
tuberculosis-recognition.pandamia.org {
    encode zstd gzip
    reverse_proxy 127.0.0.1:18501
}
```

Add this block to your current Caddy configuration or include it using the import mechanism already used on your server.

After updating the Caddy configuration, reload or restart Caddy using the existing method already configured on the server.

## 5. DNS

The subdomain `tuberculosis-recognition.pandamia.org` must point to your server IP.

If you already use a wildcard record such as `*.pandamia.org`, you do not need to add a separate DNS record.

## 6. Update the project

```bash
cd /srv/tuberculosis-recognition
git pull
docker compose up -d --build
```

## 7. Useful commands

```bash
docker compose logs -f
docker compose ps
docker compose restart
```

## Notes

- The model weights are mounted into the container and are not baked into the Docker image.
- This setup does not interfere with other projects because the app is exposed only on `127.0.0.1:18501`.
- If you want to use a different subdomain or local port, update both `Caddyfile` and `docker-compose.yml` accordingly.

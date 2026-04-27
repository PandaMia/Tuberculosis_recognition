# Deploy

This project is prepared for deployment in `/srv/tuberculosis-recognition` and exposure through `https://tuberculosis-recognition.pandamia.org`.

Production setup:
- the app runs in Docker Compose
- Streamlit is published only on `127.0.0.1:18501`
- host-level Caddy listens on `80/443` and proxies the subdomain to `127.0.0.1:18501`

## 1. Clone the repository on the server

```bash
cd /srv
git clone <YOUR_REPOSITORY_URL> tuberculosis-recognition
cd /srv/tuberculosis-recognition
```

If `/srv` is not writable by your user:

```bash
sudo mkdir -p /srv/tuberculosis-recognition
sudo chown -R $USER:$USER /srv/tuberculosis-recognition
git clone <YOUR_REPOSITORY_URL> /srv/tuberculosis-recognition
cd /srv/tuberculosis-recognition
```

## 2. Upload the model weights

The model weights are not stored in the repository. On the server, the file must exist at:

```text
/srv/tuberculosis-recognition/source/model/model_weights.pth
```

Create the directory if needed:

```bash
mkdir -p /srv/tuberculosis-recognition/source/model
```

Upload from your local machine:

```bash
scp ./model_weights.pth <user>@<server>:/srv/tuberculosis-recognition/source/model/model_weights.pth
```

If the file already exists somewhere else on the server:

```bash
cp /path/to/model_weights.pth /srv/tuberculosis-recognition/source/model/model_weights.pth
```

## 3. Start the application

```bash
cd /srv/tuberculosis-recognition
docker compose up -d --build
```

Useful checks:

```bash
docker compose ps
docker compose logs -f
curl http://127.0.0.1:18501/_stcore/health
```

Expected health response:

```text
ok
```

## 4. Configure host-level Caddy

Copy the site block from `Caddyfile` into `/etc/caddy/Caddyfile` on the server:

```caddyfile
tuberculosis-recognition.pandamia.org {
    encode zstd gzip
    reverse_proxy 127.0.0.1:18501
}
```

If you are migrating from a containerized Caddy setup, stop the old Caddy container before switching host-level Caddy onto ports `80/443`.

Validate and reload:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy
```

## 5. DNS

The subdomain `tuberculosis-recognition.pandamia.org` must resolve to your server IP.

Quick check:

```bash
dig +short A tuberculosis-recognition.pandamia.org
```

## 6. Verify HTTPS

```bash
curl -I https://tuberculosis-recognition.pandamia.org
```

Expected result:

```text
HTTP/2 200
```

## 7. Update the project

```bash
cd /srv/tuberculosis-recognition
git pull
docker compose up -d --build
sudo systemctl reload caddy
```

## 8. Useful commands

```bash
docker compose logs -f
docker compose ps
docker compose restart
sudo journalctl -u caddy -n 100 --no-pager
```

## Notes

- The model weights are mounted into the container and are not baked into the Docker image.
- The app is exposed only on `127.0.0.1:18501`, so it does not conflict with other projects on the server.
- If you change the subdomain or local port, update both `Caddyfile` and `docker-compose.yml`.

# 🚀 Production Deployment Guide

## Production-Ready Features Implemented

### ✅ Backend Enhancements
- **Configuration Management**: Environment-based settings with `.env` file
- **Logging**: Structured logging to files and console
- **Caching**: In-memory prediction caching with TTL
- **Rate Limiting**: Request throttling per IP address
- **Monitoring**: Metrics endpoint for application health
- **Error Handling**: Comprehensive exception handling
- **Security**: CORS, input validation, non-root Docker user
- **Performance**: GZip compression, response time tracking

### ✅ Infrastructure
- **Docker**: Multi-stage build for optimization
- **Docker Compose**: Complete stack orchestration
- **Nginx**: Reverse proxy with SSL support
- **Redis**: Distributed caching (optional)
- **PostgreSQL**: Database for logging/analytics (optional)

### ✅ CI/CD
- **Automated Testing**: Pytest suite with coverage
- **Code Quality**: Linting, formatting, type checking
- **Security Scanning**: Safety and Bandit checks
- **Docker Build**: Automated image building
- **Deployment**: Automated production deployment

---

## 📋 Prerequisites

### Required
- Python 3.11+
- Docker & Docker Compose
- Git
- Domain name (for production)
- SSL certificate (Let's Encrypt recommended)

### Optional
- PostgreSQL (for advanced features)
- Redis (for distributed caching)
- Monitoring tools (Prometheus, Grafana)

---

## 🔧 Setup Instructions

### Step 1: Clone and Configure

```bash
# Clone repository
git clone https://github.com/yourusername/punjab-soil-predictor.git
cd punjab-soil-predictor

# Create environment file
cp .env.example .env

# Edit configuration
nano .env
```

Update `.env` with your settings:
```env
DEBUG=False
ALLOWED_ORIGINS=["https://yourdomain.com"]
MAX_REQUESTS_PER_MINUTE=100
CACHE_TTL_SECONDS=3600
```

### Step 2: Train and Save Models

```bash
# Install dependencies
pip install -r requirements-prod.txt

# Train models
python train_and_save.py

# Verify models saved
ls -lh models/predictor_state.pkl
```

### Step 3: Run Tests

```bash
# Run test suite
pytest test_api.py -v --cov=app

# Security scan
safety check
bandit -r app.py
```

### Step 4: Local Testing

```bash
# Start development server
uvicorn app:app --reload --port 8000

# Test API
curl http://localhost:8000/api/health

# Check interactive docs
open http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Check health
curl http://localhost/api/health

# Stop services
docker-compose down
```

### Option 2: Docker Only

```bash
# Build image
docker build -t soil-predictor:latest .

# Run container
docker run -d \
  --name soil-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  soil-predictor:latest

# View logs
docker logs -f soil-api
```

---

## 🌐 Production Deployment

### AWS EC2 Deployment

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. Configure security groups (ports 80, 443, 22)
# 3. SSH into instance

ssh ubuntu@your-ec2-ip

# 4. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 5. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 6. Clone repository
git clone https://github.com/yourusername/punjab-soil-predictor.git
cd punjab-soil-predictor

# 7. Configure environment
nano .env

# 8. Start services
docker-compose up -d

# 9. Setup SSL with Let's Encrypt
sudo apt-get install certbot
sudo certbot certonly --standalone -d yourdomain.com
```

### Google Cloud Run

```bash
# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/soil-predictor

# 2. Deploy to Cloud Run
gcloud run deploy soil-predictor \
  --image gcr.io/PROJECT_ID/soil-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2

# 3. Get URL
gcloud run services describe soil-predictor --format='value(status.url)'
```

### DigitalOcean App Platform

```bash
# 1. Push code to GitHub
git push origin main

# 2. Connect GitHub repository to DigitalOcean
# 3. Configure build settings:
#    - Build Command: docker build -t soil-predictor .
#    - Run Command: uvicorn app:app --host 0.0.0.0 --port 8000

# 4. Add environment variables in dashboard
# 5. Deploy
```

---

## 📊 Monitoring & Observability

### Health Checks

```bash
# Basic health
curl https://yourdomain.com/api/health

# Detailed metrics
curl https://yourdomain.com/api/metrics
```

### Log Management

```bash
# View application logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f api

# Filter errors only
docker-compose logs api | grep ERROR
```

### Prometheus Metrics (Optional)

Add to `docker-compose.yml`:
```yaml
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

---

## 🔒 Security Best Practices

### 1. SSL/TLS Configuration

```bash
# Generate Let's Encrypt certificate
sudo certbot certonly --nginx -d yourdomain.com

# Update nginx.conf with SSL settings
# Uncomment HTTPS server block
```

### 2. Environment Variables

```bash
# Never commit .env to git
echo ".env" >> .gitignore

# Use secrets management
# AWS: Secrets Manager
# GCP: Secret Manager
# DigitalOcean: App Secrets
```

### 3. API Authentication (Optional)

```python
# Add to app.py
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
```

### 4. Rate Limiting

Already implemented! Configured in `.env`:
```env
MAX_REQUESTS_PER_MINUTE=60
```

---

## 🧪 Testing in Production

### Smoke Tests

```bash
# Health check
curl https://yourdomain.com/api/health

# Test prediction
curl -X POST https://yourdomain.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 31.227389,
    "longitude": 75.766764,
    "depth": 3.0
  }'

# Check metrics
curl https://yourdomain.com/api/metrics
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -T application/json \
  -p payload.json \
  https://yourdomain.com/api/predict
```

---

## 📈 Scaling Strategies

### Horizontal Scaling

```bash
# Docker Compose - multiple workers
docker-compose up -d --scale api=3
```

### Vertical Scaling

Update `docker-compose.yml`:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: soil-predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: soil-predictor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Setup

1. Add secrets to GitHub repository:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
   - `SERVER_HOST`
   - `SERVER_USERNAME`
   - `SSH_PRIVATE_KEY`
   - `PRODUCTION_URL`
   - `SLACK_WEBHOOK` (optional)

2. Push to trigger pipeline:
```bash
git add .
git commit -m "Deploy to production"
git push origin main
```

3. Monitor in GitHub Actions tab

---

## 🐛 Troubleshooting

### Issue: Models not loading

```bash
# Check model file exists
ls -lh models/predictor_state.pkl

# Check permissions
chmod 644 models/predictor_state.pkl

# Check logs
docker-compose logs api | grep "model"
```

### Issue: High latency

```bash
# Enable caching
# In .env:
CACHE_PREDICTIONS=True

# Check cache hit rate
curl https://yourdomain.com/api/metrics
```

### Issue: Out of memory

```bash
# Increase container memory
# In docker-compose.yml:
services:
  api:
    mem_limit: 4g
```

### Issue: Rate limit too strict

```bash
# Adjust in .env
MAX_REQUESTS_PER_MINUTE=200
```

---

## 📞 Support & Maintenance

### Regular Maintenance

```bash
# Update dependencies (monthly)
pip list --outdated
pip install --upgrade -r requirements-prod.txt

# Retrain models (quarterly)
python train_and_save.py

# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Clean up logs (weekly)
find logs/ -name "*.log" -mtime +30 -delete

# Update Docker images
docker-compose pull
docker-compose up -d
```

### Monitoring Checklist

- [ ] API health check passing
- [ ] Response times < 500ms
- [ ] Cache hit rate > 50%
- [ ] Error rate < 1%
- [ ] Disk space available
- [ ] SSL certificate valid

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/)
- [Prometheus Monitoring](https://prometheus.io/)

---

## 🎯 Performance Benchmarks

Target metrics for production:
- **Response Time**: < 500ms (p95)
- **Throughput**: > 100 req/s
- **Uptime**: > 99.9%
- **Error Rate**: < 0.1%
- **Cache Hit Rate**: > 70%

---

**Ready for Production!** 🚀

Your Punjab Soil Predictor API is now production-ready with:
✅ High performance and reliability
✅ Comprehensive monitoring
✅ Automated testing and deployment
✅ Security best practices
✅ Scalability options
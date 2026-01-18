#!/bin/bash

echo "🚀 Deploying Gesture Control Dashboard to Free Cloud Platforms"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📋 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Gesture Control Dashboard"
fi

echo "📦 Preparing deployment files..."

# Create netlify.toml for frontend
cat > Frontend/netlify.toml << 'EOF'
[build]
  command = "npm run build"
  publish = "dist"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[context.production]
  command = "npm run build"

[context.deploy-preview]
  command = "npm run build"
EOF

# Create _redirects for SPA routing
echo "/* /index.html 200" > Frontend/public/_redirects

# Update frontend environment for production
cat > Frontend/.env.production << 'EOF'
VITE_WS_URL=wss://your-backend-app.onrender.com
VITE_API_URL=https://your-backend-app.onrender.com
EOF

echo "✅ Frontend configured for Netlify deployment"

# Backend deployment preparation
cd "Backend(Ml_python)"

# Create startup script for cloud
cat > start.sh << 'EOF'
#!/bin/bash

# Download models if not present
if [ ! -f "yolov8n.pt" ]; then
    echo "📥 Downloading YOLOv8 model..."
    python -c "
import urllib.request
try:
    urllib.request.urlretrieve(
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8n.pt'
    )
    print('✅ Model downloaded successfully')
except Exception as e:
    print(f'❌ Model download failed: {e}')
    "
fi

# Start the WebSocket server
python ws_server_cloud.py
EOF

chmod +x start.sh

echo "✅ Backend configured for cloud deployment"

echo "
🎯 Deployment Instructions:

1. 📤 Push to GitHub:
   git add .
   git commit -m 'Prepare for cloud deployment'
   git push origin main

2. 🌐 Deploy Frontend (Netlify):
   - Go to netlify.com
   - Connect GitHub repository
   - Select 'Frontend' folder
   - Deploy automatically

3. 🖥️ Deploy Backend (Render):
   - Go to render.com
   - Connect GitHub repository
   - Select 'Backend(Ml_python)' folder
   - Use start command: './start.sh'

4. 🔗 Update Frontend URLs:
   - Update VITE_WS_URL in Frontend/.env.production
   - Redeploy frontend

Alternative platforms:
- Railway.app (excellent for ML)
- Fly.io (Docker deployment)
- Cyclic.sh (full-stack)

✅ All files prepared for deployment!
"
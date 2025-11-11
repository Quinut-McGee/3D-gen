module.exports = {
  apps: [
    {
      name: 'trellis-microservice',
      script: '/home/kobe/miniconda3/envs/trellis-env/bin/python',
      args: 'generation/serve_trellis.py --host 0.0.0.0 --port 10008',
      cwd: '/home/kobe/404-gen/v1/3D-gen',
      interpreter: 'none',

      // Reliability settings
      autorestart: true,
      max_restarts: 10,              // Max 10 restarts within min_uptime window
      min_uptime: '30s',             // If crashes within 30s, counts as unstable
      max_memory_restart: '8G',      // Restart if exceeds 8GB RAM (TRELLIS uses ~5-6GB)

      // Restart strategies
      exp_backoff_restart_delay: 100, // Exponential backoff starting at 100ms

      // Logging
      error_file: '/home/kobe/.pm2/logs/trellis-microservice-error.log',
      out_file: '/home/kobe/.pm2/logs/trellis-microservice-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',

      // Environment
      env: {
        SPCONV_ALGO: 'native',
        CUDA_VISIBLE_DEVICES: '0'
      }
    },
    {
      name: 'gen-worker-1',
      script: '/home/kobe/miniconda3/envs/three-gen-mining/bin/python',
      args: 'serve_competitive.py --port 10010 --enable-validation --validation-threshold 0.15 --enable-prompt-enhancement --flux-steps 6',
      cwd: '/home/kobe/404-gen/v1/3D-gen/generation',
      interpreter: 'none',

      autorestart: true,
      max_restarts: 10,
      min_uptime: '30s',
      // max_memory_restart disabled - FLUX + TRELLIS generation can spike >30GB transiently
      // Generations complete successfully; PM2 was killing service unnecessarily

      error_file: '/home/kobe/.pm2/logs/gen-worker-1-error.log',
      out_file: '/home/kobe/.pm2/logs/gen-worker-1-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',

      // Environment
      env: {
        CUDA_VISIBLE_DEVICES: '0,1'  // Both GPUs visible - SD3.5 on GPU 1 (INT8), TRELLIS on GPU 0
      }
    },
    {
      name: 'gen-worker-2',
      script: '/home/kobe/miniconda3/envs/three-gen-mining/bin/python',
      args: '-m uvicorn serve_competitive:app --host 0.0.0.0 --port 10011 --workers 1',
      cwd: '/home/kobe/404-gen/v1/3D-gen/generation',
      interpreter: 'none',

      autorestart: true,
      max_restarts: 10,
      min_uptime: '30s',
      max_memory_restart: '6G',

      error_file: '/home/kobe/.pm2/logs/gen-worker-2-error.log',
      out_file: '/home/kobe/.pm2/logs/gen-worker-2-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    },
    {
      name: 'gen-worker-3',
      script: '/home/kobe/miniconda3/envs/three-gen-mining/bin/python',
      args: '-m uvicorn serve_competitive:app --host 0.0.0.0 --port 10012 --workers 1',
      cwd: '/home/kobe/404-gen/v1/3D-gen/generation',
      interpreter: 'none',

      autorestart: true,
      max_restarts: 10,
      min_uptime: '30s',
      max_memory_restart: '6G',

      error_file: '/home/kobe/.pm2/logs/gen-worker-3-error.log',
      out_file: '/home/kobe/.pm2/logs/gen-worker-3-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
  ]
};
